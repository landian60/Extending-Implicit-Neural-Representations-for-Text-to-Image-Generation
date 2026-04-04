# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, upfirdn2d, conv2d_resample
from torch_utils.op import upfirdn2d as upfirdn2d_op
from training.attn_layer import TransformerLayer, EqualLinear, TransformerBlock, MLP
from training.attn_layer import normalize, torch_misc, get_global, get_components
from training.attn_layer import DFBLK, get_positional_encoding
from training.positional_encoding import ScalePartyPE, simple_norm, decompose_simple_2d, compose_simple_2d, decompose_simple_1d, compose_simple_1d
import dnnlib
import legacy

# ----------------------------------------------------------------------------
SLOPE = 0.2

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_dim, in_channel, out_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.linear1 = EqualLinear(in_channel, out_channel)
        self.linear2 = EqualLinear(in_channel, out_channel)

    def forward(self, input, style):
        size = input.size()
        style = self.style(style).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)
        gamma = gamma.expand(size)
        beta = beta.expand(size)
        out = self.norm(input)
        out = gamma * out + beta
        out = out.transpose(1, -1)
        input = input.transpose(1, -1)
        input = self.linear1(input)
        out = self.linear2(out)
        out = out + input
        out = out.transpose(1, -1)
        return out
    
@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

# ----------------------------------------------------------------------------


@misc.profiled_function
def modulated_fc(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    demodulate=True,  # Apply weight demodulation?
):
    batch_size = x.shape[0]
    out_channels, in_channels = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels])  # [OI]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels) / weight.norm(float("inf"), dim=[1], keepdim=True))  # max_I
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [1OI]
    w = w * styles.unsqueeze(1)  # [1OI] * [N1I] = [NOI]

    if demodulate:
        dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2)  # [NOI]

    x = torch.einsum("nihw, noi -> nohw", x, w.to(x.dtype))
    return x

'''FFT-based spectral modulation variant used in the released EIDGAN code path.'''
@misc.profiled_function
def modulated_spectral_fc_new(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels].
    styles2,
    demodulate=True,  # Apply weight demodulation?
):
    batch_size, in_channels = x.shape[0], x.shape[1]
    out_channels_complex, _ = weight.shape
    ori_dtype = x.dtype
    x = x.to(torch.float32)
    dtype = x.dtype
    misc.assert_shape(weight, [out_channels_complex, in_channels + 2])  # [OI]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles2, [batch_size, in_channels + 2])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    # if x.dtype == torch.float16 and demodulate:
    #     weight = weight * (1 / np.sqrt(in_channels) / weight.norm(float("inf"), dim=[1], keepdim=True))  # max_I
    
    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOI]
    
    w = w * styles2.unsqueeze(1)  # [NOI]
    
    if demodulate:
        dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2)  # [NOI]
        # dcoefs = (styles2.square().sum(dim=[1]) + 1e-8).rsqrt()  # [I]
        # styles2 = styles2 * dcoefs.unsqueeze(1)  # [NI]
    
    # calculate spectral frequency
    # styles2 = styles2.to(torch.float32)
    # styles2 = styles2.view(batch_size, in_channels // 2 + 1, 2).unsqueeze(-2).unsqueeze(-2)
    
    w = w.reshape(batch_size, out_channels_complex, in_channels // 2 + 1, 2)
    
    x = torch.fft.rfft(x, dim=1, norm='ortho')
    
    x = torch.einsum("nihw, noi -> nohw", x,torch.view_as_complex(w))
    # x = x * torch.view_as_complex(styles2)
    x = torch.fft.irfft(x, dim=1, norm='ortho')
    
    # x = x.to(ori_dtype)
    
    
    # w = torch.view_as_complex(w)
    # x = torch.einsum("nihw, noi -> nohw", x, w)
    # out_channels_complex = (ori_out_channels - 1) * 2
    # x = torch.fft.irfft(x, dim=1, norm='ortho')
    x = x.to(ori_dtype)
    
    return x

@misc.profiled_function
def modulated_spectral_fc_frequency(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    # weight,  # Weight tensor of shape [out_channels, in_channels].
    # weight2,
    styles2,
    h,
    demodulate=False,  # Apply weight demodulation?
    norm="instance",
    thres = 0.1,
    # norm=None, 
):
    batch_size, in_channels,w,_ = x.shape
    # out_channels, _,_ = weight2.shape
    ori_dtype = x.dtype
    # misc.assert_shape(weight, [in_channels])  # [OI]
    # misc.assert_shape(weight2, [out_channels,in_channels,2])  # [OI]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles2, [batch_size, in_channels])  # [NI]
    x_norm = x.clone()
    lam = F.softmax(h / thres, dim=0)
    
    # Pre-normalize inputs to avoid FP16 overflow.
    # if x.dtype == torch.float16 and demodulate:
    #     weight = weight * (1 / np.sqrt(in_channels) / weight.norm(float("inf"), dim=[1], keepdim=True))  # max_I
    # w = weight2.unsqueeze(0)  # [NOI]
    # style1, style2 = styles2[:,0:in_channels//2 +1],styles2[:,in_channels//2 +1: in_channels//2 +1 + in_channels]
    # style2 = style2.view(batch_size,in_channels,2).unsqueeze(-2).unsqueeze(-2)
    
    # style1, style2 = torch.split(styles2,[in_channels,in_channels],dim=1)
    # style2 = style2.view(batch_size,in_channels).unsqueeze(-1).unsqueeze(-1)
    # style1 = style1[:,:in_channels // 2 + 1].unsqueeze(-1).unsqueeze(-1)
    # style1 = styles2[:,:in_channels + 2]
    # style1 = style1.view(batch_size,in_channels // 2 + 1,2).unsqueeze(-2).unsqueeze(-2)
    # style2 = style2.unsqueeze(-1).unsqueeze(-1)
    
    #new
    # style2 = style2.view(batch_size,in_channels // 2 + 1,2).unsqueeze(-2).unsqueeze(-2)
    # if demodulate:
    #     dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()  # [NO]
    #     w = w * dcoefs.unsqueeze(2)  # [NOI]
        # dcoefs = (style2.square().sum(dim=[1]) + 1e-8).rsqrt()  # [NO]
        # style2 = style2 * dcoefs.unsqueeze(1)  # [NO]
        # dcoefs = (style1.square().sum(dim=[1]) + 1e-8).rsqrt()  # [NO]
        # style1 = style1 * dcoefs.unsqueeze(1)  # [NO]
    # w = w * style2.unsqueeze(1).unsqueeze(-1)  # [NOI2]
    # x_ori = x_ori * style1
    
    styles2 = styles2.unsqueeze(-1).unsqueeze(-1)
    x_norm = x_norm * styles2
    x_norm = simple_norm(x_norm,norm)
    x = x.to(torch.float32)
    x_norm = x_norm.to(torch.float32)
    # style2 = style2.to(torch.float32)
    # style1 = style1.to(torch.float32)
    # w = weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(x.shape[0],1, 1, 1)  # [NIwr2]
    # w = w * styles2.view(batch_size, in_channels, 2).unsqueeze(-2).unsqueeze(-2)  # [NOI]
        
    # # Normalize styles2 to prevent excessive amplification of frequency components
    # styles2 = styles2 / (styles2.norm(p=2, dim=1, keepdim=True) + 1e-8)
    
    # calculate spectral frequency
    # styles2 = styles2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).reshape(batch_size, in_channels,1,1,2).to(torch.float32)
    
    # need explore
    xn_pha, xn_amp  = decompose_simple_2d(x_norm)
    x_pha, x_amp  = decompose_simple_2d(x)
    x_amp_new = lam[0] * x_amp + lam[1] * xn_amp
    x_new = compose_simple_2d(x_pha,x_amp_new)
    
    # x2_pha, x2_amp = decompose_simple_1d(x_amp)
    # style2_pha, style2_amp = decompose_simple_1d(style2)
    # style2_new = torch.fft.rfft(style2.to(torch.float32), dim=1, norm='ortho')
    # xn_amp_new = xn_amp * style2 + x_amp
    # x_amp_new = compose_simple_1d(x2_pha,x2_amp_new)
    # x_amp_new = torch.fft.irfft(x_amp_new, dim=1, norm='ortho')
    # x3_pha, x3_amp = decompose_simple_1d(x2_amp)
    # xori_amp_new = xori_amp * style2
   
    # x3_pha = x3_pha * style1
    # x2_amp_new = compose_simple_1d(x3_pha,x3_amp)
    # x2_amp_new = x2_amp * style2
    

    # 2
    # x = torch.fft.rfft2(x, dim=(2,3), norm='ortho')
    # x = torch.fft.rfft(x, dim=1, norm='ortho')
    
    # # x = torch.einsum("nihw, noi -> nohw", x,torch.view_as_complex(w))
    # x = x * torch.view_as_complex(style1)
    # x_new = torch.fft.irfft(x, dim=1, norm='ortho')
    # print("x_norm_after",torch.mean(x_norm),torch.max(x_norm),torch.min(x_norm))
    
    x_new = x_new.to(ori_dtype)
    
    # x_real = torch.real(x)
    # x_imag = torch.imag(x)
    # x_real = x_real * style2
    # m = nn.Tanhshrink()
    # x_real = m(torch.real(x))
    # x_imag = m(torch.imag(x))
    # x = torch.complex(x_real, x_imag)
    
    # Calculate per-sample weights and demodulation coefficients.
    # w = weight.unsqueeze(0)  # [NOI]
    
    
    # w = w * styles2.unsqueeze(1)  # [NOI]

    # x = x.to(ori_dtype)
    
    # w = w.reshape(batch_size, out_channels_complex, in_channels // 2 + 1, 2)
    # w = torch.view_as_complex(w)
    # x = torch.einsum("nihw, noi -> nohw", x, styles2)
    # out_channels_complex = (ori_out_channels - 1) * 2
    
    
    # x = simple_norm(x,norm)

    # print("x_norm_after2",torch.mean(x_norm),torch.max(x_norm),torch.min(x_norm))
    # x = x_freq * x + w * style1
    # x = x
    return x_new

@misc.profiled_function
def modulated_spectral_fc(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight1,  # Weight tensor of shape [out_channels, in_channels].
    # weight2,
    styles,
    styles2,
    demodulate=True,  # Apply weight demodulation?
):
    batch_size = x.shape[0]
    out_channels, in_channels = weight1.shape
    dtype = x.dtype
    misc.assert_shape(weight1, [out_channels, in_channels])  # [OI]
    # misc.assert_shape(weight2, [1, in_channels*2])  # [OI]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    # misc.assert_shape(complex_weight, [1, in_channels, 1, None, 2])  # [NI1W]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]
    # misc.assert_shape(styles2, [batch_size, in_channels * 2])  # [NI]
    misc.assert_shape(styles2, [batch_size, in_channels + 2])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight1 = weight1 * (1 / np.sqrt(in_channels) / weight1.norm(float("inf"), dim=[1], keepdim=True))  # max_I
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I
    
    # calculate spectral frequency
    # weight2 = weight2.repeat(batch_size,1)
    # styles2 = weight2 * styles2
    styles2 = styles2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).reshape(batch_size, in_channels // 2 + 1, 1, 1, 2).to(torch.float32)

    # x = torch.fft.rfft2(x.to(torch.float32), dim=[2, 3], norm='ortho')
    x = torch.fft.rfft(x.to(torch.float32), dim=1, norm='ortho')
    
    # x_amplitude = torch.abs(x)
    # x_phase = torch.angle(x)
    # x_amplitude *= styles2

    # styles2 = styles2.view(batch_size, in_channels,1, 1, 2)
    x = x * torch.view_as_complex(styles2)
    # x = torch.fft.irfft2(x, dim=[2, 3], norm='ortho')
    x = torch.fft.irfft(x, dim=1, norm='ortho')
    # x = torch.fft.irfft2(torch.abs(x) * styles2 * torch.exp(1j * torch.angle(x)), dim=[2, 3], norm='ortho')
    # del x_amplitude, x_phase
    x = x.to(dtype)
    
    # Calculate per-sample weights and demodulation coefficients.
    w = weight1.unsqueeze(0).repeat(x.shape[0],1, 1)  # [NOI]
    w = w * styles.unsqueeze(1)  # [NOI]

    if demodulate:
        dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2)  # [NOI]

    x = torch.einsum("nihw, noi -> nohw", x, w.to(x.dtype))
    return x

# ----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        weight_norm=True,  # Apply weight normalization?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / (np.sqrt(in_features) if weight_norm else 1)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f"in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}"


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim             = 512,      # Input latent (Z) dimensionality, 0 = no latent
        c_dim             = 0,        # Conditioning label (C) dimensionality, 0 = no label
        w_dim             = 512,      # Intermediate latent (W) dimensionality
        fts_dim           = 512,
        k                 = 1,        # Number of latent vector components z_1,...,z_k
        num_broadcast     = None,     # Number of intermediate latents to output, None = do not broadcast
        num_layers        = 8,        # Number of mapping layers
        embed_dim         = None,     # Label embedding dimensionality, None = same as w_dim
        layer_dim         = None,     # Number of intermediate channels in the mapping layers, None = same as w_dim
        act               = "lrelu",  # Activation function: "relu", "lrelu", etc
        lrmul             = 0.01,     # Learning rate multiplier for the mapping layers
        w_avg_beta        = 0.995,    # Decay for tracking the moving average of W during training, None = do not track
        # GANformer-related options
        transformer       = True,    # Whereas the generator uses attention or not (i.e. GANformer or StyleGAN)
        resnet            = False,    # Use resnet connections in mapping network
        shared            = False,    # Perform a shared mapping to all latents concatenated together
        ltnt2ltnt         = False,    # Add self-attention over latents in the mapping network
        ltnt_gate         = False,    # Gate attention scores so that info may not be sent/received when value is low
        normalize_global  = True,     # Normalize the input global latent 
                                      # (True is recommended. The flag is introduced for backwards compatibility)
        use_pos           = False,    # Use positional encoding for latents
        **transformer_kwargs          # Arguments for Attn
    ):

        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.fts_dim = fts_dim
        self.w_dim = w_dim
        self.k = k
        self.num_broadcast = num_broadcast
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.normalize_global = normalize_global
        self.use_pos = use_pos

        layer_dim = layer_dim or w_dim
        embed_dim = (embed_dim or z_dim) if (c_dim > 0 or fts_dim > 0) else 0

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_dim)
        if fts_dim > 0:
            self.fsts_embed = FullyConnectedLayer(fts_dim, embed_dim, activation='lrelu', lr_multiplier=0.01)
            # self.fsts_embed = FullyConnectedLayer(fts_dim+z_dim, embed_dim, activation='lrelu', lr_multiplier=0.01)
            if k != 1:
                ta_kwargs = {
                    "from_len":  77,     "to_len":  77,         # The from/to tensor lengths
                    "from_dim":  fts_dim,                "to_dim":  fts_dim,         # The from/to tensor dimensions
                    }
                ca_kwargs = {
                    "from_len":  self.k - 1,     "to_len":  77,         # The from/to tensor lengths
                    "from_dim":  w_dim,                "to_dim":  fts_dim,         # The from/to tensor dimensions
                    }
                ta_kwargs.update(transformer_kwargs)
                ca_kwargs.update(transformer_kwargs)
                self.text_attn = TransformerLayer(dim = fts_dim, modulate = False, **ta_kwargs)
                # self.cross_attn = TransformerLayer(dim = w_dim, modulate = False, **ca_kwargs)
                self.cross_attn = TransformerLayer(dim = w_dim, **ca_kwargs)
            # self.text_attn = TransformerLayer(dim = fts_dim, pos_dim=fts_dim, **ta_kwargs)
            # self.cross_attn = TransformerLayer(dim = w_dim, pos_dim=w_dim,**ca_kwargs)

        sa_kwargs = {
           "sa": ltnt2ltnt and not shared,   "batch":   not shared,
           "from_len":  k - 1,               "to_len":  k - 1,
           "from_gate": ltnt_gate,           "to_gate": ltnt_gate,
        }
        sa_kwargs.update(transformer_kwargs)

        layers = [layer_dim] * (num_layers - 1) + [w_dim]

        self.global_mlp = MLP([z_dim] * (num_layers + 1), act = act, resnet = False, lrmul = lrmul)
        # self.global_mlp = MLP([z_dim + embed_dim] + layers, act = act, resnet = False, lrmul = lrmul)
        if transformer:
            self.mlp = MLP([z_dim] + layers, act = act, resnet = resnet, lrmul = lrmul, **sa_kwargs)

        if num_broadcast is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

        # GANformer-related setup
        self.transformer = transformer

    def forward(self, z, c, fsts=None, fts=None, pos = None, mask = None, truncation_psi = 1, truncation_cutoff = None, update_emas = False):
        # Embed, normalize, and concatenate inputs
        with torch.autograd.profiler.record_function("input"):
            torch_misc.assert_shape(z, [None, self.k, self.z_dim])
            if self.transformer:
                z, g = torch.split(z, [self.k - 1, 1], dim = 1)
                if self.normalize_global: 
                    g = normalize(g)
                    
            z = normalize(z)
            x = g if self.transformer else z
            
            if self.c_dim > 0:
                torch_misc.assert_shape(c, [None, self.c_dim])
                y = self.embed(c.to(torch.float32))
                y = normalize(y)
                x = torch.cat([x, y.unsqueeze(1)], dim = -1)


        if mask is not None:
            torch_misc.assert_shape(mask, [None, self.k - 1])
        if pos is not None:
            torch_misc.assert_shape(pos, [self.k - 1, self.w_dim])

        # Main layers
        x = self.global_mlp(x)
        if self.fts_dim > 0 and self.transformer:
            misc.assert_shape(fsts, [None,self.fts_dim])
            y = self.fsts_embed(fsts.to(torch.float32))
            # y = self.fsts_embed(torch.cat([x.squeeze(1), fsts.to(torch.float32)], dim = -1))
            y = normalize(y)
            # x = normalize(y).unsqueeze(1)
            x = torch.cat([x, y.unsqueeze(1)], dim = -1)
        if self.transformer:
            p = self.mlp(z, pos = pos if self.use_pos else None, mask = mask)
            if fts is not None and self.fts_dim > 0:
                torch_misc.assert_shape(fts, [None, 77, self.fts_dim])
                fts_,_,_ = self.text_attn(from_tensor = fts,  to_tensor = fts, from_pos=None,to_pos=None)
                p,_,_ = self.cross_attn(from_tensor = p,  to_tensor = fts_, from_pos=None,to_pos=None)
                # z = torch.cat([z,fts],dim=1)
            x = torch.cat([p, x], dim = 1)

        # Update moving average of W
        if self.w_avg_beta is not None and self.training and not update_emas:
            with torch.autograd.profiler.record_function("update_w_avg"):
                if self.transformer:
                    self.w_avg.copy_(x.detach().mean(dim = (0, 1)).lerp(self.w_avg, self.w_avg_beta))
                else:
                    # The size of tensor a (64) must match the size of tensor b (128) at non-singleton dimension 1
                    self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
        # Broadcast
        if self.num_broadcast is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(2).repeat([1, 1, self.num_broadcast, 1])

        # Apply truncation
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_broadcast is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :, :truncation_cutoff] = self.w_avg.lerp(x[:, :, :truncation_cutoff], truncation_psi)
        return x
    
@persistence.persistent_class
class MappingNetwork_old(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        fts_dim,
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.998,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.fts_dim = fts_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0 and fts_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        if fts_dim > 0:
            self.fsts_embed = FullyConnectedLayer(fts_dim, embed_features)
            # self.self_attn = CrossAttention2()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(self, z, c, fsts=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y
            if self.fts_dim > 0:
                misc.assert_shape(fsts, [None,self.fts_dim])
                z = normalize_2nd_moment(self.fsts_embed(fsts.to(torch.float32)))
                x = torch.cat([x, z], dim=1) if x is not None else z

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not update_emas:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f"z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}"


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,
        k,
        attn = False,
        rank = 8,
        poly=False,
        mod_fc = True,
        fts_dim = None,
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        fc_clamp=None,  # Clamp the output of fully connected layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
        ltnt_gate = False,        # Gate attention from latents, such that components may not send information
                                        # when gate value is low
        img_gate = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
        res_channels = 32,
        channel_attn = False,
        spectral_weight = False,
        rank_attn = False,
        **transformer_kwargs          # Arguments for Attn
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.activation = activation
        self.fc_clamp = fc_clamp
        self.fts_dim = fts_dim
        self.resolution = resolution
        self.attn = attn
        self.poly = poly
        self.mod_fc = mod_fc
        self.channel_attn = channel_attn
        self.spectral_weight = spectral_weight
        self.rank_attn = rank_attn

        if fts_dim is not None and not poly:
            self.text_affine = FullyConnectedLayer(fts_dim, in_channels, activation='lrelu', lr_multiplier=0.01)
            self.text_affine2 = FullyConnectedLayer(in_channels, in_channels, activation='lrelu', lr_multiplier=0.01)
            
            self.text_ffc = FullyConnectedLayer(2*in_channels,in_channels,bias_init=1)

        if attn:
            if res_channels != 3:
                if channel_attn:
                    kwargs = {
                    "from_len":  in_channels,               "to_len":  k - 1,         # The from/to tensor lengths
                    "from_dim":  self.resolution * rank * 2, "to_dim":  w_dim,         # The from/to tensor dimensions
                    "from_gate": img_gate,                    "to_gate": ltnt_gate,     # Gate attention flow between from/to tensors
                    }
                elif rank_attn:
                    kwargs = {
                    # "from_len":  self.resolution * rank * 2,     "to_len":  k - 1,         # The from/to tensor lengths
                    "from_len":  rank * 2,     "to_len":  k - 1,         # The from/to tensor lengths
                    "from_dim":  in_channels,                "to_dim":  w_dim,         # The from/to tensor dimensions
                    "from_gate": img_gate,                    "to_gate": ltnt_gate,     # Gate attention flow between from/to tensors
                    }
                else:
                    kwargs = {
                    "from_len":  self.resolution * rank * 2,     "to_len":  k - 1,         # The from/to tensor lengths
                    "from_dim":  in_channels,                "to_dim":  w_dim,         # The from/to tensor dimensions
                    "from_gate": img_gate,  
                    }
            else:
                kwargs = {
                    "from_len":  resolution * resolution,       "to_len":  k - 1,         # The from/to tensor lengths
                    "from_dim":  in_channels,                "to_dim":  w_dim,         # The from/to tensor dimensions
                    "from_gate": img_gate,                    "to_gate": ltnt_gate,     # Gate attention flow between from/to tensors
                }
                
            # transformer_kwargs["pos_dim"] = out_channels
            # grid_pos, params = get_positional_encoding(resolution, rank=rank, **transformer_kwargs)
            # # Register positional encoding params/buffers
            # if len(params) > 0:
            #     self.grid_pos = grid_pos
            #     for i, param in enumerate(params):
            #         setattr(self, f"pos{i}", param)
            # else:
            #     self.register_buffer("grid_pos", grid_pos)
            # self.affine = FullyConnectedLayer(resolution*rank*2, resolution*2, activation='lrelu', lr_multiplier=0.01)
            # self.affine2 = FullyConnectedLayer(resolution*2, resolution*rank*2, activation='lrelu', lr_multiplier=0.01)
            kwargs.update(transformer_kwargs)
            # self.transformer = TransformerLayer(dim = in_channels, modulate = False, **kwargs)
            self.transformer = TransformerLayer(dim = in_channels, **kwargs)
            # self.ffc = FullyConnectedLayer(out_channels,out_channels,activation='lrelu')
            # self.ff = torch.nn.Sequential(
            #     torch.nn.GELU(),
            #     torch.nn.Linear(out_channels, out_channels),
            #     torch.nn.GELU(),
            #     torch.nn.Linear(out_channels, out_channels),
            # )

        if not poly:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            
            if self.mod_fc:
                memory_format = torch.channels_last if channels_last else torch.contiguous_format
                if spectral_weight:
                    # self.affine_spectral = FullyConnectedLayer(w_dim, in_channels + 2, bias_init=1)
                    # self.weight1 = torch.nn.Parameter(torch.randn([out_channels, in_channels]).to(memory_format=memory_format))
                    
                    if self.spectral_weight == 'channel':
                        # fft1 方法
                        self.fc_new_weight = torch.nn.Parameter(torch.randn([out_channels // 2 + 1, in_channels + 2]).to(memory_format=memory_format))
                        self.affine_spectral_new = FullyConnectedLayer(w_dim, in_channels + 2)
                    elif self.spectral_weight == 'amplitude':
                        # ffc2 方法
                        self.affine_spectral = FullyConnectedLayer(w_dim, in_channels)
                        self.h = nn.Parameter(torch.zeros(2, 1, 1))
                        # self.weight = torch.nn.Parameter(torch.randn([in_channels], dtype=torch.float32).to(memory_format=memory_format))
                    # # self.weight2 = torch.nn.Parameter(torch.randn([out_channels, in_channels, 2]).to(memory_format=memory_format))
                    # self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
                    
                    
                    # self.affine = FullyConnectedLayer(w_dim, in_channels * 2, bias_init=1)
                    # 
                else:
                    # self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
                    self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]).to(memory_format=memory_format))
                # if spectral_weight:

                #     self.complex_weight = torch.nn.Parameter(torch.randn([1, in_channels, 1, rank + 1, 2], dtype=torch.float32) * 0.02)
                    
            else:
                self.adain = AdaptiveInstanceNorm(in_channels, in_channels, out_channels)
                # self.df = DFBLK(in_channels, in_channels, out_channels)
        else:
            self.poly_layer = PolyLayer(in_channels,out_channels,w_dim,rank)
        
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

    def forward(self, x, w, grid, fsts=None, noise_mode="random", gain=1):
        assert noise_mode in ["random", "const", "none"]  # unused
        shape = x.shape
        res_factor = 1.0
        
        if self.attn:
            # if x.shape[2] != self.resolution:
            #     # 没用上
            #     res_factor = self.resolution / x.shape[2]
            #     x = F.interpolate(x, scale_factor=(1,res_factor), mode='bilinear')
            if not self.poly:
                
                if not self.channel_attn:
                    if self.rank_attn:
                        x = x.permute(0, 2, 3, 1) # b,c,r,2rank-> b,r,2rank,c
                    else:
                        x = x.reshape(shape[0], self.in_channels, -1) # b,c,r,2rank -> b,c,r*2rank
                        x = x.permute(0, 2, 1) # b,c,r*2rank-> b,r*2rank,c
                    
                x_,_,_ = self.transformer(
                    from_tensor = x,  to_tensor = get_components(w.to(x.dtype)),
                    # from_pos=self.grid_pos,
                    from_pos=None,
                    to_pos=None
                )
                if not self.channel_attn:
                    if self.rank_attn:
                        x = x_.permute(0, 3, 1, 2)
                    else:
                        x = x_.permute(0, 2, 1)
                else:
                    x = x_    

                x = x.reshape(shape[0],self.in_channels,shape[2],shape[3])
            else:
                x_,_,_ = self.transformer(
                    from_tensor = x,  to_tensor = get_components(w.to(x.dtype)),
                    from_pos=None,to_pos=None
                )
                # b,R*rank*2,C -> b,C,R,rank*2
                x = x_.permute(0, 2, 1)
                x = x.reshape(shape[0],self.in_channels,self.resolution,-1)
                
            if res_factor != 1: # not used
                res_factor_r = 1 / res_factor
                x  = F.interpolate(x, scale_factor=(1,res_factor_r), mode='bilinear')

        # 降维
        if not self.poly:
            styles = self.affine(get_global(w))
            if self.fts_dim is not None and fsts is not None:
                fsts = self.text_affine(fsts)
                fsts = self.text_affine2(fsts)
                styles = self.text_ffc(torch.cat([fsts,styles], dim=-1))
            if self.mod_fc:
                if self.spectral_weight:
                    # styles = self.affine(get_global(w))
                    if self.fts_dim is not None and fsts is not None:
                        if self.spectral_weight == 'amplitude':
                            # ffc2 
                            styles2 = self.affine_spectral(styles)
                        elif self.spectral_weight == 'channel':
                            # ffc1 方法
                            fc_new_styles2 = self.affine_spectral_new(styles)
                        else:
                            print("spectral_weight type error!")
                    else:
                        if self.spectral_weight == 'amplitude':
                            # ffc2 
                            styles2 = self.affine_spectral(get_global(w))
                        elif self.spectral_weight == 'channel':
                            # ffc1 方法
                            fc_new_styles2 = self.affine_spectral_new(get_global(w))
                        else:
                            print("spectral_weight type error!")
                        
                    # x = modulated_spectral_fc(x=x, weight1=self.weight1, styles=styles, styles2=styles2)
                    if self.spectral_weight == 'channel':
                        # fft1方法
                        x = modulated_spectral_fc_new(x=x, 
                                                  weight=self.fc_new_weight,
                                                  styles2=fc_new_styles2)
                    elif self.spectral_weight == 'amplitude':
                        # fft2方法
                        x = modulated_spectral_fc_frequency(x=x, 
                                                            h = self.h,
                                                            # weight=self.weight, weight2=self.weight2,
                                                            styles2=styles2)
                else:
                    x = modulated_fc(x=x, weight=self.weight, styles=styles)
            else:
                x = self.adain(x, styles)
                # x = self.df(x, styles)
        else:
            # B,C,R,2*rank -> B,R*2*rank,C
            x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1)
            x, gird_new = self.poly_layer(x,get_global(w),grid)
        
                    
        act_gain = self.act_gain * gain
        act_clamp = self.fc_clamp * gain if self.fc_clamp is not None else None
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        if not self.poly:
            return x
        else:
            return x, gird_new

    def extra_repr(self):
        return " ".join(
            [
                f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d},",
                f"w_dim={self.w_dim:d}, activation={self.activation:s}",
            ]
        )


# ----------------------------------------------------------------------------
class PolyLayer(torch.nn.Module):
    def __init__(
        self, 
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution for input layer.
        rank=8,  # Rank of output biline.
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.w_dim = w_dim
        # self.w_dim1 = channels
        self.rank = rank

        coords = torch.linspace(0, 1, resolution).reshape(1, -1, 1, 1)
        coords = coords.repeat(1, 1, 2, 1)
        self.register_buffer("coords", coords)
        
        #  lelvel-0
        self.fc0_1 = nn.Linear(self.w_dim, 2*self.in_channels)
        self.fc0_2 = nn.Linear(self.w_dim, self.in_channels)
        self.fc0_0 = nn.Linear(self.in_channels, self.in_channels)
        self.fc0_3 = nn.Linear(self.w_dim, 2*2)
        self.fc0_4 = nn.Linear(self.w_dim, 2)
        self.fc0_5 = FullyConnectedLayer(self.w_dim, self.w_dim, activation='lrelu', lr_multiplier=1.0)
        self.fc1 = nn.Linear(in_channels, out_channels)

    def forward(self, x, w, grid):
        xy1 = self.fc0_5(w)
        xy_b =  (self.fc0_2(xy1)).view(-1, 1,self.in_channels)
        xy_ = (self.fc0_1(xy1)).view(-1, 2, self.in_channels)
        xy_b1 =  self.fc0_4(xy1).view(-1, 1,2)
        xy_1 = self.fc0_3(xy1).view(-1,2 , 2)
        grid = torch.matmul( grid, xy_1) + xy_b1
        g1 = torch.matmul(grid, xy_) + xy_b
        x =  x*F.leaky_relu(g1, negative_slope=SLOPE)
        x = F.leaky_relu((self.fc1(x)), negative_slope=SLOPE)
        return x, grid

class PolyInput(torch.nn.Module):
    def __init__(
        self, 
        channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution for input layer.
        rank=8,  # Rank of output biline.
        ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.w_dim = w_dim
        self.w_dim1 = channels
        self.rank = rank

        coords = torch.linspace(0, 1, resolution).reshape(1, -1, 1, 1)
        coords = coords.repeat(1, 1, 2, 1)
        self.register_buffer("coords", coords)
        
        #  lelvel-0
        self.fc0_1 = nn.Linear(self.w_dim, 2*self.w_dim1)
        self.fc0_2 = nn.Linear(self.w_dim, self.w_dim1)
        self.fc0_0 = nn.Linear(self.w_dim1, self.w_dim1)
        self.fc0_3 = nn.Linear(self.w_dim, 2*2)
        self.fc0_4 = nn.Linear(self.w_dim, 2)
        self.fc0_5 = FullyConnectedLayer(self.w_dim, self.w_dim, activation='lrelu', lr_multiplier=1.0)

    def forward(self, w, coords=None):
        if coords is not None:
            # B,R*2*rank,2
            grid = coords.repeat(w.shape[0], self.rank * 2, 1, 1).squeeze(-1).to(w.device)
        else:
            grid = self.coords
        xy1 = self.fc0_5(get_global(w))
        xy_b =  (self.fc0_2(xy1)).view(-1, 1,self.w_dim1)
        xy_ = (self.fc0_1(xy1)).view(-1, 2, self.w_dim1)
        xy_b1 =  self.fc0_4(xy1).view(-1, 1,2)
        xy_1 = self.fc0_3(xy1).view(-1,2 , 2)
        grid = torch.matmul( grid, xy_1) + xy_b1
        g1 = torch.matmul(grid, xy_) + xy_b
        x = F.leaky_relu((self.fc0_0(g1)), negative_slope=SLOPE)
        # b,R*rank*2,C -> b,C,R,rank*2
        x = x.permute(0, 2, 1)
        x = x.reshape(w.shape[0],self.w_dim1,self.resolution,-1)
        return x, grid
        

class SynthesisInput(torch.nn.Module):
    def __init__(
        self,
        channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution for input layer.
        attn = False,
        rank=8,  # Rank of output biline.
        bandwidth=10,  # Bandwidth of fourier feature.
        spectral_weight = False, 
    ):
        super().__init__()
        self.channels = channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.rank = rank
        self.bandwidth = bandwidth
        self.attn = attn
        self.spectral_weight = spectral_weight
        # 归一化的频率用于控制正弦函数在一个周期内的变化
        freqs = torch.randn([1, channels // 2 * rank]) * bandwidth
        coords = torch.linspace(0, 1, resolution).reshape(1, -1, 1, 1)
        coords = coords.repeat(1, 1, 2, 1)

        self.register_buffer("freqs", freqs)
        self.register_buffer("coords", coords)

        if attn:
            # self.transformer = TransformerLayer(dim = channels,pos_dim=w_dim,
            self.transformer = TransformerLayer(dim = channels,
                                                from_len=resolution*rank*2,to_len=1,
                                                from_dim=channels,to_dim=w_dim)
            
        else:
            # if not self.spectral_weight:
            self.weight = torch.nn.Parameter(torch.randn([1, channels, 1, 2 * rank]))
            # else:
                # self.complex_weight = torch.nn.Parameter(torch.randn([1, channels, 1, rank + 1, 2], dtype=torch.float32) * 0.02)
            self.affine = FullyConnectedLayer(w_dim, channels * 2 * rank, bias_init=1)

    def forward(self, w, coords=None):
        if coords is None:
            x = self.coords.repeat(w.shape[0], 1, 1, 1) @ self.freqs
        else:
            # coords = nn.Parameter(coords, requires_grad=False)
            coords = coords.to(w.device)
            x = coords.repeat(w.shape[0], 1, 1, 1) @ self.freqs
        x = x.reshape(*x.shape[:2], 2, self.channels // 2, self.rank)

        x = x.permute(0, 3, 1, 2, 4) * (np.pi * 2)
        x = x.reshape(*x.shape[:3], -1)

        x = torch.cat([x, x + np.pi / 2], dim=1)
        x = torch.sin(x)

        if self.attn:
            shape = x.shape
            x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1) # b,c,r,rank -> b,r*rank,c
            # w = w.unsqueeze(1)
            x,_,_ = self.transformer(x,get_components(w),from_pos=None,to_pos=None)
            x = x.permute(0, 2, 1).reshape(shape[0],self.channels,shape[2],shape[3])
            
        else:
            # if self.spectral_weight:
            #     weight = self.complex_weight
            # else:
            weight = self.weight * self.affine(get_global(w)).reshape(w.shape[0], self.channels, 1, -1)
            x = x * weight / np.sqrt(self.channels)

        return x

    def extra_repr(self):
        return " ".join(
            [
                f"channels={self.channels:d}, w_dim={self.w_dim:d}, rank={self.rank}",
                f"resolution={self.resolution:d}, bandwidth={self.bandwidth:0.2f}",
            ]
        )

class SynthesisInputSin(torch.nn.Module):
    def __init__(
        self,
        channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution for input layer.
        attn = False,
        rank=8,  # Rank of output biline.
        # bandwidth=10,  # Bandwidth of fourier feature.
        is_first = True,
        spectral_weight = False, # TODO
    ):
        super().__init__()
        self.channels = channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.rank = rank
        # self.bandwidth = bandwidth
        self.attn = attn
        self.spectral_weight = spectral_weight
        # 归一化的频率用于控制正弦函数在一个周期内的变化
        self.conv = nn.Conv2d(2, channels * 2 * rank, kernel_size=1, padding=0, bias=True)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / 2), np.sqrt(9 / 2))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / 2), np.sqrt(3 / 2))
        
        coords = torch.linspace(0, 1, resolution).reshape(1, 1, -1, 1)
        coords = coords.repeat(1, 2, 1, 1)

        self.register_buffer("coords", coords)

        if attn:
            # self.transformer = TransformerLayer(dim = channels,pos_dim=w_dim,
            self.transformer = TransformerLayer(dim = channels,
                                                from_len=resolution*rank*2,to_len=1,
                                                from_dim=channels,to_dim=w_dim)
            
        else:
            # if not self.spectral_weight:
            self.weight = torch.nn.Parameter(torch.randn([1, channels, 1, 2 * rank]))
            # else:
                # self.complex_weight = torch.nn.Parameter(torch.randn([1, channels, 1, rank + 1, 2], dtype=torch.float32) * 0.02)
            self.affine = FullyConnectedLayer(w_dim, channels * 2 * rank, bias_init=1)

    def forward(self, w, coords=None):
        if coords is None:
            x = self.conv(self.coords.repeat(w.shape[0], 1, 1, 1))
        else:
            # coords = nn.Parameter(coords, requires_grad=False)
            coords = coords.permute(0,2,1,3).to(w.device)
            x = self.conv(coords.repeat(w.shape[0], 1, 1, 1))
        # x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0], self.channels, x.shape[2], 2 * self.rank)
        x = torch.sin(x)

        if self.attn:
            shape = x.shape
            x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1) # b,c,r,rank -> b,r*rank,c
            # w = w.unsqueeze(1)
            x,_,_ = self.transformer(x,get_components(w),from_pos=None,to_pos=None)
            x = x.permute(0, 2, 1).reshape(shape[0],self.channels,shape[2],shape[3])
            
        else:
            # if self.spectral_weight:
            #     weight = self.complex_weight
            # else:
            weight = self.weight * self.affine(get_global(w)).reshape(w.shape[0], self.channels, 1, -1)
            x = x * weight / np.sqrt(self.channels)

        return x

    def extra_repr(self):
        return " ".join(
            [
                f"channels={self.channels:d}, w_dim={self.w_dim:d}, rank={self.rank}",
                f"resolution={self.resolution:d}, bandwidth={self.bandwidth:0.2f}",
            ]
        )
        

# ----------------------------------------------------------------------------


@persistence.persistent_class
class ToResLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resolution, k,
                 attn_res = True, rank=8, fts_dim = None,
                 fc_clamp=None, channels_last=False,
                 spectral_weight = False,
                 ltnt_gate = False,        # Gate attention from latents, such that components may not send information
                                        # when gate value is low
                 img_gate = False,        # Gate attention for images, such that some image positions may not get updated
                                        # or receive information when gate value is low
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.k = k
        self.attn_res = attn_res
        self.fts_dim = fts_dim
        self.spectral_weight = spectral_weight
        
        # 先修改了spectral_weight，2023年09月08日
        if fts_dim is not None:
            self.text_affine = FullyConnectedLayer(fts_dim, in_channels * 2, activation='lrelu', lr_multiplier=0.01)
            self.text_affine2 = FullyConnectedLayer(in_channels, in_channels * 2, activation='lrelu', lr_multiplier=0.01)
            self.text_ffc = FullyConnectedLayer(4*in_channels,in_channels * 2,bias_init=1)
        if attn_res:
            if out_channels != 3:
                kwargs = {
                    "from_len":  resolution * rank * 2,       "to_len":  k - 1,         # The from/to tensor lengths
                    "from_dim":  out_channels,                "to_dim":  w_dim,         # The from/to tensor dimensions
                    "from_gate": img_gate,                    "to_gate": ltnt_gate,     # Gate attention flow between from/to tensors
                }
            else:
                kwargs = {
                    "from_len":  resolution * resolution,       "to_len":  k - 1,         # The from/to tensor lengths
                    "from_dim":  out_channels,                "to_dim":  w_dim,         # The from/to tensor dimensions
                    "from_gate": img_gate,                    "to_gate": ltnt_gate,     # Gate attention flow between from/to tensors
                }
            # self.pe,_ = get_positional_encoding(resolution, in_channels,rank=rank)
            self.transformer = TransformerLayer(dim = out_channels, modulate = False, **kwargs)
            # self.transformer = TransformerLayer(dim = out_channels,pos_dim=in_channels,**kwargs)
            # self.ffc = EqualLinear(in_channels,out_channels)

        self.fc_clamp = fc_clamp
        
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        if self.spectral_weight:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]).to(memory_format=memory_format))
            self.weight_gain = 1 / np.sqrt(in_channels)
        else:
            # TODO
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight_gain = 1 / np.sqrt(in_channels)
            
            # self.affine_spectral = FullyConnectedLayer(w_dim, in_channels + 2, bias_init=1)
            # self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]).to(memory_format=memory_format))
            
            self.affine_spectral_new = FullyConnectedLayer(w_dim, in_channels + 2, bias_init=1)
            self.weight = torch.nn.Parameter(torch.randn([out_channels // 2 + 2, in_channels + 2]).to(memory_format=memory_format))
            # self.affine = FullyConnectedLayer(w_dim, in_channels * 2, bias_init=1)
            # self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels * 2]).to(memory_format=memory_format))
            # self.weight_gain = 1 / np.sqrt(in_channels * 2)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        

    def forward(self, x, w, fsts=None): 
        styles = self.affine(get_global(w)) * self.weight_gain
        if self.fts_dim is not None and fsts is not None:
            fsts = self.text_affine(fsts)
            fsts = self.text_affine2(fsts)
            styles = self.text_ffc(torch.cat([fsts,styles], dim=-1))
        if self.spectral_weight:
            x = modulated_fc(x=x, weight=self.weight, styles=styles, demodulate=False)
        else:
            if self.fts_dim is not None and fsts is not None:
                # styles2 = self.affine_spectral(styles)
                styles2 = self.affine_spectral_new(styles)
            else:
                # styles2 = self.affine_spectral(get_global(w))
                styles2 = self.affine_spectral_new(get_global(w))
                
            x = modulated_spectral_fc_new(x=x, weight=self.weight, styles2=styles2)
            x = x[:,:self.out_channels] # for situation that out_channels=3
            # x = modulated_spectral_fc(x=x, weight1=self.weight, styles=styles, styles2=styles2)
        
        if self.attn_res:
            shape = x.shape
            # pos = self.pe
            # pos = pos.to(x.device)
            x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1) # b,c,r,rank -> b,r*rank,c
            # w = w.unsqueeze(1)
            # x,_,_ = self.transformer(x,w,from_pos=pos,to_pos=None)
            x,_,_ = self.transformer(
                from_tensor = x, to_tensor = get_components(w.to(x.dtype)),
                from_pos=None,to_pos=None
            )
            # x = self.ffc(x)
            x = x.permute(0, 2, 1).reshape(shape[0],self.out_channels,shape[2],shape[3])
            
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.fc_clamp)
        return x

    def extra_repr(self):
        return f"in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}"


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 1 = first block.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this block.
        is_last,  # Is this the last block?
        k,
        attn = False,
        attn_res = True,
        selfattn = False,
        rank = 8,
        res_channels=None,  # Number of output residual channels.
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        fc_clamp=256,  # Clamp the output of fully connected layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        poly = False,
        spectral_weight = False,
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ["orig", "skip"]
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.k = k
        self.res_channels = res_channels
        self.is_last = is_last
        self.architecture = architecture if in_channels > 1 else "orig"
        self.use_fp16 = use_fp16
        self.selfattn = selfattn
        self.channels_last = use_fp16 and fp16_channels_last
        self.num_fc = 0
        self.num_tores = 0
        self.poly = poly

        if in_channels == 1:
            if poly is False:
                self.input = SynthesisInput(out_channels, w_dim=w_dim, resolution=resolution,
                                            attn=False, rank=rank)
                # self.input = SynthesisInputSin(out_channels, w_dim=w_dim, resolution=resolution,
                #                             attn=False, rank=rank)
            else:
                self.input = PolyInput(out_channels, w_dim=w_dim, resolution=resolution, rank=rank)
            self.fc1 = SynthesisLayer(
                out_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                k = k,
                attn = attn,
                rank=rank,
                poly=poly,
                fc_clamp=fc_clamp,
                channels_last=self.channels_last,
                res_channels = res_channels,
                spectral_weight = spectral_weight,
                **layer_kwargs,
            )

        if in_channels != 1:
            self.fc0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                k = k,
                attn = attn,
                rank=rank,
                poly=poly,
                fc_clamp=fc_clamp,
                channels_last=self.channels_last,
                res_channels = res_channels,
                spectral_weight = spectral_weight,
                **layer_kwargs,
            )
            self.num_fc += 1
            self.fc1 = SynthesisLayer(
                out_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                k = k,
                attn = attn,
                rank=rank,
                poly=poly,
                fc_clamp=fc_clamp,
                channels_last=self.channels_last,
                res_channels = res_channels,
                spectral_weight = spectral_weight,
                **layer_kwargs,
            )
        self.num_fc += 1
        if selfattn:
            self.self_attn = TransformerBlock(out_channels,w_dim=w_dim)
            self.num_fc += 1

        if is_last or architecture == "skip":
            self.tores = ToResLayer(
                out_channels,
                self.res_channels,
                w_dim=w_dim,
                k = k,
                attn_res = attn_res,
                rank=rank,
                resolution = resolution,
                fc_clamp=fc_clamp,
                channels_last=self.channels_last,
                spectral_weight = False,
            )
            self.num_tores += 1

    def forward(self, x, res, ws, fsts=None, coords=None, force_fp32=False, update_emas=False, **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(
            ws,
            [None, self.k, (self.in_channels == 1) + self.num_fc + self.num_tores, self.w_dim],
        )
        w_iter = iter(ws.unbind(dim=2))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if self.in_channels == 1:
            if self.poly:
                x, grid_0 = self.input(next(w_iter), coords)
            else:
                x = self.input(next(w_iter), coords).to(dtype=dtype, memory_format=memory_format)
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)
        # Main layers.
        if self.in_channels == 1:
            if self.poly:
                x, grid_1 = self.fc1(x, next(w_iter), grid_0, fsts, **layer_kwargs)
            else:
                x = self.fc1(x, next(w_iter),fsts, **layer_kwargs)
        else:
            if self.poly:
                x, grid_0 = self.fc0(x, next(w_iter), coords, fsts, **layer_kwargs)
                x, grid_1 = self.fc1(x, next(w_iter), grid_0, fsts, **layer_kwargs)
            else:
                x = self.fc0(x, next(w_iter),fsts, **layer_kwargs)
                x = self.fc1(x, next(w_iter),fsts, **layer_kwargs)
        # Attn.
        if self.selfattn:
            x = self.self_attn([x,next(w_iter)])
        
        # ToRes.
        if self.is_last or self.architecture == "skip":
            y = self.tores(x, next(w_iter),fsts)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            res = res.add(y) if res is not None else y

        assert x.dtype == dtype
        assert res is None or res.dtype == torch.float32
        if not self.poly:
            return x, res
        else:
            return x, res, grid_1

    def extra_repr(self):
        return f"resolution={self.resolution:d}, architecture={self.architecture:s}"


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        k,
        attn = False,
        attn_res = True,
        final_img_resolution = 256, # Decide the block numbers of synthesis network
        biline_channels=32,  # Number of biline channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        poly = True,
        spectral_weight = False,
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        assert channel_base > 4 and channel_max > 1 and biline_channels > img_channels
        super().__init__()
        self.w_dim = w_dim
        self.k = k
        self.img_resolution = img_resolution
        self.final_img_resolution = final_img_resolution
        self.img_resolution_log2 = int(np.log2(final_img_resolution))
        self.img_channels = img_channels
        self.biline_channels = biline_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2**i for i in range(2, self.img_resolution_log2 + 1)]
        self.poly = poly
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 1
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 1
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.final_img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                attn = attn,
                k = k,
                attn_res = attn_res,
                resolution=img_resolution,
                res_channels=biline_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                poly = poly,
                spectral_weight = spectral_weight,
                **block_kwargs,
            )
            self.num_ws += block.num_fc
            if is_last:
                self.num_ws += block.num_tores
            setattr(self, f"b{res}", block)

        for res in self.block_resolutions[1:]:
            decoder = torch.nn.Sequential(
                FullyConnectedLayer(biline_channels, biline_channels * 2, activation="lrelu"),
                FullyConnectedLayer(biline_channels * 2, biline_channels * 4, activation="lrelu"),
                FullyConnectedLayer(biline_channels * 4, biline_channels * 2, activation="lrelu"),
                FullyConnectedLayer(biline_channels * 2, biline_channels, activation="lrelu"),
            )
            setattr(self, f"decoder_{res}", decoder)
        self.decoder_last = torch.nn.Sequential(
            FullyConnectedLayer(biline_channels, biline_channels * 2, activation="lrelu"),
            FullyConnectedLayer(biline_channels * 2, biline_channels * 4, activation="lrelu"),
            FullyConnectedLayer(biline_channels * 4, biline_channels * 2, activation="lrelu"),
            FullyConnectedLayer(biline_channels * 2, biline_channels, activation="lrelu"),
        )
        self.decoder_img = FullyConnectedLayer(biline_channels, img_channels)

        self.block0 = SynthesisBlock(
            biline_channels,
            biline_channels * 4,
            w_dim=w_dim,
            attn = False,
            attn_res = False,
            # attn_res = attn_res,
            k = k,
            resolution=img_resolution,
            res_channels=img_channels,
            is_last=False,
            use_fp16=True,
            # use_fp16=False,
            fc_clamp=256,
            poly = False,
            # spectral_weight = spectral_weight,
            spectral_weight = False,
        )
        self.block1 = SynthesisBlock(
            biline_channels * 4,
            biline_channels * 2,
            w_dim=w_dim,
            attn = False,
            attn_res = False,
            k = k,
            resolution=img_resolution,
            res_channels=img_channels,
            is_last=True,
            use_fp16=True,
            # use_fp16=False,
            fc_clamp=256,
            poly = False,
            # spectral_weight = spectral_weight,
            spectral_weight = False,
        )

    def forward(self, ws, coords, fsts=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(ws, [None, self.k, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                block_ws.append(ws.narrow(2, w_idx, (res == 4) + block.num_fc + block.num_tores))
                w_idx += block.num_fc

        x = biline = None
        feature = None
        grid = coords
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f"b{res}")
            if not self.poly:
                x, biline = block(x, biline, cur_ws, fsts=fsts, coords=coords, **block_kwargs)
            else:
                x, biline, grid = block(x, biline, cur_ws, fsts=fsts, coords=grid, **block_kwargs)
            if biline is not None:
                x_line, y_line = (biline * np.sqrt(0.5)).chunk(2, dim=-1)
                new_f = x_line @ y_line.transpose(-2, -1)
                batch, channel, height, width = new_f.shape
                new_f = new_f.permute(0, 2, 3, 1).reshape(-1, channel)
                if feature is None:
                    feature = new_f
                else:
                    decoder = getattr(self, f"decoder_{res}")
                    feature = decoder(feature) + new_f
        del new_f
        feature = self.decoder_last(feature)
        img = self.decoder_img(feature)
        feature = feature.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        img = img.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        ws = ws[:,:, -1:, :].repeat(1, 1, 3, 1)
        feature, img = self.block0(feature, img, ws, **block_kwargs)
        feature, img = self.block1(feature, img, ws, **block_kwargs)
        del feature
        return img

    def extra_repr(self):
        return " ".join(
            [
                f"w_dim={self.w_dim:d}, num_ws={self.num_ws:d},",
                f"img_resolution={self.img_resolution:d}, final_resolution={self.final_img_resolution:d}, img_channels={self.img_channels:d},",
                f"num_fp16_res={self.num_fp16_res:d}",
            ]
        )


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        fts_dim,
        k, # Number of latent vector components z_1,...,z_k
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        min_resolution=0.5,
        max_resolution=1.0,
        mapping_kwargs={},  # Arguments for MappingNetwork.
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.fts_dim = fts_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.k = k
        self.input_shape = [None, k, z_dim]
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.head_pos_enc = ScalePartyPE(min_resolution, max_resolution,pad=0)
        self.coords = self.head_pos_enc.make_grid2d(
            n_height=self.img_resolution, n_width=self.img_resolution,
        )
        torch_misc.assert_shape(self.coords, [None, self.img_resolution, 2, 1])
        
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            # fts_dim = fts_dim,
            k = k,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, fts_dim=fts_dim, k=k, num_broadcast=self.num_ws, **mapping_kwargs)
        # self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, fts_dim=fts_dim, k=k, num_ws=self.num_ws, **mapping_kwargs)
        # self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim,fts_dim=0, num_ws=self.num_ws, **mapping_kwargs)

    def forward(
        self,
        z,
        c,
        fsts = None,
        fts = None,
        coords = None,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        **synthesis_kwargs,
    ):
        if fsts is None:
            fsts = torch.randn(z.size()[0], self.fts_dim).to(z.device)
            fsts = fsts/fsts.norm(dim=1,keepdim=True)
        if fts is None and self.k != 1:
            fts = torch.randn(z.size()[0],77, self.fts_dim).to(z.device)
            fts = fts/fts.norm(dim=1,keepdim=True)
        ws = self.mapping(
            z,
            c,
            fsts,
            fts=fts,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        if coords is None:
            coords = self.coords
        else:
            coords = coords
        torch_misc.assert_shape(ws, [None, self.k, self.num_ws, self.w_dim])
        img = self.synthesis(ws, coords=coords, update_emas=update_emas, **synthesis_kwargs)
        # img = self.synthesis(ws, fsts=fsts, update_emas=update_emas, **synthesis_kwargs)
        return img

@persistence.persistent_class
class SuperresGenerator(torch.nn.Module):
    def __init__(
        self,
        path_stem,
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        final_img_resolution = 256, # Decide the block numbers of synthesis network
        z_dim = 512,  # Input latent (Z) dimensionality.
        c_dim = 0,  # Conditioning label (C) dimensionality.
        w_dim = 512,  # Intermediate latent (W) dimensionality.
        fts_dim = 512,
        k = 16, # Number of latent vector components z_1,...,z_k
        min_resolution=0.5,
        max_resolution=1.0,
        mapping_kwargs={},  # Arguments for MappingNetwork.
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        # self.z_dim = z_dim
        # self.c_dim = c_dim
        # self.w_dim = w_dim
        # self.fts_dim = fts_dim
        self.img_resolution = img_resolution
        # self.img_channels = img_channels
        
        # self.input_shape = [None, k, z_dim]
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.head_pos_enc = ScalePartyPE(min_resolution, max_resolution,pad=0)
        self.coords = self.head_pos_enc.make_grid2d(
            n_height=self.img_resolution, n_width=self.img_resolution,
        )
        # torch_misc.assert_shape(self.coords, [None, self.img_resolution, 2, 1])
        with dnnlib.util.open_url(path_stem) as f:
            G_stem = legacy.load_network_pkl(f)['G_ema']
        self.mapping = G_stem.mapping
        self.synthesis = G_stem.synthesis
        self.input_shape = G_stem.input_shape
        self.k = G_stem.k
        self.z_dim = G_stem.z_dim
        self.c_dim = G_stem.c_dim
        self.w_dim = G_stem.w_dim
        self.fts_dim = G_stem.fts_dim
        self.img_resolution = img_resolution
        self.img_channels = G_stem.img_channels
        
        # self.synthesis = SynthesisNetwork(
        #     w_dim=G_stem.w_dim,
        #     # fts_dim = fts_dim,
        #     k = G_stem.k,
        #     img_resolution=img_resolution,
        #     img_channels=img_channels,
        #     **synthesis_kwargs,
        # )
        # self.num_ws = self.synthesis.num_ws
        # self.mapping = MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, fts_dim=self.fts_dim, k=k, num_broadcast=self.num_ws, **mapping_kwargs)
        # change_layer = ["b4.fc0.transformer",
        #                 "b4.fc1.transformer",
        #                 "b8.fc0.transformer",
        #                 "b8.fc1.transformer",
        #                 "b16.fc0.transformer",
        #                 "b16.fc1.transformer",
        #                 "b32.fc0.transformer",
        #                 "b32.fc1.transformer",
        #                 "b64.fc0.transformer",
        #                 "b64.fc1.transformer",
        #                 "b128.fc0.transformer",
        #                 "b128.fc1.transformer",
        #                 "b256.fc0.transformer",
        #                 "b256.fc1.transformer",
        #                 ]
        # for name, module in [("G", G)
        # for name, layer in G_stem.synthesis.name_children():
        # misc.copy_params_and_buffers_without(G_stem.synthesis, self.synthesis, change_layer, require_all=False)
        # G_stem.input_shape
        
        self.synthesis.img_resolution = img_resolution

    def forward(self, z, c, 
                fsts = None,
                fts = None,
                coords = None,
                truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        if fsts is None:
            fsts = torch.randn(z.size()[0], self.fts_dim).to(z.device)
            fsts = fsts/fsts.norm(dim=1,keepdim=True)
        if fts is None and self.k != 1:
            fts = torch.randn(z.size()[0],77, self.fts_dim).to(z.device)
            fts = fts/fts.norm(dim=1,keepdim=True)
        ws = self.mapping(
            z,
            c,
            fsts,
            fts=fts,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )
        if coords is None:
            coords = self.coords
        else:
            coords = coords
        # torch_misc.assert_shape(ws, [None, self.k, self.num_ws, self.w_dim])
        img = self.synthesis(ws, coords=coords, update_emas=update_emas, **synthesis_kwargs)
        return img
    
# ----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        wavelet             = True,
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.wavelet = wavelet
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if wavelet == True:
            if self.in_channels != 0:
                self.iwt = InverseHaarTransform(3)
                self.dwt = HaarTransform(3)
            else:
                # self.dwt = HaarTransform(3)
                self.fromrgb_ori = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.fromrgb = Conv2dLayer(img_channels * 4, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)
                
        elif in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        # to do
        # self.affine = FullyConnectedLayer(512, (tmp_channels + tmp_channels) * 10, bias_init=0)
        # self.weight = torch.nn.Parameter(torch.randn([tmp_channels, tmp_channels, 1, 1]))
        
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)
        
        
        
    def forward(self, x, img, force_fp32=False, fsts=None):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.wavelet == True:
            img = img.to(dtype=dtype, memory_format=memory_format)
            if self.in_channels == 0:
                # img = self.dwt(img)
                y = self.fromrgb_ori(img)
            else:
                img = self.dwt(img)
                y = self.fromrgb(img)
                img = self.iwt(img)
                img = upfirdn2d.downsample2d(img, self.resample_filter)

            # if self.in_channels != 0:
            #     img = self.dwt(img)
            x = x + y if x is not None else y
        elif self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None


        # if fsts is not None:
        #     styles = self.affine(fsts)
        #     x = fmm_modulate_linear(x=x, weight=self.weight, styles=styles, noise=None, activation="demod")
        
        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        use_fts,
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        fsts_dim            = 512,
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.use_fts = use_fts

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        if use_fts:
            self.fsts = FullyConnectedLayer(in_channels, fsts_dim)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, use_norm=True, fsts=None, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        fc = self.fc(x.flatten(1))
        x = self.out(fc)

        if self.use_fts:
            d_fsts = self.fsts(fc)
            if use_norm:
                d_fsts = d_fsts/d_fsts.norm(dim=-1, keepdim=True)
        
            x = x + (d_fsts*fsts).sum()
        else:
            d_fsts = fsts
        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x, d_fsts

#----------------------------------------------------------------------------

def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
    
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d_op(input, self.ll, down=2)
        lh = upfirdn2d_op(input, self.lh, down=2)
        hl = upfirdn2d_op(input, self.hl, down=2)
        hh = upfirdn2d_op(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)
    

class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d_op(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d_op(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d_op(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d_op(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh
    
@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        use_norm            = False, 
        use_fts             = True,
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.use_norm = use_norm # new,default False
        self.use_fts  = use_fts # new,default true
        # {256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4],use_fts=self.use_fts, cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, fsts=None, update_emas=False,  **block_kwargs):
        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, fsts=fsts, **block_kwargs)
        assert torch.isnan(x).int().sum()==0,"nan in x1_d."
        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x, d_fsts = self.b4(x, img, cmap, fsts=fsts, use_norm=self.use_norm)
        assert torch.isnan(x).int().sum()==0,"nan in x2_d."
        assert torch.isnan(d_fsts).int().sum()==0,"nan in d_fsts."
        return x, d_fsts

#----------------------------------------------------------------------------
