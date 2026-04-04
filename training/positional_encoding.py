import numpy as np
import torch
import math
from training import misc
from random import randint, uniform

# frequency mechanism
def simple_norm(x,norm=None):
    if norm is None:
        pass
    elif norm == "instance":
        # instance_norm = nn.InstanceNorm2d(in_channels, eps=1e-8, affine=False)
        # x_bias = instance_norm(x_bias)
        # Instance normalization
        norm_axis = (2,3)  # Normalize along the spatial dimensions
        mean = x.mean(dim=norm_axis, keepdim=True)
        var = x.var(dim=norm_axis, keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + 1e-8))
    elif norm == "layer":
        # layer_norm = nn.LayerNorm(x_bias.shape[1:], eps=1e-8, elementwise_affine=False)
        # x_bias = layer_norm(x_bias)
        # Layer normalization
        norm_axis = (1,2,3)  # Normalize across the channel axis
        mean = x.mean(dim=norm_axis, keepdim=True)
        var = x.var(dim=norm_axis, keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + 1e-8))
    return x

def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y

def decompose(x, dim=(2,3), mode='all'):
    # [m, n, 2]
    fft_im = torch.view_as_real(torch.fft.rfft2(x, dim=dim, norm='ortho'))

    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1)
        # fft_amp = torch.sqrt(replace_denormals(fft_amp))
        fft_amp = torch.sqrt(fft_amp)
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        # fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
        fft_pha = torch.atan2(fft_im[..., 1], fft_im[..., 0])
    else:
        fft_pha = None
    return fft_pha, fft_amp

def decompose_ori(x, dim=(2,3), mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, dim=dim, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp

def compose(phase, amp, dim=(2,3)):
    # x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1) 
    # # x = x / math.sqrt(x.shape[dim[0]] * x.shape[dim[1]])
    # x = torch.view_as_complex(x)
    real_part = torch.cos(phase) * amp
    imag_part = torch.sin(phase) * amp
    x = torch.complex(real_part, imag_part)
    return torch.fft.irfft2(x, dim=dim, norm='ortho')

# def compose_ori(phase, amp, dim=(2,3)):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1) 
    x = x / math.sqrt(x.shape[dim[0]] * x.shape[dim[1]])
    x = torch.view_as_complex(x)
    return torch.fft.irfft2(x, dim=dim, norm='ortho')

def decompose_single(x, dim=1, mode='all'):
    fft_im = torch.view_as_real(torch.fft.rfft(x, dim=dim, norm='ortho'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_amp = torch.sqrt(fft_im.pow(2).sum(dim=-1))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp

def compose_single(phase, amp, dim=1):
    real_part = torch.cos(phase) * amp
    imag_part = torch.sin(phase) * amp
    x = torch.complex(real_part, imag_part)
    return torch.fft.irfft(x, dim=dim, norm='ortho')

def compose_simple_2d(x_phase, x_amplitude, dim=(2,3)):
    x = torch.fft.irfft2(x_amplitude * torch.exp(1j * x_phase), dim=dim, norm='ortho')
    return x

def decompose_simple_2d(x, dim=(2,3)):
    x = torch.fft.rfft2(x.to(torch.float32), dim=dim, norm='ortho')
    x_amplitude = replace_denormals(torch.abs(x))
    x_phase = replace_denormals(torch.angle(x))
    return x_phase, x_amplitude

def compose_simple_1d(x_phase, x_amplitude, dim=1):
    x = torch.fft.irfft(x_amplitude * torch.exp(1j * x_phase), dim=dim, norm='ortho')
    return x

def decompose_simple_1d(x, dim=1):
    x = torch.fft.rfft(x.to(torch.float32), dim=dim, norm='ortho')
    x_amplitude = replace_denormals(torch.abs(x))
    x_phase = replace_denormals(torch.angle(x))
    return x_phase, x_amplitude

# Positional encoding
# ----------------------------------------------------------------------------

# creps PE
def get_encoding(size, dim, rank):
    coords = torch.linspace(0, 1, size).reshape(-1,1,1)
    coords = coords.repeat(1,2,1)
    bandwidth = 10
    freqs =torch.randn([1,dim // 2 * rank]) * bandwidth
    x = coords @ freqs # s * 2 * (dim // 2 * rank)
    x = x.reshape(x.shape[0], 2, dim // 2, rank)

    x = x.permute(2, 0, 1, 3) * (np.pi * 2)
    x = x.reshape(*x.shape[:2], -1)

    x = torch.cat([x, x + np.pi / 2], dim=0)
    x = torch.sin(x)
    x = x.permute(1,2,0) # s * 2rank * dim
    return x


# 2d linear embeddings [size, size, dim] in a [-rng, rng] range where size = grid size and
# dim = the embedding dimension. Each embedding consists of 'num' parts, with each part measuring
# positional similarity along another direction, uniformly spanning the 2d space.
def get_linear_encoding(size, dim, num, rng = 1.0):
    theta = torch.arange(0, math.pi, step = math.pi / num)
    dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)
    embs = torch.nn.Parameter(torch.rand([num, int(dim / num)]))

    c = torch.linspace(-rng, rng, size)
    x = c.unsqueeze(0).tile([size, 1])
    y = c.unsqueeze(1).tile([1, size])
    xy = torch.stack([x, y], dim = -1)

    lens = (xy.unsqueeze(2) * dirs).sum(dim = -1, keepdim = True)
    emb = (lens * embs).reshape(size, size, dim)
    return emb

# 2d sinusoidal embeddings [size, size, dim] with size = grid size and dim = embedding dimension
# (see "Attention is all you need" paper)
def get_sinusoidal_encoding(size, dim, num = 2):
    # Standard positional encoding in the two spatial w,h directions
    if num == 2:
        c = torch.linspace(-1.0, 1.0, size).unsqueeze(-1)
        i = torch.arange(int(dim / 4)).to(torch.float32)

        peSin = torch.sin(c / (torch.pow(10000.0, 4 * i / dim)))
        peCos = torch.cos(c / (torch.pow(10000.0, 4 * i / dim)))

        peSinX = peSin.unsqueeze(0).tile([size, 1, 1])
        peCosX = peCos.unsqueeze(0).tile([size, 1, 1])
        peSinY = peSin.unsqueeze(1).tile([1, size, 1])
        peCosY = peCos.unsqueeze(1).tile([1, size, 1])

        emb = torch.cat([peSinX, peCosX, peSinY, peCosY], dim = -1)
    # Extension to 'num' spatial directions. Each embedding consists of 'num' parts, with each
    # part measuring positional similarity along another direction, uniformly spanning the 2d space.
    # Each such part has a sinus and cosine components.
    else:
        theta = torch.arange(0, math.pi, math.pi / num)
        dirs = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)

        c = torch.linspace(-1.0, 1.0, size)
        x = c.unsqueeze(0).tile([size, 1])
        y = c.unsqueeze(1).tile([1, size])
        xy = torch.stack([x, y], dim = -1)

        lens = (xy.unsqueeze(2) * dirs).sum(dim = -1, keepdim = True)

        i = torch.arange(int(dim / (2 * num))).to(torch.float32)
        sins = torch.sin(lens / (torch.pow(10000.0, 2 * num * i / dim)))
        coss = torch.cos(lens / (torch.pow(10000.0, 2 * num * i / dim)))
        emb = torch.cat([sins, coss], dim = -1).reshape(size, size, dim)

    return emb

# 2d positional encoding of dimension 'dim' in a range of resolutions from 2x2 up to 'max_res x max_res'
#
# pos_type: supports several types of embedding schemes:
# - sinus: (see "Attention is all you need")
# - linear: where each position gets a value of [-1, 1] * trainable_vector, in each spatial
#   direction based on its location.
# - trainable: where an embedding of position [w,h] is [emb_w, emb_h] (independent parts in
#   each spatial direction)
# - trainable2d: where an embedding of position [w,h] is emb_{w,h} (a different embedding for
#   each position)
#
# dir_num: Each embedding consists of 'dir_num' parts, with each path measuring positional similarity
# along another direction, uniformly spanning the 2d space.
#
# shared: True for using same embeddings for all directions / parts
# init: uniform or normal distribution for trainable embeddings initialization
def get_positional_encoding(
        res, 
        pos_dim,                # Positional encoding dimension
        rank = 0,
        pos_type = "sinus",     # Positional encoding type: linear, sinus, trainable, trainable2d
        pos_init = "uniform",   # Positional encoding initialization distribution: normal or uniform
        pos_directions_num = 2, # Positional encoding number of spatial directions
        shared = False,         # Share embeddings for x and y axes
        crop_ratio = None,      # Crop the embedding features to the ratio 
        **_kwargs):             # Ignore unrecognized keyword args

    params = []
    initializer = torch.rand if pos_init == "uniform" else torch.randn
    if rank != 0:
        emb = get_encoding(res,pos_dim,rank)
    elif pos_type == "sinus":
        emb = get_sinusoidal_encoding(res, pos_dim, num = pos_directions_num)
    elif pos_type == "linear":
        emb = get_linear_encoding(res, pos_dim, num = pos_directions_num)
    elif pos_type == "trainable2d":
        emb = torch.nn.Parameter(initializer([res, res, pos_dim]))
        params = [emb]
    else: # pos_type == "trainable"
        xemb = torch.nn.Parameter(initializer([res, int(pos_dim / 2)]))
        yemb = xemb if shared else torch.nn.Parameter(initializer(res, int(pos_dim / 2)))
        params = [xemb, yemb]
        xemb = xemb.unsqueeze(0).tile([res, 1, 1])
        yemb = yemb.unsqueeze(1).tile([1, res, 1])
        emb = torch.cat([xemb, yemb], dim = -1)

    emb = misc.crop_tensor(emb, crop_ratio)
    return emb, params

# Produce trainable embeddings of shape [size, dim], uniformly/normally initialized
def get_embeddings(size, dim, init = "uniform", name = None):
    if size == 0:
        return None
    initializer = torch.rand if init == "uniform" else torch.randn
    emb = torch.nn.Parameter(initializer([size, dim]))
    return emb

# Scale party
class ScalePartyPE(torch.nn.Module):
    def __init__(self,
                 min_resolution=0.5,
                 max_resolution=1.0,
                 pad=6,
                 clamp = True,
                 ):
        super().__init__()
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.pad = pad 
        self.clamp = clamp 
        assert pad % 2 == 0


    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    # def make_grid2d(self, n_height, n_width, num_batches=1, start_h=None, start_w=None, width=None, height=None, requires_grad=False):
    def make_grid2d(self, n_height, n_width, # n_height和n_width是原图的尺寸,只关乎分辨率，越大采样点越多分辨率越大
                    num_batches=1, start_h=0, start_w=0, # start_h和start_w是相对于原图的起始点位置，都是0.5则位于图像的中心
                    width=1.0, height=1.0, # 与原图像分辨率有关，代表着呈现的内容是是原图像分辨率的倍数，即scale是原图像/width or 原图像/height
                    requires_grad=False):

        max_resolution =  self.max_resolution
        min_resolution = self.min_resolution
        half_pad = self.pad // 2

        if not width and not height:
            width = height = uniform(min_resolution, max_resolution)

        if n_height == n_width:
            pe = torch.zeros(2, n_height + self.pad, 1)
        else:
            pe = torch.zeros(2, n_height + self.pad, n_width + self.pad)

        if start_h is None:
            start_h = uniform(-1, 1 - 2*height)
        else:
            start_h = start_h*2 - 1
        if start_w is None:
            start_w = uniform(-1, 1 - 2*width) 
        else:
            start_w = start_w*2 - 1

        step_w = 2 * width /  n_width
        step_h = 2 * height / n_height

        start_w += step_w/2
        start_h += step_h/2
        # start_w = (2 * start_w) / max_resolution - 1 + (step_w / 2)  
        # start_h = (2 * start_h) / max_resolution - 1 + (step_h / 2)  

        pos_w = torch.arange(start_w - (half_pad * step_w), start_w + (n_width + half_pad) * step_w, step_w)[:(n_width+self.pad)].unsqueeze(1)
        pos_h = torch.arange(start_h - (half_pad * step_h), start_h + (n_height + half_pad) * step_h, step_h)[:(n_height+self.pad)].unsqueeze(1)

        if self.clamp: # 限制值的范围
            pos_w = pos_w.clamp_(-1, 1)
            pos_h = pos_h.clamp_(-1, 1)

        if n_height == n_width:
            pe[0] = pos_w
            pe[1] = pos_h
        else:
            pe[0] = pos_w.transpose(0,1).repeat(1, n_width+self.pad, 1)
            pe[1] = pos_h.repeat(1, 1, n_height+self.pad)
        
        pe = pe.repeat(num_batches, 1, 1, 1)
        pe = pe.permute(0,2,1,3)
        return pe

 
        # return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)