# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, upfirdn2d

import clip
import torch.nn.functional as F
import torchvision.transforms as T

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg
    ):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(
        self,
        device,
        G,
        D,
        augment_pipe=None,
        r1_gamma=10,
        style_mixing_prob=0,
        pl_weight=0,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_no_weight_grad=False,
        pl_start_kimg=0,
        blur_init_sigma=0,
        blur_fade_kimg=0,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_start_kimg = pl_start_kimg
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        clip_model, _ = clip_model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model here
        self.clip_model = clip_model.eval().float()

    def run_G(self, z, c, fsts, fts=None, update_emas=False):
        ws = self.G.mapping(z, c, fsts, fts=fts, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function("style_mixing"):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        coords = self.G.head_pos_enc.make_grid2d(n_height=self.G.img_resolution, n_width=self.G.img_resolution)
        img = self.G.synthesis(ws, coords=coords, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, fsts, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function("blur"):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits, d_fsts = self.D(img, c, fsts=fsts, update_emas=update_emas)
        return logits, d_fsts

    def normalize(self):
        return T.Compose([
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])    
    
    def full_preprocess(self, img, mode='bicubic', ratio=0.5):
        full_size = img.shape[-2]

        if full_size < 224:
            pad_1 = torch.randint(0, 224-full_size, ())
            pad_2 = torch.randint(0, 224-full_size, ())
            m = torch.nn.ConstantPad2d((pad_1, 224-full_size-pad_1, pad_2, 224-full_size-pad_2), 1.)
            reshaped_img = m(img)
        else:
            cut_size = torch.randint(int(ratio*full_size), full_size, ())
            left = torch.randint(0, full_size-cut_size, ())
            top = torch.randint(0, full_size-cut_size, ())
            cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
            reshaped_img = F.interpolate(cropped_img, (224, 224), mode=mode, align_corners=False)
        reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
        reshaped_img = self.normalize()(reshaped_img)
        return  reshaped_img
    
    def contra_loss(self, temp, mat1, mat2, lam):
        sim = torch.cosine_similarity(mat1.unsqueeze(1), mat2.unsqueeze(0), dim=-1) # dim=n时，只有n维度的值变，其他不变，作为一个tensor求相似度
        assert torch.isnan(sim).int().sum()==0,"nan in sim."
        if temp > 0.:
            # sim = torch.exp(sim/temp) # This implementation is incorrect, it should be sim=sim/temp.
            # However, this incorrect implementation can reproduce our results with provided hyper-parameters.
            # If you want to use the correct implementation, please manually revise it.
            # The correct implementation should lead to better results, but don't use our provided hyper-parameters, you need to carefully tune lam, temp, itd, itc and other hyper-parameters
            # 如果是图像和文本特征，那么lam=0是表示一个batch文本生成的所有图像和特定文本的相似度经过softmax后取特定文本生成的图像和特定文本最终得到的相似度作为输出
            # 反之，lam=1表示一个batch的所有文本特征和特定文本生成的图像的相似度经过softmax后取特定文本和特定文本生成的图像最终得到的相似度作为输出
            sim = sim/temp # conrrect implementation
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp # dim=1为sim中mat2的每个特征与mat1中特定特征的softmax的分别相似度
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp # dim=0为sim中mat1的每个特征与mat2中特定特征的softmax的分别相似度
            # global pri # 测试用，输出shape一次
            # if pri:# 测试
            #     print("测试sim shape:",sim.shape)
            #     print("sim1 shape:",sim1.shape)
            #     print("sim2 shape:",sim2.shape)
            #     print("sim1 ?= sim2:",torch.eq(sim1,sim2))
            #     pri = False
                
            if 0.<lam < 1.:
                return lam*torch.log(sim1) + (1.-lam)*torch.log(sim2)
            elif lam == 0:
                return torch.log(sim2)
            else:
                return torch.log(sim1)
        else:
            return torch.diagonal(sim)
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg,
                             sent_fts, txt_fts, temp, lam, gather, d_use_fts, itd, itc):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        if self.pl_weight == 0 or cur_nimg < self.pl_start_kimg * 1e3:
            phase = {"Greg": "none", "Gboth": "Gmain"}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {"Dreg": "none", "Dboth": "Dmain"}.get(phase, phase)
        blur_sigma = (
            max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        )

        img_txt_d = itd # discriminator
        img_txt_c = itc # clip
        
        # Gmain: Maximize logits for generated images.
        if phase in ["Gmain", "Gboth"]:
            with torch.autograd.profiler.record_function("Gmain_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sent_fts, fts=txt_fts)
                gen_logits, gen_d_fsts = self.run_D(gen_img, gen_c, fsts=sent_fts, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                
                
                # copied from galip,值得借鉴，如果出现nan
                # fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                # fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                if img_txt_c > 0.: # for fully-supervised
                    normed_gen_full_img = self.full_preprocess(gen_img)
                    img_fts_gen_full = self.clip_model.encode_image(normed_gen_full_img)
                    img_fts_gen_full = img_fts_gen_full/img_fts_gen_full.norm(dim=-1, keepdim=True)
                    # assert torch.isnan(gen_img).int().sum()==0,"nan in gen_img."
                    # assert torch.isnan(normed_gen_full_img).int().sum()==0,"nan in normed_gen_full_img."
                    # assert torch.isnan(sent_fts).int().sum()==0,"nan in sent_fts."
                    clip_loss_img_txt = self.contra_loss(temp, img_fts_gen_full, sent_fts, lam)
                    # assert torch.isnan(clip_loss_img_txt).int().sum()==0,"nan in clip_loss_img_txt."
                    loss_Gmain = loss_Gmain - img_txt_c*clip_loss_img_txt.mean()
                    training_stats.report('Loss/G_itc_only/loss', loss_Gmain)
                
                if img_txt_d > 0.:
                    loss_Gmain = loss_Gmain - img_txt_d*self.contra_loss(temp, gen_d_fsts, sent_fts, lam).mean()

                training_stats.report("Loss/G/loss", loss_Gmain)
            with torch.autograd.profiler.record_function("Gmain_backward"):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        # 让微小图像变化在w上的变化接近其均值，随着优化动态调整，自动找到一个全局最优值
        if phase in ["Greg", "Gboth"]:
            with torch.autograd.profiler.record_function("Gpl_forward"):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                sent_fts_0 = sent_fts[:batch_size]
                sent_fts_0.requires_grad_()
                txt_fts_0 = txt_fts[:batch_size]
                txt_fts_0.requires_grad_()
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size],sent_fts_0, fts=txt_fts_0)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function("pl_grads"), conv2d_gradfix.no_weight_gradients(
                    self.pl_no_weight_grad
                ):
                    if d_use_fts:
                        pl_grads = torch.autograd.grad(
                            outputs=[(gen_img * pl_noise).sum()], 
                            inputs=[gen_ws, sent_fts_0],
                            create_graph=True,
                            only_inputs=True)[0]
                    else:
                        pl_grads = torch.autograd.grad(
                            outputs=[(gen_img * pl_noise).sum()],
                            inputs=[gen_ws], 
                            create_graph=True, 
                            only_inputs=True)[0]
                # pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_lengths = pl_grads.square().sum(3).mean([1,2]).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report("Loss/pl_penalty", pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report("Loss/G/reg", loss_Gpl)
            with torch.autograd.profiler.record_function("Gpl_backward"):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ["Dmain", "Dboth"]:
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sent_fts, fts=txt_fts, update_emas=True)
                gen_logits, gen_d_fsts = self.run_D(gen_img, gen_c, fsts=sent_fts, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function("Dgen_backward"):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ["Dmain", "Dreg", "Dboth"]:
            name = "Dreal" if phase == "Dmain" else "Dr1" if phase == "Dreg" else "Dreal_Dr1"
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp = real_img.detach().requires_grad_(phase in ["Dreg", "Dboth"])
                real_logits, real_d_fts = self.run_D(real_img_tmp, real_c, fsts=sent_fts, blur_sigma=blur_sigma)
                training_stats.report("Loss/scores/real", real_logits)
                training_stats.report("Loss/signs/real", real_logits.sign())

                loss_Dreal = 0
                if phase in ["Dmain", "Dboth"]:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    if img_txt_d > 0.:
                        loss_Dreal = loss_Dreal - img_txt_d*self.contra_loss(temp, real_d_fts, sent_fts, lam).mean()
                    training_stats.report("Loss/D/loss", loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ["Dreg", "Dboth"]:
                    with torch.autograd.profiler.record_function("r1_grads"), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------
