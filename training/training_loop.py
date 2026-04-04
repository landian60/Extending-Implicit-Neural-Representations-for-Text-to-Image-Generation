# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import copy
import json
import os
import pickle
import time

import dnnlib
import legacy
import numpy as np
import PIL.Image
import psutil
import torch
from metrics import metric_main
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
import torchvision

import clip
from training.clip_txt import CLIP_TXT_ENCODER as TXT_CLIP

# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 8)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 8)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels,_,_ = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    training_set_kwargs={},  # Options for training set.
    testing_set_kwargs={},
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    G_kwargs={},  # Options for generator network.
    D_kwargs={},  # Options for discriminator network.
    G_opt_kwargs={},  # Options for generator optimizer.
    D_opt_kwargs={},  # Options for discriminator optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
    loss_kwargs={},  # Options for loss function.
    metrics=[],  # Metrics to evaluate during training.
    random_seed=0,  # Global random seed.
    num_gpus=1,  # Number of GPUs participating in the training.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,  # Number of samples processed at a time by one GPU.
    ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,  # Initial value of augmentation probability.
    ada_target=None,  # ADA target value. None = fixed p.
    ada_interval=4,  # How often to perform ADA adjustment?
    ada_kimg=500,  # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,  # Progress snapshot interval.
    image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
    resume_pkl=None,  # Network pickle to resume training from.
    resume_kimg=0,  # First kimg to report when resuming training.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    abort_fn=None,  # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
    fts_dim                 = 512,
    d_use_norm              = True,
    d_use_fts               = True,
    mixing_prob             = 0., 
    lam                     = 0.5,
    temp                    = 1.,
    gather                  = False,
    itd                     = 10.,
    itc                     = 10.,
    metric_only_test        = False,
    cache_dir               = None,
    
    # multi scale
    multiscale              = True,
    multiscale_chance       = 0.20,
    extra_scale             = 2,
    min_resolution          = 0.5,
    max_resolution          = 1.0,

):
    # Initialize.
    start_time = time.time()
    device = torch.device("cuda", rank)
    # device = torch.device('cuda')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    k = G_kwargs.k
    # Load training set.
    if rank == 0:
        print("Loading training set...")
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed
    )
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set,
            sampler=training_set_sampler,
            batch_size=batch_size // num_gpus,
            **data_loader_kwargs,
        )
    )
    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print("Label shape:", training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(
        c_dim=training_set.label_dim,
        img_resolution=training_set.resolution,
        img_channels=training_set.num_channels,
    )
    G = (
        dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    )  # subclass of torch.nn.Module
    D = (
        dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    )  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [("G", G), ("D", D), ("G_ema", G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, *G.input_shape[1:]], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        fsts = torch.empty([batch_gpu, fts_dim], device=device)
        if k != 1:
            fts = torch.empty([batch_gpu, 77, fts_dim], device=device) # TODO:try to module the 77 as optional
        else:
            fts = None
        img = misc.print_module_summary(G, [z, c, fsts, fts])
        misc.print_module_summary(D, [img, c, fsts])

    # Setup augmentation.
    if rank == 0:
        print("Setting up augmentation...")
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = (
            dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        )  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex="Loss/signs/real")

    # Distribute across GPUs.
    if rank == 0:
        print(f"Distributing across {num_gpus} GPUs...")
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)
 
    # Setup training phases.
    if rank == 0:
        print("Setting up training phases...")
    loss = dnnlib.util.construct_class_by_name(
        device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs
    )  # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [
        ("G", G, G_opt_kwargs, G_reg_interval),
        ("D", D, D_opt_kwargs, D_reg_interval),
    ]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(
                params=module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "both", module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta**mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(
                module.parameters(), **opt_kwargs
            )  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "main", module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + "reg", module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(
            images,
            os.path.join(run_dir, "reals.png"),
            drange=[0, 255],
            grid_size=grid_size,
        )
        grid_z = torch.randn([labels.shape[0], *G.input_shape[1:]], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(
            images,
            os.path.join(run_dir, "fakes_init.png"),
            drange=[-1, 1],
            grid_size=grid_size,
        )
        if training_set_kwargs.dataset_name == 'birds':
            text = ['This is a white and grey bird with black wings and black stripe by its eyes', 
                    'A bird with a brown and black wings, red crown and throat and the bill is short pointed',
                    'This bird has a yellow throat, belly, abdomen and sides with lots of brown streaks on them',
                    'This bird has a white belly and breast, with a blue crown and nape']
        if training_set_kwargs.dataset_name == 'coco':
            text = ['A living area with a television and a table',
                    'A child eating a birthday cake near some balloons',
                    'A small kitchen with low a ceiling',
                    'A group of skiers are preparing to ski down a mountain',
                    'A school bus in the forest',
                    'A green train is coming down the tracks']
            
#         text = ['this is a sneaker shoes, with red lace up closure',
#                'a women, high heel shoes made of brown leather',
#                'a slipper for men']


    
#         text = ['A man wearing glasses with beard, he has brown hair',
#                'A woman with long blonde hair and earrings, she is smiling',
#                'A man has blue hair and no beard']
    
    
        with torch.no_grad():
            clip_model, _ = clip.load("ViT-B/32", device=device, download_root=cache_dir)  # Load CLIP model here
            # previous_w_list = [None]*len(grid_c)
            clip_model = clip_model.eval().float()
            sent_fts_list = []
            txt_fts_list = []
            for step in range(len(text)):
                tokenized_text = clip.tokenize(text[step]).to(device)
                text_encode = TXT_CLIP(clip_model)
                sent_emb, txt_emb = text_encode(tokenized_text)
                # txt_fts = clip_model.encode_text(tokenized_text).view(1, -1).repeat(labels.shape[0], 1)
                sent_emb = sent_emb.view(1, -1).repeat(labels.shape[0], 1)
                txt_emb = txt_emb.view(1, txt_emb.shape[-2],txt_emb.shape[-1]).repeat(labels.shape[0], 1, 1)
                
                f_sent = sent_emb.to(device)
                f_sent /= f_sent.norm(dim=-1, keepdim=True)
                f_sent_ = f_sent.split(batch_gpu)
                sent_fts_list.append(f_sent_)
                
                f_txt = txt_emb.to(device)
                f_txt /= f_txt.norm(dim=-1, keepdim=True)
                f_txt_ = f_txt.split(batch_gpu)
                txt_fts_list.append(f_txt_)
                
                img_list = []
    #                 current_w_list = []
                for s, t, z, c in zip(f_sent_, f_txt_, grid_z, grid_c):
                    if k != 1:
                        img_ = G_ema(z=z, c=c, fsts=s, fts=t, noise_mode='const')
    #                     img_, w_ = G_ema(z=z, c=c, fts=f, return_w=True, w=w, )
                    else:
                        img_ = G_ema(z=z, c=c, fsts=s, fts=None, noise_mode='const')
                    img_list.append(img_)
    #                     current_w_list.append(w_)
    #                 previous_w_list = current_w_list
                # print([img.size() for img in img_list])
                images = torch.cat([img.cpu() for img in img_list]).numpy()            
                save_image_grid(images, os.path.join(run_dir, 'fakes_init_txt_step_%i.png'%step), drange=[-1,1], grid_size=grid_size)   
            del  clip_model

    # Initialize logs.
    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_jsonl_save = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl_save = 'stats:'
        # stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        try:
            import torch.utils.tensorboard as tensorboard

            stats_tfevents = tensorboard.SummaryWriter(os.path.join(run_dir,"tensorboard"))
        except ImportError as err:
            print("Skipping tfevents export:", err)

    # Train.
    if rank == 0:
        print(f"Training for {total_kimg} kimg...")
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function("data_fetch"):
            phase_real_img, phase_real_c, phase_txt_tokens, phase_real_txt = next(training_set_iterator)
            # phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_img, phase_txt_tokens = phase_real_img.to(device), phase_txt_tokens.to(device)
            def transf_to_CLIP_input(inputs): # 实际上没有用到
                device = inputs.device
                if len(inputs.size()) != 4:
                    raise ValueError('Expect the (B, C, X, Y) tensor.')
                else:
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                        .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
                    var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                        .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
                    inputs = torch.nn.functional.interpolate(inputs*0.5+0.5, size=(224, 224))
                    inputs = ((inputs+1)*0.5-mean)/var
                    return inputs
            def full_preprocess(img, mode='bicubic', ratio=0.5):
                full_size = img.shape[-2]
                device = img.device
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
                    reshaped_img = torch.nn.functional.interpolate(cropped_img, (224, 224), mode=mode, align_corners=False)
                reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
                norm = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
                reshaped_img = norm(reshaped_img)
                return  reshaped_img.to(device)
            clip_img = transf_to_CLIP_input(phase_real_img) # 没有用到
            # clip_img = full_preprocess(phase_real_img)
            with torch.no_grad():
                clip_model, _ = clip_model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model here
                clip_model = clip_model.eval().float()
                text_encode = TXT_CLIP(clip_model)
                phase_sent_features, phase_txt_features = text_encode(phase_txt_tokens)
                # phase_txt_features = clip_model.encode_text(phase_txt_tokens)
                phase_img_features = clip_model.encode_image(clip_img)
                del  clip_model
            phase_img_features /= phase_img_features.norm(dim=-1, keepdim=True)
            phase_sent_features /= phase_sent_features.norm(dim=-1, keepdim=True)
            phase_txt_features /= phase_txt_features.norm(dim=-1, keepdim=True)
            phase_img_features = phase_img_features.to(device).split(batch_gpu)
            phase_sent_features = phase_sent_features.to(device).split(batch_gpu)
            phase_txt_features = phase_txt_features.to(device).split(batch_gpu)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, *G.input_shape[1:]], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [
                training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)
            ]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c, real_img_feature, real_sent_feature, real_txt_feature in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_img_features, phase_sent_features, phase_txt_features):
                loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    gen_z=gen_z,
                    gen_c=gen_c,
                    gain=phase.interval,
                    cur_nimg=cur_nimg,
                    sent_fts=real_sent_feature,
                    txt_fts=real_txt_feature if k !=1 else None,
                    temp=temp,
                    lam=lam,
                    gather=gather,
                    d_use_fts=d_use_fts,
                    itd=itd,
                    itc=itc
                )
            phase.module.requires_grad_(False)
            
            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + "_opt"):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function("Gema"):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = (
                np.sign(ada_stats["Loss/signs/real"] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            )
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"
        ]
        # fields += [f"G_lr {G_opt_kwargs['lr']:<5.2e}"]
        # fields += [f"D_lr {D_opt_kwargs['lr']:<5.2e}"]

        training_stats.report0("Timing/total_hours", (tick_end_time - start_time) / (60 * 60))
        training_stats.report0("Timing/total_days", (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(" ".join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print("Aborting...")

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode="const").cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(
                images,
                os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}.png"),
                drange=[-1, 1],
                grid_size=grid_size,
            )
            
            for step in range(len(text)):
                img_list = []            
                # style_list = []
                # w_list = []
                for t, s, z, c in zip(txt_fts_list[step],sent_fts_list[step], grid_z, grid_c):
                    img_= G_ema(z=z, c=c, fsts=s, fts=t, noise_mode='const')
#                     img_, w_, style= G_ema(z=z, c=c, fts=f, return_w=True, return_styles=True)
                    img_list.append(img_)
                    # print("style", style)
                    # style_list.append(style)
                    # w_list.append(w_)
                images = torch.cat([img.cpu() for img in img_list]).numpy()            
                save_image_grid(images, os.path.join(run_dir, f'fakes_{cur_nimg//1000:06d}txt_step_{step}.png'), drange=[-1,1], grid_size=grid_size)   
                img_list = []

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(
                G=G,
                D=D,
                G_ema=G_ema,
                augment_pipe=augment_pipe,
                training_set_kwargs=dict(training_set_kwargs),
            )
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r".*\.[^.]+_(avg|ema)")
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl")
            if rank == 0:
                with open(snapshot_pkl, "wb") as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics. Not first time to save time 
        if (snapshot_data is not None) and (len(metrics) > 0) and cur_tick != 0:
            if rank == 0:
                print("Evaluating metrics...")
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    G=snapshot_data["G_ema"],
                    metric_only_test=metric_only_test,
                    cache_dir=cache_dir,
                    testset_kwargs=testing_set_kwargs,
                    dataset_kwargs=training_set_kwargs,
                    data_loader_kwargs = data_loader_kwargs,
                    num_gpus=num_gpus,
                    rank=rank,
                    device=device,
                )
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0("Timing/" + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl_save is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            with open(os.path.join(run_dir, 'stats_log.jsonl'), 'a') as stats_f:
                stats_f.write(json.dumps(fields) + '\n' + '\n')
            stats_jsonl_save = stats_jsonl_save + json.dumps(fields)
            # if cur_tick % 20 == 0:
            #     with open(os.path.join(run_dir, 'stats_total.jsonl'), 'w') as stats_fi:
            #         stats_fi.write(stats_jsonl_save + '\n')
            
            # stats_jsonl.write(json.dumps(fields) + "\n")
            # stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f"Metrics/{name}", value, global_step=global_step, walltime=walltime)
            try:
                stats_tfevents.flush()
            except:
                pass
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print("Exiting...")


# ----------------------------------------------------------------------------
