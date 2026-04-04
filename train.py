# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import json
import os
os.environ['ENABLE_FLOPS_STATISTICS'] = '-2'
os.environ['MODEL_PARAMS_THOP'] = '-2'
import re
import tempfile

import click
import dnnlib
import torch
from metrics import metric_main
from torch_utils import custom_ops, training_stats
from training import training_loop

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, DictConfig

# ----------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=c.num_gpus,
            )
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=c.num_gpus,
            )

    # Init torch_utils.
    sync_device = torch.device("cuda", rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = "none"

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)


# ----------------------------------------------------------------------------


def launch_training(c, desc, multigpus, outdir, dry_run):
    # dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
    # assert not os.path.exists(c.run_dir)


    # Dry run?
    if dry_run:
        print("Dry run; exiting.")
        return

    # Create output directory.
    if multigpus == True and (c.rank != 0):
        custom_ops.verbosity = 'none'
    else:
        print("Creating output directory...")
        os.makedirs(c.run_dir)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    # Launch processes.
    if (c.rank == 0):
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)
    
        # Print options.
    if c.rank == 0:
        print()
        print("Training options:")
        print(json.dumps(c, indent=2))
        print()
        print(f"Output directory:    {c.run_dir}")
        print(f"Number of GPUs:      {c.num_gpus}")
        print(f"Batch size:          {c.batch_size} images")
        print(f"Training duration:   {c.total_kimg} kimg")
        print(f"Dataset path:        {c.training_set_kwargs.data_dir}")
        print(f"Dataset size:        {c.training_set_kwargs.max_size} images")
        print(f"Dataset resolution:  {c.training_set_kwargs.resolution}")
        print(f"Dataset labels:      {c.training_set_kwargs.use_labels}")
        print(f"Dataset x-flips:     {c.training_set_kwargs.xflip}")
        print()
    
    if multigpus == True and (c.rank != 0):
        print("Launching processes...")
        
    training_loop.training_loop(**c)
    # torch.multiprocessing.set_start_method("spawn")
    # with tempfile.TemporaryDirectory() as temp_dir:
        # if c.num_gpus == 1:
            # subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        # else:
            # torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


# ----------------------------------------------------------------------------


def init_dataset_kwargs(data_cfg):
    try:
        training_set_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.TextImgDataset",
            split='train',
            clip4text={'src':"clip", 'type':'ViT-B/32'} ,
            dataset_name=data_cfg.get('dataset_name', {}),
            resolution=data_cfg.get('resolution', {}),
            data_dir=data_cfg.get('data_dir', {}),
            # use_labels=True,
            # max_size=None,
            xflip=False,
        )
        testing_set_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.TextImgDataset",
            split='test',
            clip4text={'src':"clip", 'type':'ViT-B/32'} ,
            dataset_name=data_cfg.get('dataset_name', {}),
            resolution=data_cfg.get('resolution', {}),
            data_dir=data_cfg.get('data_dir', {}),
            use_labels=False,
            # max_size=None,
            xflip=False,
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # Subclass of training.dataset.Dataset.
        training_set_kwargs.resolution = dataset_obj.resolution  # Be explicit about resolution.
        training_set_kwargs.use_labels = dataset_obj.has_labels  # Be explicit about labels.
        training_set_kwargs.max_size = len(dataset_obj)  # Be explicit about dataset size.
        return training_set_kwargs, testing_set_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f"--data: {err}")


# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


# ----------------------------------------------------------------------------


@click.command()
# Required.
@click.option(
    "--cfg",
    help="Base configuration",
    type=click.Choice(["eidgan"]),
    required=True,
)
# @click.option("--data", help="Training data", metavar="[ZIP|DIR]", type=str, required=True)
# @click.option(
#     "--gpus",
#     help="Number of GPUs to use",
#     metavar="INT",
#     type=click.IntRange(min=1),
#     required=True,
# )
# @click.option(
#     "--batch",
#     help="Total batch size",
#     metavar="INT",
#     type=click.IntRange(min=1),
#     required=True,
# )
# @click.option(
#     "--gamma",
#     help="R1 regularization weight",
#     metavar="FLOAT",
#     type=click.FloatRange(min=0),
#     required=True,
# )
# Optional features.
@click.option(
    "--cond",
    help="Train conditional model",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--mirror",
    help="Enable dataset x-flips",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--aug",
    help="Augmentation mode",
    type=click.Choice(["noaug", "ada", "fixed"]),
    default="ada",
    show_default=True,
)
@click.option("--resume", help="Resume from given network pickle", metavar="[PATH|URL]", type=str)
@click.option(
    "--freezed",
    help="Freeze first layers of D",
    metavar="INT",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
)
# Misc hyperparameters.
@click.option(
    "--p",
    help="Probability for --aug=fixed",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1),
    default=0.2,
    show_default=True,
)
@click.option(
    "--target",
    help="Target value for --aug=ada",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1),
    default=0.6,
    show_default=True,
)
@click.option(
    "--batch-gpu",
    help="Limit batch size per GPU",
    metavar="INT",
    type=click.IntRange(min=1),
)
@click.option(
    "--cbase",
    help="Capacity multiplier",
    metavar="INT",
    type=click.IntRange(min=1),
    default=32768,
    show_default=True,
)
@click.option(
    "--cmax",
    help="Max. feature maps",
    metavar="INT",
    type=click.IntRange(min=1),
    default=512,
    show_default=True,
)
@click.option(
    "--glr",
    help="G learning rate  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
)
@click.option(
    "--dlr",
    help="D learning rate",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0.002,
    show_default=True,
)
@click.option(
    "--map-depth",
    help="Mapping network depth  [default: varies]",
    metavar="INT",
    type=click.IntRange(min=1),
)
@click.option(
    "--mbstd-group",
    help="Minibatch std group size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
)
# Misc settings.
@click.option("--desc", help="String to include in result dir name", metavar="STR", type=str)
@click.option(
    "--metrics",
    help="Quality metrics",
    metavar="[NAME|A,B,C|none]",
    type=parse_comma_separated_list,
    default="fid50k_full",
    show_default=True,
)
@click.option(
    "--kimg",
    help="Total training duration",
    metavar="KIMG",
    type=click.IntRange(min=1),
    default=25000,
    show_default=True,
)
@click.option(
    "--tick",
    help="How often to print progress",
    metavar="KIMG",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
)
@click.option(
    "--snap",
    help="How often to save snapshots",
    metavar="TICKS",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
)
@click.option(
    "--seed",
    help="Random seed",
    metavar="INT",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--fp32",
    help="Disable mixed-precision",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--nobench",
    help="Disable cuDNN benchmarking",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--workers",
    help="DataLoader worker processes",
    metavar="INT",
    type=click.IntRange(min=1),
    default=12,
    show_default=True,
)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
@click.option('--local_rank','--local-rank', type=int, metavar='INT', default=0)

# Configuration
@click.option('--hydra_cfg_name', help='Name of the Hydra hyperparams config', type=str, metavar='STR')
@click.option('--dataset_cfg_name', help='Name of the datasets config', type=str, metavar='STR')
# 必写
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--multigpus', help='Number of GPUs to use [default: 1]', type=bool, metavar='BOOL')

def main(outdir, multigpus, local_rank, **kwargs):
    """Train EIDGAN on a text-image dataset.

    Examples:

    \b
    # Train EIDGAN on MS-COCO with a single process.
    python train.py --cfg=eidgan --outdir=training-runs --multigpus=False \\
        --hydra_cfg_name=eidgan.yml --dataset_cfg_name=dataset/coco.yml

    \b
    # Train EIDGAN with torchrun on multiple GPUs.
    torchrun --nproc_per_node=4 train.py --cfg=eidgan --outdir=training-runs \\
        --multigpus=True --hydra_cfg_name=eidgan.yml --dataset_cfg_name=dataset/coco.yml
    """
    if multigpus:
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        rank = torch.distributed.get_rank()
        num_workers = torch.distributed.get_world_size()
    else:   
        num_workers = 1
        rank = local_rank
    
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.
    initialize(config_path="./configs", job_name="INR-GAN training")
    hydra_cfg = compose(config_name=opts.hydra_cfg_name)
    data_cfg = compose(config_name=opts.dataset_cfg_name)
    
    fts_dim = data_cfg.get('fts_dim', 512)
    assert isinstance(fts_dim, int)
    c.fts_dim = fts_dim
    
    # probability to use img_fts instead of txt_fts, None use
    c.mixing_prob = data_cfg.get('mixing_prob', 0)
    
    c.metric_only_test = data_cfg.get('metric_only_test', False)
    c.cache_dir = data_cfg.get('cache_dir', None)
        
    c.lam = data_cfg.get('lam', 0)
    
    c.temp = data_cfg.get('temp', 0.5)
    
    c.itd = data_cfg.get('itd', 5.)
    c.itc = data_cfg.get('itc', 10.)
    
    # c.gather = data_cfg.get('gather', False)
    
    # muilti scale party
    c.multiscale = data_cfg.get('multiscale', False)
    if c.multiscale:
        c.multiscale_chance = data_cfg.get('multiscale_chance', 0.)
        c.extra_scale = data_cfg.get('extra_scale', 1)
        c.min_resolution = data_cfg.get('min_resolution', 0.5)
        c.max_resolution = data_cfg.get('max_resolution', 1.0)
        
    
    
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=hydra_cfg.generator.get('z_dim', 512), w_dim=hydra_cfg.generator.get('w_dim', 512), fts_dim=512, 
                                  mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(
        # class_name="training.networks_creps_pos.Discriminator",
        class_name=f"training.networks_{opts.cfg}.Discriminator",
        block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(),
    )
    c.G_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=[0, 0.99], eps=1e-8)
    # c.loss_kwargs = dnnlib.EasyDict(class_name="training.loss_scale.StyleGAN2Loss")
    c.loss_kwargs = dnnlib.EasyDict(class_name="training.loss.StyleGAN2Loss")
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=8, prefetch_factor=2) # diffenrent from machine

    # Training set.
    c.training_set_kwargs, c.testing_set_kwargs, dataset_name = init_dataset_kwargs(data_cfg)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException("--cond=True requires labels specified in dataset.json")
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Transformer args
    c.G_kwargs.k = hydra_cfg.generator.get('k', 1)
    c.G_kwargs.mapping_kwargs.transformer = hydra_cfg.generator.get('transformer', True)
    c.G_kwargs.mapping_kwargs.resnet = hydra_cfg.generator.get('resnet', False)
    c.G_kwargs.mapping_kwargs.shared = hydra_cfg.generator.get('shared', False)
    c.G_kwargs.mapping_kwargs.ltnt2ltnt = hydra_cfg.generator.get('ltnt2ltnt', False)
    c.G_kwargs.mapping_kwargs.integration = hydra_cfg.generator.get('integration', "mul")
    c.G_kwargs.mapping_kwargs.norm = hydra_cfg.generator.get('norm', None)
    c.G_kwargs.integration = hydra_cfg.generator.get('integration', "mul")
    c.G_kwargs.num_heads = hydra_cfg.generator.get('num_heads', 1)
    c.G_kwargs.norm = hydra_cfg.generator.get('norm', None)
    c.G_kwargs.attn = hydra_cfg.generator.get('attn', False)
    c.G_kwargs.attn_res = hydra_cfg.generator.get('attn_res', False)
    c.G_kwargs.selfattn = hydra_cfg.generator.get('selfattn', False)
    
    # other model args
    c.G_kwargs.poly = hydra_cfg.generator.get('poly', False) # use poly-inr or not 
    c.G_kwargs.mod_fc = hydra_cfg.generator.get('mod_fc', True) # use modulated_fc or not 
    c.G_kwargs.channel_attn = hydra_cfg.generator.get('channel_attn', False) # use channel attention or not
    c.G_kwargs.spectral_weight = hydra_cfg.generator.get('spectral_weight', False) # use what kind of spectral_weight: False, "channel", "amplitude"
    c.G_kwargs.rank_attn = hydra_cfg.generator.get('rank_attn', False)

    # Hyperparameters & settings.
    c.num_gpus = num_workers
    c.rank = rank
    c.batch_size = data_cfg.get('batch_gpu', 16) * c.num_gpus
    c.batch_gpu = c.batch_size // c.num_gpus
    c.G_kwargs.channel_base  = int(hydra_cfg.generator.get('fmaps', 1) * 32768)
    c.D_kwargs.channel_base = int(hydra_cfg.discriminator.get('fmaps', 1) * 32768)
    
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = 8 if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group #?
    c.loss_kwargs.r1_gamma = hydra_cfg.loss_kwargs.r1_gamma
    c.G_opt_kwargs.lr = 0.002 if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = hydra_cfg.get('optim', {}).get('lr', opts.dlr)
    c.metrics = opts.metrics
    
    c.total_kimg = data_cfg.get('kimg', opts.kimg)
    c.kimg_per_tick = data_cfg.get('tick', opts.tick)
    c.image_snapshot_ticks = c.network_snapshot_ticks = data_cfg.get('snap', 50)
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed

    # new added
    c.D_kwargs.use_norm = hydra_cfg.discriminator.get('d_use_norm', False)
    c.D_kwargs.use_fts = hydra_cfg.discriminator.get('d_use_fts', True)
    c.D_kwargs.block_kwargs.wavelet = hydra_cfg.discriminator.get('wavelet', True)

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException("--batch must be a multiple of --gpus")
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException("--batch must be a multiple of --gpus times --batch-gpu")
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException("--batch-gpu cannot be smaller than --mbstd")
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException(
            "\n".join(["--metrics can only contain the following values:"] + metric_main.list_valid_metrics())
        )

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    superres = data_cfg.get('superres', False)
    if superres:
        c.G_kwargs.class_name = "training.networks_eidgan.SuperresGenerator"
        c.G_kwargs.path_stem = superres
    else:
        c.G_kwargs.class_name = "training.networks_eidgan.Generator"
    c.G_reg_interval = 4
    c.loss_kwargs.pl_weight = 0

    # Augmentation.
    if opts.aug != "noaug":
        c.augment_kwargs = dnnlib.EasyDict(
            class_name="training.augment.AugmentPipe",
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
        )
        if opts.aug == "ada":
            c.ada_target = opts.target
        if opts.aug == "fixed":
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0  # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.conv_clamp = None
        c.D_kwargs.conv_clamp = None
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f"{opts.cfg:s}-{data_cfg.get('description', None)}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    # Launch.
    launch_training(c=c, desc=desc, multigpus=multigpus, outdir=outdir, dry_run=opts.dry_run)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
