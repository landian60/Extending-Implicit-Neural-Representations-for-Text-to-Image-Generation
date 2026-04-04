# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import math
from typing import List, Optional, Tuple, Union
# os.environ['ENABLE_FLOPS_STATISTICS'] = '-2'
# os.environ['MODEL_PARAMS_THOP'] = '-2'
import click
import dnnlib
import legacy
import numpy as np
import PIL.Image
import torch

import clip
from training.clip_txt import CLIP_TXT_ENCODER as TXT_CLIP

# ----------------------------------------------------------------------------

def parse_start_points(s: Union[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Parse a comma separated list of start points and return a list of tuples.

    Example: '(0.0,0.0),(0.0,0.5),(0.5,0.0),(0.5,0.5)' returns [(0.0,0.0), (0.0,0.5), (0.5,0.0), (0.5,0.5)]
    """
    if isinstance(s, list):
        return s
    if s is None or s == "":
        return []
    points = []
    point_re = re.compile(r"\((-?\d+\.\d+),(-?\d+\.\d+)\)")
    for p in point_re.findall(s):
        points.append((float(p[0]), float(p[1])))
    return points


def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(",")
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"cannot parse 2-vector {s}")


# ----------------------------------------------------------------------------


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------

def make_coords(resolution: float, scale: float):
    coords = torch.linspace(-0.5, 1.5, resolution * scale)
    coords = coords.reshape(1, -1, 1, 1)
    coords = coords.repeat(1, 1, 2, 1)
    return coords


def slugify_text(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return slug.strip("-") or "prompt"


def resolve_query_points(
    startpoints: List[Tuple[float, float]], startw: float, starth: float
) -> List[Tuple[float, float]]:
    """Use explicit startpoints when provided, otherwise fall back to one window."""
    return startpoints if len(startpoints) > 0 else [(startw, starth)]


def get_tile_crop_ratio(mode: str, height: float) -> float:
    """Crop each generated tile before building the preview mosaic.

    The current CLI produces a simple tiled preview rather than a geometry-aware stitch.
    These crop ratios preserve the previous visualization behavior.
    """
    if mode == "interpolation":
        return 1 / height / 2
    if mode == "extrapolation":
        return 1 / height
    raise ValueError(f"unsupported mode: {mode}")


def crop_preview_tile(img: PIL.Image.Image, crop_ratio: float) -> PIL.Image.Image:
    """Center-crop the visible portion used in the combined preview image."""
    crop_w = int(img.width * crop_ratio)
    crop_h = int(img.height * crop_ratio)
    return img.crop((0, 0, crop_w, crop_h))


def combine_tiles(img_list: List[PIL.Image.Image]) -> PIL.Image.Image:
    """Build a simple row-major contact sheet from individually rendered windows.

    This is intentionally a lightweight preview mosaic, not a geometry-aware stitch.
    """
    if len(img_list) == 1:
        return img_list[0]

    cols = math.ceil(math.sqrt(len(img_list)))
    rows = math.ceil(len(img_list) / cols)
    tile_w, tile_h = img_list[0].size
    canvas = PIL.Image.new("RGB", (tile_w * cols, tile_h * rows))

    for idx, tile in enumerate(img_list):
        col = idx % cols
        row = idx // cols
        canvas.paste(tile, (col * tile_w, row * tile_h))

    return canvas

@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option(
    "--seeds",
    type=parse_range,
    help="List of random seeds (e.g., '0,1,4-6')",
    required=True,
)
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option(
    "--class",
    "class_idx",
    type=int,
    help="Class label (unconditional if not specified)",
)
@click.option(
    "--noise-mode",
    help="Noise mode",
    type=click.Choice(["const", "random", "none"]),
    default="const",
    show_default=True,
)
@click.option(
    "--translate",
    help="Translate XY-coordinate (e.g. '0.3,1')",
    type=parse_vec2,
    default="0,0",
    show_default=True,
    metavar="VEC2",
)
@click.option(
    "--rotate",
    help="Rotation angle in degrees",
    type=float,
    default=0,
    show_default=True,
    metavar="ANGLE",
)
@click.option(
    "--scale",
    help="Scale of output image",
    type=float,
    default=1.0,
    show_default=True,
    metavar="ANGLE",
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    type=str,
    required=True,
    metavar="DIR",
)
@click.option("--text", "prompt", help="Text prompt", type=str, required=True)
@click.option(
    "--starth",
    help="0.5 will be the center",
    type=float,
    default=0.5,
    show_default=True,
    metavar="ANGLE",
)
@click.option(
    "--startw",
    help="0.5 will be the center",
    type=float,
    default=0.5,
    show_default=True,
    metavar="ANGLE",
)
@click.option(
    "--mode",
    help="mode about interpolation or extrapolation",
    type=click.Choice(["interpolation", "extrapolation"]),
    default="interpolation",
    show_default=True,
)
@click.option(
    "--height",
    help="2.0 will be the double size of orginal grid",
    type=float,
    default=1.0,
    show_default=True,
    metavar="ANGLE",
)
@click.option(
    "--startpoints",
    type=parse_start_points,
    default="",
    help="Optional list of start points for multi-window generation. When omitted, --starth/--startw define a single window.",
)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    prompt: str,
    translate: Tuple[float, float],
    rotate: float,
    scale: int,
    height: float,
    starth: float,
    startw: float,
    mode: float,
    startpoints: List[Tuple[float, float]],
    class_idx: Optional[int],
):
    """Generate images using pretrained network pickle.

    \b
    python gen_images.py --network=network-snapshot.pkl --outdir=out \\
        --text="a public transit bus on a city street" --seeds=0-3
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException("Must specify class label with --class when using a conditional network")
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    text = [prompt]
    prompt_slug = slugify_text(prompt)
    with torch.no_grad():
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.eval().float()
        tokenized_text = clip.tokenize(text).to(device)
        text_encode = TXT_CLIP(clip_model)
        sent_emb, txt_emb = text_encode(tokenized_text)
        f_sent = sent_emb.to(device)
        f_sent /= f_sent.norm(dim=-1, keepdim=True)
        
        f_txt = txt_emb.to(device)
        f_txt /= f_txt.norm(dim=-1, keepdim=True)

    # Each query point renders one fixed-resolution window. When multiple points are
    # provided, the script saves both the individual tiles and a simple combined mosaic.
    query_points = resolve_query_points(startpoints, startw, starth)
    tile_crop_ratio = get_tile_crop_ratio(mode, height)

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, *G.input_shape[1:])).to(device)
        z = z.float()
        tile_images = []
        # if hasattr(G.synthesis, "input"):
        #     m = make_transform(translate, rotate)
        #     m = np.linalg.inv(m)
        #     G.synthesis.b4.input.transform.copy_(torch.from_numpy(m))
        #     print("tranform...")
            
        # Construct optional coordinate inputs for interpolation or extrapolation.
        # if hasattr(G.synthesis.b4, "input"):
        #     # 第一次是-1,2；第二、三次是-0.5,1.5
        #     G.synthesis.b4.input.coords = make_coords(G.img_resolution, scale).to(device)
        # print("startpoints",startpoints)
        for startpoint in query_points:
            startw_, starth_ = startpoint
            print(startw_, starth_)
            # Keep the queried coordinate count fixed at the training-time image
            # resolution and move the queried window via start_h/start_w/height.
            coords = G.head_pos_enc.make_grid2d(n_height=G.img_resolution,n_width=G.img_resolution,start_h=starth_,start_w=startw_, # start_h和start_w是相对于原图的起始点位置
                        width=height, height=height)
            # coords = coords
            img = G(z, label, fsts=f_sent, fts=f_txt, coords=coords, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
            tile_images.append(crop_preview_tile(img_pil, tile_crop_ratio))
            img_pil.save(f"{outdir}/{mode}-starth{starth_}-startw{startw_}-size{height}-seed{seed:04d}-text-{prompt_slug}.png")

        combined_img = combine_tiles(tile_images)
        combined_img.save(f"{outdir}/combined-seed{seed:04d}-text-{prompt_slug}.png")

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
