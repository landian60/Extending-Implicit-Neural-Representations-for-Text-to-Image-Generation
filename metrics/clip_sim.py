# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
import torch
import dnnlib
import clip
from training.clip_txt import CLIP_TXT_ENCODER as TXT_CLIP
import copy

from . import metric_utils


# ----------------------------------------------------------------------------

    
def compute_CLIP_sim(opts, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    clip_score = metric_utils.compute_CLIP_sim_for_generator(
        opts=opts,
        rel_lo=0,
        rel_hi=1,
        capture_mean_cov=True,
        max_items=num_gen,
    )



    if opts.rank != 0:
        return float("nan")
    
    return float(clip_score)

def compute_clipR(opts, num_gen):
    clip_R = metric_utils.compute_CLIP_R_precision_for_generator(
        opts=opts,
        rel_lo=0,
        rel_hi=1,
        capture_all=True,
        max_items=num_gen,
    ).get_all_torch()
    
    if opts.rank != 0:
        return float("nan")
    num_chunks=10
    clip_R = clip_R.squeeze(1)
    print("clip_R.shape",clip_R.shape)
    predictions = []
    for i in range(clip_R.shape[0]):
        if clip_R[i] == 0:
            predictions.append(1)
        else:
            predictions.append(0)

    num_preds = len(predictions)
    chunk_size = int(num_preds / num_chunks)

    predictions = torch.Tensor(list(map(float, predictions)))
    # print("predictions_pre",predictions)
    torch.randperm(num_preds, out=torch.arange(num_preds))  # 随机打乱顺序
    # print("predictions_after",predictions)
    chunks = torch.zeros(num_chunks)
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunks[i] = torch.mean(predictions[start:end])
    # print("chunks",chunks.shape)
    return float(torch.mean(chunks).cpu()), float(torch.std(chunks).cpu())

def compute_clipRTrue(opts, num_gen):
    clip_R = metric_utils.compute_CLIP_R_precision_for_dataset(
        opts=opts,
        rel_lo=0,
        rel_hi=1,
        capture_all=True,
        max_items=num_gen,
    ).get_all_torch()
    
    if opts.rank != 0:
        return float("nan")
    num_chunks=10
    clip_R = clip_R.squeeze(1)
    # print("clip_R.shape",clip_R.shape)
    predictions = []
    for i in range(clip_R.shape[0]):
        if clip_R[i] == 0:
            predictions.append(1)
        else:
            predictions.append(0)

    num_preds = len(predictions)
    chunk_size = int(num_preds / num_chunks)

    predictions = torch.Tensor(list(map(float, predictions)))
    # print("predictions_pre",predictions)
    torch.randperm(num_preds, out=torch.arange(num_preds))  # 随机打乱顺序
    # print("predictions_after",predictions)
    chunks = torch.zeros(num_chunks)
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunks[i] = torch.mean(predictions[start:end])
    # print("chunks",chunks.shape)
    return float(torch.mean(chunks).cpu()), float(torch.std(chunks).cpu())


# ----------------------------------------------------------------------------
