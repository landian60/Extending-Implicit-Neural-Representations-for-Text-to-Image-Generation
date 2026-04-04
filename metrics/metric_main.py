# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main API for computing and reporting quality metrics."""

import json
import os
import time

import dnnlib
import torch

from . import (
    frechet_inception_distance,
    metric_utils,
    clip_sim,
)


# ----------------------------------------------------------------------------

_metric_dict = dict()  # name => fn


def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn


def is_valid_metric(metric):
    return metric in _metric_dict


def list_valid_metrics():
    return list(_metric_dict.keys())


# ----------------------------------------------------------------------------


def calc_metric(metric, **kwargs):  # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)
    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results=dnnlib.EasyDict(results),
        metric=metric,
        total_time=total_time,
        total_time_str=dnnlib.util.format_time(total_time),
        num_gpus=opts.num_gpus,
    )


# ----------------------------------------------------------------------------


# def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
#     metric = result_dict["metric"]
#     assert is_valid_metric(metric)
#     if run_dir is not None and snapshot_pkl is not None:
#         snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

#     jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
#     print(jsonl_line)
#     if run_dir is not None and os.path.isdir(run_dir):
#         with open(os.path.join(run_dir, f"metric-{metric}.jsonl"), "at") as f:
#             f.write(jsonl_line + "\n")
def report_metric(result_dict, run_dir=None, snapshot_pkl=None, eval_result='', cal=False):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    jsonl_line = jsonl_line + '\n'
    if run_dir is not None and os.path.isdir(run_dir):
        if cal:
        # oss的保存无法追加，使用eval_result保存之前的结果；如果计算指标则添加后缀名
            with open(os.path.join(run_dir, f'stats-metric-{metric}-{snapshot_pkl}.jsonl'), 'at') as f:
                f.write(jsonl_line + eval_result)
        else:
            with open(os.path.join(run_dir, f'stats-metric-{metric}.jsonl'), 'at') as f:
                f.write(jsonl_line + eval_result)
    return jsonl_line

# ----------------------------------------------------------------------------
# Recommended metrics.

@register_metric
def clipsim(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    clip_score = clip_sim.compute_CLIP_sim(opts, num_gen=30000)
    return dict(clipsim=clip_score)

@register_metric
def clipRprecision(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    R_mean, R_std = clip_sim.compute_clipR(opts, num_gen=10000)
    return dict(R_mean=R_mean, R_std=R_std)

@register_metric
def clipRprecisionTrue(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    R_mean, R_std = clip_sim.compute_clipRTrue(opts, num_gen=10000)
    return dict(R_mean=R_mean, R_std=R_std)

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=30000) #num_gen不起作用，不过确实是30000
    return dict(fid50k_full=fid)

