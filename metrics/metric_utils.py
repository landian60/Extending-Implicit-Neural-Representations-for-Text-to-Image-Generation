# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import copy
import hashlib
import os
import pickle
import time
import uuid

import dnnlib
import numpy as np
import torch

import clip
from training.clip_txt import CLIP_TXT_ENCODER as TXT_CLIP
import torchvision.transforms as T
import time
# import torchvision
import torch.nn.functional as F

# ----------------------------------------------------------------------------


class MetricOptions:
    def __init__(
        self,
        G=None,
        G_kwargs={},
        dataset_kwargs={},
        testset_kwargs={},
        data_loader_kwargs={},
        num_gpus=1,
        rank=0,
        device=None,
        progress=None,
        cache=True,
        metric_only_test=False,
        cache_dir=None,
    ):
        assert 0 <= rank < num_gpus
        self.G = G
        self.G_kwargs = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.testset_kwargs = dnnlib.EasyDict(testset_kwargs)
        self.data_loader_kwargs = dnnlib.EasyDict(data_loader_kwargs)
        self.num_gpus = num_gpus
        self.rank = rank
        self.device = device if device is not None else torch.device("cuda", rank)
        self.progress = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache = cache
        self.cache_dir = cache_dir
        
        self.metric_only_test = metric_only_test


# ----------------------------------------------------------------------------

_feature_detector_cache = dict()


def get_feature_detector_name(url):
    return os.path.splitext(url.split("/")[-1])[0]


def get_feature_detector(url, device=torch.device("cpu"), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = rank == 0
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier()  # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier()  # others follow
    return _feature_detector_cache[key]


# ----------------------------------------------------------------------------


def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c


# ----------------------------------------------------------------------------


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[: self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1)  # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, "rb") as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


# ----------------------------------------------------------------------------


class ProgressMonitor:
    def __init__(
        self,
        tag=None,
        num_items=None,
        flush_interval=1000,
        verbose=False,
        progress_fn=None,
        pfn_lo=0,
        pfn_hi=1000,
        pfn_total=1000,
    ):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (
            self.num_items is None or cur_items < self.num_items
        ):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(
                f"{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}"
            )
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(
                self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items),
                self.pfn_total,
            )

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag=tag,
            num_items=num_items,
            flush_interval=flush_interval,
            verbose=self.verbose,
            progress_fn=self.progress_fn,
            pfn_lo=self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi=self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total=self.pfn_total,
        )


# ----------------------------------------------------------------------------


def compute_feature_stats_for_dataset(
    opts,
    detector_url,
    detector_kwargs,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    # data_loader_kwargs=None,
    max_items=None,
    **stats_kwargs,
):
    if opts.metric_only_test:
        dataset = dnnlib.util.construct_class_by_name(**opts.testset_kwargs)
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if opts.data_loader_kwargs is None:
        # data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        data_loader_kwargs = dict(pin_memory=True, num_workers=12, prefetch_factor=2)
    else:
        data_loader_kwargs = opts.data_loader_kwargs

    # Try to lookup from cache.
    cache_file = None
    dnnlib.util.set_cache_dir(opts.cache_dir)
    if opts.cache:
        # Choose cache file name.
        if opts.metric_only_test:
            args = dict(dataset_kwargs=opts.testset_kwargs, 
                        detector_url=detector_url, 
                        detector_kwargs=detector_kwargs, 
                        stats_kwargs=stats_kwargs)
        else:
            args = dict(dataset_kwargs=opts.dataset_kwargs, 
                        detector_url=detector_url, 
                        detector_kwargs=detector_kwargs, 
                        stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode("utf-8"))
        cache_tag = f"{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}"
        if opts.metric_only_test:
            cache_tag += '-test_only'
        cache_file = dnnlib.make_cache_dir_path("gan-metrics", cache_tag + ".pkl")

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = float(flag.cpu()) != 0

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag="dataset features", num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose,
    )

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _, _, _ in torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=item_subset,
        batch_size=batch_size,
        **data_loader_kwargs,
    ):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + "." + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file)  # atomic
    return stats


# ----------------------------------------------------------------------------


def compute_feature_stats_for_generator(
    opts,
    detector_url,
    detector_kwargs,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    max_items=None,
    **stats_kwargs,
):
    # if batch_gen is None:
    #     batch_gen = min(batch_size, 4)
    # assert batch_size % batch_gen == 0
    batch_gen = batch_size = 16
    dataset = dnnlib.util.construct_class_by_name(**opts.testset_kwargs)
    if opts.data_loader_kwargs is None:
        # data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        data_loader_kwargs = dict(pin_memory=True, num_workers=12, prefetch_factor=2)
    else:
        data_loader_kwargs = opts.data_loader_kwargs

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    if 'celeba' in opts.testset_kwargs['data_dir'].lower():
        num_items = 6000  # 6000 images are used for Multi-modal CelebA-HQ dataset
    else:
        num_items = 30000  # 30000 images are used for MS-COCO, CUB and LN-COCO datasets
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    # assert stats.max_items is not None
    progress = opts.progress.sub(
        tag="generator features",
        num_items=num_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose,
    )
    item_subset = [(i * opts.num_gpus + opts.rank) % len(dataset) for i in range((len(dataset) - 1) // opts.num_gpus + 1)]
    dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
    clip_model, _ = clip.load("ViT-B/32", device=opts.device)
    clip_model = clip_model.eval().float()
    # Main loop.
    while not stats.is_full():
        try:
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
        # images = []
        with torch.no_grad():
            text_encode = TXT_CLIP(clip_model)
            sent_emb, txt_emb = text_encode(txt_tokens.to(opts.device))
            clip_sent_features = sent_emb/sent_emb.norm(dim=-1, keepdim=True)
            clip_txt_features = txt_emb/txt_emb.norm(dim=-1, keepdim=True)#.view((batch_size, -1))
            z = torch.randn([txt_emb.size()[0], *G.input_shape[1:]], device=opts.device)
            imgs = G(z=z, c=_labels.to(opts.device), fsts=clip_sent_features.to(opts.device), fts=clip_txt_features.to(opts.device))
            imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # for _i in range(batch_size // batch_gen):
        #     z = torch.randn([batch_gen, G.z_dim], device=opts.device)
        #     img = G(z=z, c=next(c_iter), **opts.G_kwargs)
        #     img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #     images.append(img)
        # images = torch.cat(images)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat([1, 3, 1, 1])
        features = detector(imgs, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

def compute_CLIP_sim_for_generator(
    opts,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    max_items=None,
    **stats_kwargs,
    ):
    # clip_cos = torch.FloatTensor([0.0]).to(opts.device)
    batch_gen = batch_size = 16
    dataset = dnnlib.util.construct_class_by_name(**opts.testset_kwargs)
    if opts.data_loader_kwargs is None:
        # data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        data_loader_kwargs = dict(pin_memory=True, num_workers=12, prefetch_factor=2)
    else:
        data_loader_kwargs = opts.data_loader_kwargs

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Initialize.
    if 'celeba' in opts.testset_kwargs['data_dir'].lower():
        num_items = 6000  # 6000 images are used for Multi-modal CelebA-HQ dataset
    else:
        num_items = 30000  # 30000 images are used for MS-COCO, CUB and LN-COCO datasets
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    # assert stats.max_items is not None
    progress = opts.progress.sub(
        tag="generator features",
        num_items=num_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    item_subset = [(i * opts.num_gpus + opts.rank) % len(dataset) for i in range((len(dataset) - 1) // opts.num_gpus + 1)]
    dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
    clip_model, _ = clip.load("ViT-B/32", device=opts.device)
    clip_model = clip_model.eval().float()
    # Main loop.
    while not stats.is_full():
        try:
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
        # images = []
        with torch.no_grad():
            text_encode = TXT_CLIP(clip_model)
            sent_emb, txt_emb = text_encode(txt_tokens.to(opts.device))
            clip_sent_features = sent_emb/sent_emb.norm(dim=-1, keepdim=True)
            clip_txt_features = txt_emb/txt_emb.norm(dim=-1, keepdim=True)#.view((batch_size, -1))
            z = torch.randn([txt_emb.size()[0], *G.input_shape[1:]], device=opts.device)
            imgs = G(z=z, c=_labels.to(opts.device), fsts=clip_sent_features.to(opts.device), fts=clip_txt_features.to(opts.device))
            normed_gen_full_img = full_preprocess(imgs)
            img_fts_gen_full = clip_model.encode_image(normed_gen_full_img)
            img_fts_gen_full = img_fts_gen_full/img_fts_gen_full.norm(dim=-1, keepdim=True)
        
            clip_sim = torch.cosine_similarity(img_fts_gen_full, sent_emb)
            # print("clip_sim",clip_sim.shape) # [batch]
        # clip_cos = clip_cos + clip_sim
            stats.append_torch(clip_sim.unsqueeze(1), num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    clip_score, _ = stats.get_mean_cov()
    return clip_score

def compute_CLIP_R_precision_for_generator(
    opts,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    max_items=None,
    **stats_kwargs,
    ):
    # clip_cos = torch.FloatTensor([0.0]).to(opts.device)
    batch_gen = batch_size = 16
    dataset = dnnlib.util.construct_class_by_name(**opts.testset_kwargs)
    if opts.data_loader_kwargs is None:
        # data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        data_loader_kwargs = dict(pin_memory=True, num_workers=12, prefetch_factor=2)
    else:
        data_loader_kwargs = opts.data_loader_kwargs

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Initialize.
    if 'celeba' in opts.testset_kwargs['data_dir'].lower():
        num_items = 6000  # 6000 images are used for Multi-modal CelebA-HQ dataset
    else:
        num_items = 10000  # 30000 images are used for MS-COCO, CUB and LN-COCO datasets
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    # assert stats.max_items is not None
    progress = opts.progress.sub(
        tag="generator features",
        num_items=num_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    item_subset = [(i * opts.num_gpus + opts.rank) % len(dataset) for i in range((len(dataset) - 1) // opts.num_gpus + 1)]
    index = 0
    dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
    clip_model, _ = clip.load("ViT-B/32", device=opts.device)
    clip_model = clip_model.eval().float()
    # Main loop.
    while not stats.is_full():
        try:
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
            current_index = item_subset[index]
        except:
            dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
            current_index = item_subset[index]
        index += 1
        if index > (len(dataset) - 1) // opts.num_gpus:
            index = 0
        # images = []
        indices = torch.arange(current_index, current_index + batch_gen) % len(dataset)
        mismatched_captions, mismatched_tokens = dataset.get_mismatched_caption(indices)
        # print("mismatched_captions",len(mismatched_captions))
        # print("real_txt", len(real_txt))
        with torch.no_grad():
            text_encode = TXT_CLIP(clip_model)
            # print("txt_tokens",txt_tokens.shape)
            sent_emb, txt_emb = text_encode(txt_tokens.to(opts.device))
            clip_mis_sent_features = []
            for i in range(len(mismatched_captions)):
                mis_sent_emb, _ = text_encode(mismatched_tokens[i].unsqueeze(0).to(opts.device))
                clip_mis_sent_features.append(mis_sent_emb/mis_sent_emb.norm(dim=-1, keepdim=True))
                
            clip_sent_features = sent_emb/sent_emb.norm(dim=-1, keepdim=True)
            clip_mis_sent_features_all = torch.cat(clip_mis_sent_features,0)
            clip_mis_sent_features_all = clip_mis_sent_features_all.unsqueeze(0).repeat(clip_sent_features.shape[0],1,1)
            clip_txt_features = txt_emb/txt_emb.norm(dim=-1, keepdim=True)#.view((batch_size, -1))
            z = torch.randn([txt_emb.size()[0], *G.input_shape[1:]], device=opts.device)
            imgs = G(z=z, c=_labels.to(opts.device), fsts=clip_sent_features.to(opts.device), fts=clip_txt_features.to(opts.device))
            normed_gen_full_img = full_preprocess(imgs)
            img_fts_gen_full = clip_model.encode_image(normed_gen_full_img)
            img_fts_gen_full = img_fts_gen_full/img_fts_gen_full.norm(dim=-1, keepdim=True)
            all_texts_features = torch.cat([clip_sent_features.unsqueeze(1), clip_mis_sent_features_all], dim=1)
            # print("all_texts_features",all_texts_features,all_texts_features.shape)
            clip_logits = torch.cosine_similarity(img_fts_gen_full.unsqueeze(1), all_texts_features,dim=-1)
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * clip_logits
            # print("logits_per_image",logits_per_image)
            # 相似度在排序后的序列中的下标
            clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[:, 0]
            # print("clip_prediction",clip_prediction) # [batch]
        # clip_cos = clip_cos + clip_sim
            stats.append_torch(clip_prediction.unsqueeze(1), num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

def compute_CLIP_R_precision_for_dataset(
    opts,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    max_items=None,
    **stats_kwargs,
    ):
    # clip_cos = torch.FloatTensor([0.0]).to(opts.device)
    batch_gen = batch_size = 16
    dataset = dnnlib.util.construct_class_by_name(**opts.testset_kwargs)
    if opts.data_loader_kwargs is None:
        # data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        data_loader_kwargs = dict(pin_memory=True, num_workers=12, prefetch_factor=2)
    else:
        data_loader_kwargs = opts.data_loader_kwargs

    # Setup generator and labels.
    # G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Initialize.
    if 'celeba' in opts.testset_kwargs['data_dir'].lower():
        num_items = 6000  # 6000 images are used for Multi-modal CelebA-HQ dataset
    else:
        num_items = 10000  # 30000 images are used for MS-COCO, CUB and LN-COCO datasets
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    # assert stats.max_items is not None
    progress = opts.progress.sub(
        tag="generator features",
        num_items=num_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    item_subset = [(i * opts.num_gpus + opts.rank) % len(dataset) for i in range((len(dataset) - 1) // opts.num_gpus + 1)]
    index = 0
    dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
    clip_model, _ = clip.load("ViT-B/32", device=opts.device)
    clip_model = clip_model.eval().float()
    # Main loop.
    while not stats.is_full():
        try:
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
            current_index = item_subset[index]
        except:
            dataloader_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_gen, **data_loader_kwargs))
            images, _labels, txt_tokens, real_txt = next(dataloader_iterator)
            current_index = item_subset[index]
        index += 1
        if index > (len(dataset) - 1) // opts.num_gpus:
            index = 0
        
        indices = torch.arange(current_index, current_index + batch_gen) % len(dataset)
        mismatched_captions, mismatched_tokens = dataset.get_mismatched_caption(indices)
        # print("mismatched_captions",mismatched_captions)
        # print("real_txt", real_txt)
        with torch.no_grad():
            text_encode = TXT_CLIP(clip_model)
            # print("txt_tokens",txt_tokens.shape)
            sent_emb, txt_emb = text_encode(txt_tokens.to(opts.device))
            clip_mis_sent_features = []
            for i in range(len(mismatched_captions)):
                mis_sent_emb, _ = text_encode(mismatched_tokens[i].unsqueeze(0).to(opts.device))
                clip_mis_sent_features.append(mis_sent_emb/mis_sent_emb.norm(dim=-1, keepdim=True))
                
            clip_sent_features = sent_emb/sent_emb.norm(dim=-1, keepdim=True)
            clip_mis_sent_features_all = torch.cat(clip_mis_sent_features,0)
            clip_mis_sent_features_all = clip_mis_sent_features_all.unsqueeze(0).repeat(clip_sent_features.shape[0],1,1)
            # clip_txt_features = txt_emb/txt_emb.norm(dim=-1, keepdim=True)#.view((batch_size, -1))
            # z = torch.randn([txt_emb.size()[0], *G.input_shape[1:]], device=opts.device)
            # imgs = G(z=z, c=_labels.to(opts.device), fsts=clip_sent_features.to(opts.device), fts=clip_txt_features.to(opts.device))
            normed_gen_full_img = full_preprocess(torch.tensor(images, dtype=torch.float32))
            # normed_gen_full_img = transf_to_CLIP_input(images.to(opts.device))
            img_fts_gen_full = clip_model.encode_image(normed_gen_full_img.to(opts.device))
            img_fts_gen_full = img_fts_gen_full/img_fts_gen_full.norm(dim=-1, keepdim=True)
            # print("img_fts_gen_full",img_fts_gen_full.shape)
            all_texts_features = torch.cat([clip_sent_features.unsqueeze(1), clip_mis_sent_features_all], dim=1)
            # print("all_texts_features",all_texts_features,all_texts_features.shape)
            clip_logits = torch.cosine_similarity(img_fts_gen_full.unsqueeze(1), all_texts_features,dim=-1)
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image = logit_scale * clip_logits
            print("logits_per_image",logits_per_image)
            # 相似度在排序后的序列中的下标
            clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[:, 0]
            print("clip_prediction",clip_prediction) # [batch]
        # clip_cos = clip_cos + clip_sim
            stats.append_torch(clip_prediction.unsqueeze(1), num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

def full_preprocess(img, mode='bicubic', ratio=0.5):
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
    reshaped_img = normalize()(reshaped_img)
    return  reshaped_img

def normalize():
    return T.Compose([
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def transf_to_CLIP_input(inputs):
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
        return inputs.float()
                

# ----------------------------------------------------------------------------
