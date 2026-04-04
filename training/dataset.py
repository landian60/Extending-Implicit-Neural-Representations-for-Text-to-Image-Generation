# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import json
import os
import zipfile
import sys

import dnnlib
import numpy as np
import PIL.Image
import torch
from PIL import Image
import clip as clip
import numpy.random as random
import pandas as pd
import torchvision.transforms as transforms
try:
    import pyspng
except ImportError:
    pyspng = None
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    
# ----------------------------------------------------------------------------


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
        use_clip = False,
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._use_clip = use_clip

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------
def load_clip(clip_info, device=torch.device("cuda")):
    import clip as clip
    model,_ = clip.load(clip_info['type'], device=device)
    return model

def get_fix_data(train_dl, test_dl, text_encoder, args):
    ''' return fixed_image, fixed_sent, fixed_word, fixed_noise, captions_train, captions_test
    '''
    fixed_image_train, captions_train, _, fixed_sent_train, fixed_word_train, fixed_key_train = get_one_batch_data(train_dl, text_encoder, args)
    fixed_image_test, captions_test, _, fixed_sent_test, fixed_word_test, fixed_key_test= get_one_batch_data(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device) # z_dim = 100 in birds
    return fixed_image, fixed_sent, fixed_word, fixed_noise, captions_train, captions_test


def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, args.device)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys


def prepare_data(data, text_encoder, device):
    imgs, captions, CLIP_tokens, keys = data
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis] # HW -> HWC
    img = img.transpose(2, 0, 1) # HWC -> CHW
    if normalize is not None:
        img = normalize(img)
    return img


def get_tokens(cap_path,clip_info, device=torch.device("cuda")):
    # 随机选择一个caption并做tokenize，返回选定的caption和tokens[0]
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    sent_ix = random.randint(0, len(eff_captions))
    caption = eff_captions[sent_ix]
    txt_tokens = clip.tokenize(caption, truncate=True)
    # model = load_clip(clip_info,device)
    # txt_fts = model.encode_text(tokens)
    # img_fts = model.encode_image(imgs)
    return caption, txt_tokens[0]

class TextImgDataset(Dataset):
    def __init__(self, split, 
                 transform=None, 
                 clip4text=None, 
                 data_dir=None, 
                 dataset_name=None,
                 resolution=None, # Ensure specific resolution, None = highest available.
                **super_kwargs,         # Additional arguments for the Dataset base class.
                 ):
        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize(int(resolution * 76 / 64)),
                transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip(),
                ])
        self.transform = image_transform
        self.clip4text = clip4text
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.norm = None
        # self.norm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ])
        self.split=split
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.filenames = self.load_filenames(self.data_dir, split)
        self.number_example = len(self.filenames)
        raw_shape = [self.number_example] + list(self._load_raw_image(0).shape) # image num,c,h,w
        super().__init__(name=dataset_name, raw_shape=raw_shape, **super_kwargs)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        # print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames} # 字典的value是空的
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            # print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames
    
    def _load_raw_data(self, raw_idx):
        for raw_idx in self.number_example:
            key = self.filenames[raw_idx]
            data_dir = self.data_dir
            #
            if self.bbox is not None:
                bbox = self.bbox[key]
            else:
                bbox = None
            #
            if self.dataset_name.lower().find('coco') != -1:
                if self.split=='train':
                    img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key)
                else:
                    img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key)
            elif self.dataset_name.lower().find('cc3m') != -1:
                if self.split=='train':
                    img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
                else:
                    img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
            elif self.dataset_name.lower().find('cc12m') != -1:
                if self.split=='train':
                    img_name = '%s/images/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
                else:
                    img_name = '%s/images/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        caps, txttokens = get_tokens(text_name,self.clip4text)
        return imgs, caps, txttokens

    def get_mismatched_caption(self, indices):
        mismatch_captions = []
        mismatch_tokens = []
        data_dir = self.data_dir
        while len(mismatch_captions) < 9:
            sampled_idx = random.randint(0, self.number_example-1)
            key = self.filenames[sampled_idx]
            if sampled_idx in indices:
                continue
            #
            if self.dataset_name.lower().find('coco') != -1:
                if self.split=='train':
                    # img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key)
                else:
                    # img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key)
            elif self.dataset_name.lower().find('cc3m') != -1:
                if self.split=='train':
                    # img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
                else:
                    # img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
            elif self.dataset_name.lower().find('cc12m') != -1:
                if self.split=='train':
                    # img_name = '%s/images/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
                else:
                    # img_name = '%s/images/%s.jpg' % (data_dir, key)
                    text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                # img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            caps, txttokens = get_tokens(text_name,self.clip4text)
            mismatch_captions.append(caps)
            mismatch_tokens.append(txttokens)

        # mismatch_captions = torch.cat(mismatch_captions, 0).long()
        # mismatch_tokens = torch.cat(mismatch_tokens,0).long()

        return mismatch_captions, mismatch_tokens

    def _load_raw_image(self, raw_idx):
        key = self.filenames[raw_idx]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        #
        if self.dataset_name.lower().find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
                # text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
                # text_name = '%s/text/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('cc3m') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                # text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                # text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
        elif self.dataset_name.lower().find('cc12m') != -1:
            if self.split=='train':
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                # text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                # text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
        else:
            img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
            # text_name = '%s/text/%s.txt' % (data_dir, key)
        image = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        
        return image
        
    def _load_raw_labels(self):
        return None

    def _load_clip_img_features(self):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            clip_features = json.load(f)['clip_img_features']
        if clip_features is None:
            return None
        clip_features = dict(clip_features)
        clip_features = [clip_features[fname.replace('\\', '/')] for fname in self._image_fnames]
        clip_features = np.array(clip_features)
        clip_features = clip_features.astype(np.float32)
        return clip_features

    def _load_clip_txt_features(self):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            clip_features = json.load(f)['clip_txt_features']
        if clip_features is None:
            return None
        clip_features = dict(clip_features)
        clip_features = [clip_features[fname.replace('\\', '/')] for fname in self._image_fnames]
        return clip_features
    
    def __getitem__(self, index):
        #
        key = self.filenames[index]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        #
        if self.dataset_name.lower().find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('cc3m') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
        elif self.dataset_name.lower().find('cc12m') != -1:
            if self.split=='train':
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
        else:
            img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
            text_name = '%s/text/%s.txt' % (data_dir, key)
        #
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        caps,txt_tokens = get_tokens(text_name,self.clip4text)
        return imgs, self.get_label(index), txt_tokens, caps

    def __len__(self):
        return len(self.filenames)
    
    
class ImageFolderDataset(Dataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        resolution=None,  # Ensure specific resolution, None = highest available.
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()
        self._image_fnames = sorted(
            fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == ".png":
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = "dataset.json"
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


# ----------------------------------------------------------------------------
