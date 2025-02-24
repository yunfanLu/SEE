import os
import random
import sys
from datetime import timedelta
from os import listdir
from os.path import exists, isdir, isfile, join, splitext
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import debug, error, flags, info, warn
from pudb import set_trace
from scipy import special
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELBC
from see.datasets.basic_batch import get_ev_low_light_batch
from see.utils.event_representation_builder import EventRepresentationBuilder

"""
This dataset is contributed by:
@inproceedings{liang2024towards,
  title={Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel
    Approach},
  author={Liang, Guoqiang and Chen, Kanghao and Li, Hangyu and Lu, Yunfan and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23--33},
  year={2024}
}
"""


class SeeingDynamicWithEventVideoDataset(Dataset):
    def __init__(self, video_root, in_frames, crop_h, crop_w, ev_rep_cfg, is_training):
        """
        video_root: str, the root directory of the dataset. e.g. "dataset/CVPR24/0-Low-Light-CVPR24/event_in/train/i_0"

        The structure of this dataset is:
            ├── event_in
                ├── i_0
                │   ├── low
                        ├── <timestamp-1>.png
                        ├── <timestamp-1>.npz
                        ├── <timestamp-2>.png
                        ...
                │   └── normal
            ├── event_out
                ├── o_90
                ├── o_93
        """
        super().__init__()
        assert in_frames % 2 == 1, "input frames must be odd number"

        self.root = video_root
        self.video = video_root.split("/")[-1]
        self.H = 260
        self.W = 346
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.in_frames = in_frames
        self.is_training = is_training
        self.erpcfg = ev_rep_cfg
        # load dataset
        self.normal_root = join(video_root, "normal")
        self.lowlgt_root = join(video_root, "low")
        # load all files
        self.normal_frames = sorted([f for f in listdir(self.normal_root) if f.endswith(".png")])
        self.lowlgt_frames = sorted([f for f in listdir(self.lowlgt_root) if f.endswith(".png")])
        self.lowlgt_events = sorted([f for f in listdir(self.lowlgt_root) if f.endswith(".npz")])
        if "lowlight_event.npz" in self.lowlgt_events:
            self.lowlgt_events.remove("lowlight_event.npz")
        # event representation builder
        self.using_event = self.erpcfg.type != "empty"
        self.erpcfg.H = self.H
        self.erpcfg.W = self.W
        self.erbuilder = EventRepresentationBuilder(self.erpcfg)

    def __len__(self):
        return len(self.normal_frames)

    def __getitem__(self, index) -> Any:
        idxs = list(range(index - self.in_frames // 2, index + self.in_frames // 2 + 1))
        idxs = [max(0, min(len(self.normal_frames) - 1, i)) for i in idxs]
        # 1. load data from files
        # 1.1 load events
        if self.using_event:
            event_stream = []
            for idx in [index - self.in_frames // 2 - 1] + idxs:
                idx = max(0, min(len(self.lowlgt_events) - 1, idx))
                event_npy = join(self.lowlgt_root, self.lowlgt_events[idx])
                event = np.load(event_npy)["arr_0"]
                if event.ndim != 2 or event.shape[1] != 4:
                    warn(f"ERROR: Video: {self.video}, Frame: {index}, Event: {event_npy}")
                    continue
                event_stream.append(event)
            events = self.erbuilder(np.concatenate(event_stream))
        else:
            events = np.zeros(shape=(self.erpcfg.channel, self.H, self.W))

        # 1.2 load frames
        lfs, lbs, lis = [], [], []
        nfs, nbs, nis = [], [], []
        for idx in idxs:
            lowlgt_frame_path = join(self.lowlgt_root, self.lowlgt_frames[idx])
            lf, lb, li = self._load_frame_and_blur_and_illmap(lowlgt_frame_path)
            normal_frame_path = join(self.normal_root, self.normal_frames[idx])
            nf, nb, ni = self._load_frame_and_blur_and_illmap(normal_frame_path)
            # load in to list
            lfs.append(lf)
            lbs.append(lb)
            lis.append(li)
            nfs.append(nf)
            nbs.append(nb)
            nis.append(ni)
        lfs = np.concatenate(lfs, axis=0)
        lbs = np.concatenate(lbs, axis=0)
        lis = np.concatenate(lis, axis=0)
        nfs = np.concatenate(nfs, axis=0)
        nbs = np.concatenate(nbs, axis=0)
        nis = np.concatenate(nis, axis=0)
        # 2. data augmentation
        (
            events,
            lowlgt_frames,
            lowlgt_frame_blurs,
            lowlgt_frame_illmaps,
            normal_frames,
            normal_frame_blurs,
            normal_frame_illmaps,
        ) = self._totensor_crop_flip(
            events,
            lfs,
            lbs,
            lis,
            nfs,
            nbs,
            nis,
        )
        # 3. construct batch
        batch = get_ev_low_light_batch()
        batch[ELBC.E] = events
        batch[ELBC.LL] = lowlgt_frames
        batch[ELBC.LLB] = lowlgt_frame_blurs
        batch[ELBC.ILL] = lowlgt_frame_illmaps
        batch[ELBC.NL] = normal_frames
        batch[ELBC.NLB] = normal_frame_blurs
        batch[ELBC.INL] = normal_frame_illmaps
        # 3.1 add filename and video name
        frame_name = self.lowlgt_frames[index].split(".")[0]
        batch[ELBC.FRAME_NAME] = frame_name
        video_name = self.root.split("/")[-1]
        batch[ELBC.VIDEO_NAME] = video_name
        # info(f"Frame: {frame_name}, Video: {video_name}")
        return batch

    def _totensor_crop_flip(self, *chw_ndarrays):
        # To Torch Tensor
        chws = [torch.from_numpy(x) for x in chw_ndarrays]
        # Crop
        crop_h, crop_w = self.crop_h, self.crop_w
        if self.is_training:
            top = random.randint(0, self.H - crop_h) // 4 * 4
            left = random.randint(0, self.W - crop_w) // 4 * 4
        else:
            top, left = 0, 0
        chw_ndarrays = [x[..., top : top + crop_h, left : left + crop_w] for x in chws]
        # Flip for horizontal
        if self.is_training and random.random() < 0.5:
            chws = [x.flip(-1) for x in chws]
        # Flip for vertical
        if self.is_training and random.random() < 0.5:
            chws = [x.flip(-2) for x in chws]
        return chw_ndarrays

    def _load_frame_and_blur_and_illmap(self, image_path):
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_blur = cv2.blur(frame, (5, 5))
        frame = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
        frame_blur = frame_blur.astype(np.float32).transpose(2, 0, 1) / 255.0
        # illumiantion map is the max value of RGB channels.
        frame_illmap = np.max(frame, axis=0, keepdims=True)
        return frame, frame_blur, frame_illmap


def get_seeing_dynamic_with_event_dataset_forset(root, in_frames, crop_h, crop_w, ev_rep_cfg, is_training):
    all_videos = sorted(listdir(root))
    datasets = []
    for video in all_videos:
        video_root = join(root, video)
        datasets.append(
            SeeingDynamicWithEventVideoDataset(video_root, in_frames, crop_h, crop_w, ev_rep_cfg, is_training)
        )
    return ConcatDataset(datasets)


def get_seeing_dynamic_with_event_dataset(dataset_root, in_frames, crop_h, crop_w, ev_rep_cfg):
    train_set_root = join(dataset_root, "train")
    test_set_root = join(dataset_root, "test")
    train_set = get_seeing_dynamic_with_event_dataset_forset(
        train_set_root, in_frames, crop_h, crop_w, ev_rep_cfg, is_training=True
    )
    test_set = get_seeing_dynamic_with_event_dataset_forset(
        test_set_root, in_frames, crop_h, crop_w, ev_rep_cfg, is_training=False
    )
    return train_set, test_set


def get_seeing_dynamic_with_event_dataset_all(root, in_frames, crop_h, crop_w, ev_rep_cfg):
    in_root = join(root, "event_in")
    out_root = join(root, "event_out")
    train_set_in, test_set_in = get_seeing_dynamic_with_event_dataset(in_root, in_frames, crop_h, crop_w, ev_rep_cfg)
    train_set_out, test_set_out = get_seeing_dynamic_with_event_dataset(out_root, in_frames, crop_h, crop_w, ev_rep_cfg)
    train_set_all = ConcatDataset([train_set_in, train_set_out])
    test_set_all = ConcatDataset([test_set_in, test_set_out])
    return train_set_all, test_set_all
