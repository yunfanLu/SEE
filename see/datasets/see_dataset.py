import json
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
from tqdm import tqdm

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELBC
from see.datasets.basic_batch import get_ev_low_light_batch
from see.utils.event_representation_builder import EventRepresentationBuilder

"""
video_name-0:
video_name-1:
    frame_events:
        1721524679594711_0_0_1721524679594711_1721524679595711.png
        1721524679594711_1721524679634711.npy
        1721524679594711_1721524679634711_vis.png
        1721524679634711_0_0_1721524679634711_1721524679635711.png
        1721524679634711_1721524679674711.npy
        1721524679634711_1721524679674711_vis.png
        1721524679674711_0_0_1721524679674711_1721524679675711.png
        1721524679674711_1721524679714711.npy
        1721524679674711_1721524679714711_vis.png
"""


class SeeEverythingEveryTimePairedVideoDataset(Dataset):
    def __init__(
        self,
        group_folder,
        input_video,
        normal_video,
        inputs_frame_events,
        outputs_frame_events,
        in_frames,
        crop_h,
        crop_w,
        ev_rep_cfg,
        is_training,
        input_exposure_states,
        sample_step,
        single_output=True,
    ):
        """
        group_folder: Folder containing the group
        input_video: Input video folder name
        normal_video: Normal video folder name
        inputs_frame_events: List of input frame events. [[frame_list], [event_list]]
        outputs_frame_events: List of outputs frame events. [[frame_list], [event_list]]
        in_frames: Number of frames to use for each sample.
        crop_h: Height of the crop.
        crop_w: Width of the crop.
        ev_rep_cfg: Event representation configuration.
        is_training: Whether the dataset is for training or not.
        input_exposure_states: List of input exposure states. "low-light" or "normal-light" or "high-light"
        sample_step: Number of frames to skip when sampling.
        """
        super().__init__()
        self.group_folder = group_folder
        self.input_video = input_video
        self.normal_video = normal_video
        group_name = group_folder.split("/")[-1]
        self.dataset_video_name = f"{group_name}-{input_exposure_states}-{input_video}-{normal_video}"

        self.in_frames_count = in_frames
        self.inputs_frame = inputs_frame_events[0]
        self.inputs_event = inputs_frame_events[1]
        self.outputs_frame = outputs_frame_events[0]
        self.outputs_event = outputs_frame_events[1]
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.is_training = is_training
        self.input_exposure_states = input_exposure_states
        self.erpcfg = ev_rep_cfg
        self.sample_step = sample_step
        self.single_output = single_output
        # DVS 346 camera height and width
        self.H = 260
        self.W = 346
        # event representation builder
        self.using_event = self.erpcfg.type != "empty"
        self.erpcfg.H = self.H
        self.erpcfg.W = self.W
        self.erbuilder = EventRepresentationBuilder(self.erpcfg)
        #
        self.items = self._generate_items()
        # #
        # info(f"Video Group: {self.group_folder}")
        # info(f"  - input video : {self.input_video}, with {self.input_exposure_states} exposure")
        # info(f"  - normal video: {self.normal_video}")
        # info(f"  - length      : {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        event_files, in_frame_files, ou_frame_files = self.items[index]
        if self.using_event:
            event_stream = []
            for event_file in event_files:
                event = np.load(event_file)
                if event.ndim != 2 or event.shape[1] != 4:
                    warn(f"ERROR: Video: {self.video}, Frame: {index}, Event: {event_file}")
                    continue
                event_stream.append(event)
            event_stream = np.concatenate(event_stream, axis=0)
            events = self.erbuilder(event_stream)
        else:
            events = np.zeros(shape=(self.erpcfg.channel, self.H, self.W))
        # 1.2 load frames
        lfs, lbs, lis = [], [], []
        nfs, nbs, nis = [], [], []
        for lowlgt_frame_path, normal_frame_path in zip(in_frame_files, ou_frame_files):
            lf, lb, li = self._load_frame_and_blur_and_illmap(lowlgt_frame_path)
            nf, nb, ni = self._load_frame_and_blur_and_illmap(normal_frame_path)
            # load in to list
            for x, y in zip([lfs, lbs, lis, nfs, nbs, nis], [lf, lb, li, nf, nb, ni]):
                x.append(y)
        [lfs, lbs, lis, nfs, nbs, nis] = [np.concatenate(x, axis=0) for x in [lfs, lbs, lis, nfs, nbs, nis]]
        # 2. data augmentation
        (
            events,
            lq_frames,
            lq_frame_blurs,
            lq_frame_illmaps,
            normal_frames,
            normal_frame_blurs,
            normal_frame_illmaps,
        ) = self._totensor_crop_flip(events, lfs, lbs, lis, nfs, nbs, nis)
        # 3. construct batch
        batch = get_ev_low_light_batch()
        batch[ELBC.E] = events
        batch[ELBC.LQET] = self.input_exposure_states
        batch[ELBC.LL] = lq_frames
        # More information of inputs.
        batch[ELBC.LLB] = lq_frame_blurs
        batch[ELBC.ILL] = lq_frame_illmaps
        if self.single_output:
            CN, H, W = normal_frames.shape
            N = CN // 3
            normal_frames = normal_frames.reshape(N, 3, H, W)
            batch[ELBC.NL] = normal_frames[N // 2]
        else:
            batch[ELBC.NL] = normal_frames
        batch[ELBC.NLB] = normal_frame_blurs
        batch[ELBC.INL] = normal_frame_illmaps
        # 3.1 add filename and video name
        batch[ELBC.FRAME_NAME] = in_frame_files[len(in_frame_files) // 2].split("/")[-1].split(".")[0]
        batch[ELBC.VIDEO_NAME] = self.dataset_video_name
        return batch

    def _generate_items(self):
        # Align the inputs and outputs
        length = min(len(self.inputs_event), len(self.inputs_frame), len(self.outputs_event), len(self.outputs_frame))
        self.inputs_event = self.inputs_event[:length]
        self.inputs_frame = self.inputs_frame[:length]
        self.outputs_event = self.outputs_event[:length]
        self.outputs_frame = self.outputs_frame[:length]
        #
        items = []
        bias = self.in_frames_count // 2
        for i in range(bias + 1, length - bias - 1, self.sample_step):
            idxs = list(range(i - bias, i + bias + 1))
            # read more events
            in_events = [self.inputs_event[i - bias - 1]] + [self.inputs_event[idx] for idx in idxs]
            in_frames = [self.inputs_frame[idx] for idx in idxs]
            ou_frames = [self.outputs_frame[idx] for idx in idxs]
            # join the goup folder and video to full path.
            in_events = [join(self.group_folder, self.input_video, "frame_event", f) for f in in_events]
            in_frames = [join(self.group_folder, self.input_video, "frame_event", f) for f in in_frames]
            ou_frames = [join(self.group_folder, self.normal_video, "frame_event", f) for f in ou_frames]
            items.append([in_events, in_frames, ou_frames])
        return items

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


def get_see_everything_everytime_with_event_dataset_for_each_group(
    group_folder, in_frames, crop_h, crop_w, ev_rep_cfg, is_training, mapping_type, sample_step
):
    def _get_dataset_by_in_out_video_name(input_video, normal_video, input_exposure_states):
        sample = SeeEverythingEveryTimePairedVideoDataset(
            group_folder,
            input_video,
            normal_video,
            video_to_frame_events[input_video],
            video_to_frame_events[normal_video],
            in_frames,
            crop_h,
            crop_w,
            ev_rep_cfg,
            is_training,
            input_exposure_states=input_exposure_states,
            sample_step=sample_step,
        )
        return sample

    with open(join(group_folder, "registrate_result.json"), "r") as f:
        registrate_result = json.load(f)
    video_to_frame_events = {}
    for key, value in registrate_result.items():
        start_timestamp = value["start_timestamp"]
        end_timestamp = value["end_timestamp"]
        frame_event_folder = join(group_folder, key, "frame_event")
        # f ends with png
        files = [f for f in listdir(frame_event_folder) if f.endswith(".png")]
        files = sorted(files)
        # timestamp is the first number of files
        frame_event_files = [[], []]
        for f in files:
            timestamp = float(f.split("_")[0])
            if start_timestamp <= timestamp <= end_timestamp:
                if "_vis" in f:  # events file
                    event_file_name = f.replace("_vis.png", ".npy")
                    frame_event_files[1].append(event_file_name)
                else:
                    frame_event_files[0].append(f)
        video_to_frame_events[key] = frame_event_files
    # make the dataset from lowlight or highlight to normal light
    with open(join(group_folder, "exposure_state.json"), "r") as f:
        exposure_state = json.load(f)
    normal_video_list = []
    lowlight_video_list = []
    highlight_video_list = []
    for video_name, values in exposure_state.items():
        video_exposure_state = values["exposure_state"]
        if video_exposure_state == "normal-light":
            normal_video_list.append(video_name)
        elif video_exposure_state == "low-light":
            lowlight_video_list.append(video_name)
        elif video_exposure_state == "high-light":
            highlight_video_list.append(video_name)

    if not is_training and len(normal_video_list) == 0:
        info(f"Testing: no normal-light video in {group_folder}")
        return []
    # Construct dataset for low to normal
    dataset = []
    # 1. low-light to normal-light
    if "low-normal" in mapping_type:
        for input_video in lowlight_video_list:
            for normal_video in normal_video_list:
                sample = _get_dataset_by_in_out_video_name(input_video, normal_video, "low-normal")
                dataset.append(sample)
    # 2. high-light to normal-light
    if "high-normal" in mapping_type:
        for input_video in highlight_video_list:
            for normal_video in normal_video_list:
                sample = _get_dataset_by_in_out_video_name(input_video, normal_video, "high-normal")
                dataset.append(sample)
    # 3. low-light to high-light
    if "low-high" in mapping_type:
        for input_video in lowlight_video_list:
            for output_video in highlight_video_list:
                sample = _get_dataset_by_in_out_video_name(input_video, output_video, "low-high")
                dataset.append(sample)
    # 4. high-light to low-light
    if "high-low" in mapping_type:
        for input_video in highlight_video_list:
            for output_video in lowlight_video_list:
                sample = _get_dataset_by_in_out_video_name(input_video, output_video, "high-low")
                dataset.append(sample)
    # Construct dataset for normal-light to normal-light
    if "normal-normal" in mapping_type:
        normal_video_count = len(normal_video_list)
        for i in range(normal_video_count):
            for j in range(normal_video_count):
                if i != j:
                    sample = _get_dataset_by_in_out_video_name(
                        normal_video_list[i], normal_video_list[j], "normal-normal"
                    )
                    dataset.append(sample)
    if "low-low" in mapping_type:
        low_video_count = len(lowlight_video_list)
        for i in range(low_video_count):
            for j in range(low_video_count):
                if i != j:
                    sample = _get_dataset_by_in_out_video_name(
                        lowlight_video_list[i], lowlight_video_list[j], "low-low"
                    )
                    dataset.append(sample)
    if "high-high" in mapping_type:
        high_video_count = len(highlight_video_list)
        for i in range(high_video_count):
            for j in range(high_video_count):
                if i != j:
                    sample = _get_dataset_by_in_out_video_name(
                        highlight_video_list[i], highlight_video_list[j], "high-high"
                    )
                    dataset.append(sample)
    return dataset


def get_see_everything_everytime_with_event_dataset_all(
    root, in_frames, crop_h, crop_w, ev_rep_cfg, testing_mapping_type, training_mapping_type, sample_step
):
    all_train_dataset, all_test_dataset = [], []
    video_all_folder = join(root, "VIDEOS-ALL")
    video_all_text = join(root, "videos_all.txt")

    for group in sorted(listdir(video_all_folder)):
        group_folder = join(video_all_folder, group)
        if isdir(group_folder):
            if group in TESTING_GROUPS:
                dataset_in_one_group = get_see_everything_everytime_with_event_dataset_for_each_group(
                    group_folder,
                    in_frames,
                    crop_h,
                    crop_w,
                    ev_rep_cfg,
                    is_training=False,
                    mapping_type=testing_mapping_type,
                    sample_step=sample_step,
                )
                if len(dataset_in_one_group) == 0:
                    info(f"Empty Group (Testing) : {group_folder}")
                    continue
                all_test_dataset.extend(dataset_in_one_group)
            else:
                dataset_in_one_group = get_see_everything_everytime_with_event_dataset_for_each_group(
                    group_folder,
                    in_frames,
                    crop_h,
                    crop_w,
                    ev_rep_cfg,
                    is_training=True,
                    mapping_type=training_mapping_type,
                    sample_step=sample_step,
                )
                if len(dataset_in_one_group) == 0:
                    debug(f"Empty Group (Training): {group_folder}")
                    continue
                all_train_dataset.extend(dataset_in_one_group)
    info(f"all_test_dataset: {len(all_test_dataset)}")
    info(f"all_train_dataset: {len(all_train_dataset)}")
    return ConcatDataset(all_train_dataset), ConcatDataset(all_test_dataset)


"""
CONFIG of the dataset
"""

TESTING_GROUPS = [
    "000-indoor_ceiling_table_light",
    "001-indoor_wall_displayboard_wood_luggage",
    "002-indoor_trophy_shelf_wall",
    "006-indoor_shot",
    "012-indoor",
    "018-indoor",
    "030-indoor",
    "042-indoor",
    "048-indoor",
    "054-indoor",
    "060-indoor",
    "065-indoor",
    "070-indoor",
    "074-indoor-ResolutionBoard",
    "075-indoor-ICLR",
    "100-outdoor",
    "106-outdoor",
    "112-outdoor",
    "118-outdoor",
    "124-outdoor",
    "130-outdoor",
    "136-outdoor",
    "142-outdoor",
    "148-outdoor",
    "154-outdoor",
    "160-outdoor",
    "166-outdoor",
    "173-outdoor",
    "184-outdoor",
    "189-outdoor",
    "194-outdoor",
    "200-outdoor",
    "206-outdoor",
    "212-outdoor",
    "217-outdoor",
    "222-outdoor",
    "225-outdoor",
]
