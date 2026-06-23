#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from see.datasets.sde import get_seeing_dynamic_with_event_dataset_all
from see.datasets.see_dataset import get_see_everything_everytime_with_event_dataset_all


def get_dataset(config):
    if config.NAME == "seeing_dynamic_with_event":
        return get_seeing_dynamic_with_event_dataset_all(
            root=config.root,
            in_frames=config.in_frames,
            crop_h=config.crop_h,
            crop_w=config.crop_w,
            ev_rep_cfg=config.event_representation_config,
        )
    elif config.NAME == "see_everything_everytime_dataset":
        return get_see_everything_everytime_with_event_dataset_all(
            root=config.root,
            in_frames=config.in_frames,
            crop_h=config.crop_h,
            crop_w=config.crop_w,
            ev_rep_cfg=config.event_representation_config,
            testing_mapping_type=config.testing_mapping_type,
            training_mapping_type=config.training_mapping_type,
            sample_step=config.sample_step,
            testing_anchor_start=getattr(config, "testing_anchor_start", None),
            event_window_frames=getattr(config, "event_window_frames", None),
            all_groups_as_testing=getattr(config, "all_groups_as_testing", False),
            infer_exposure_state_from_brightness=getattr(config, "infer_exposure_state_from_brightness", False),
        )
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")
