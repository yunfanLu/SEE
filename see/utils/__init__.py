#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch


def print_batch(batch, prefix="", ppfun=print):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            ppfun(f"{prefix} {key}:{value.shape}, {value.dtype}")
        elif isinstance(value, list):
            ppfun(f"{prefix} {key}:{len(value)}")
            for i, v in enumerate(value):
                print_batch({f"{key} {i}": v}, prefix="  ")
        else:
            ppfun(f"{prefix} {key}:{value}")


def event_voxel_grid_to_image(event_voxel_grid):
    """
    event_voxel_grid: C, H, W. type: np.ndarray
    """
    event_voxel_grid = event_voxel_grid.sum(axis=0)
    H, W = event_voxel_grid.shape
    image_visualization = np.zeros((H, W, 3), dtype=np.uint8) + 255
    image_visualization[event_voxel_grid > 0] = np.array([255, 0, 0])
    image_visualization[event_voxel_grid < 0] = np.array([0, 0, 255])
    return image_visualization
