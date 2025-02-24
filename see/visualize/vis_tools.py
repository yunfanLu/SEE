import os
from os.path import join

import cv2
import numpy as np
import torch
from absl.logging import debug, flags, info


def event_vis(event):
    event = event.sum(axis=0)
    H, W = event.shape
    event_image = np.zeros((H, W, 3), dtype=np.uint8) + 255
    event_image[event > 0] = [0, 0, 255]
    event_image[event < 0] = [255, 0, 0]
    return event_image
