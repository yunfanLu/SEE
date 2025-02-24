import os
import pickle
from datetime import datetime, timedelta
from os import listdir, makedirs
from os.path import isdir, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pudb
import seaborn as sns
from absl import app, logging
from absl.logging import info
from dv import AedatFile
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

FLAGS = app.flags.FLAGS

app.flags.DEFINE_string("video_root", "", "Path to the video folder")


def main(argv):
    # root = "testdata/IMURegistration/"
    root = FLAGS.video_root
    # videos = ["indoor-22-2024_06_20_16_35_04", "indoor-22-2024_06_20_16_36_08"]
    videos = sorted(listdir(root))

    for video_folder in videos:
        video_folder_path = join(root, video_folder)
        if not isdir(video_folder_path):
            continue

        frame_event_folder = join(video_folder_path, "frame_event")
        # load all events data, which are end with .npy
        events = [f for f in listdir(frame_event_folder) if f.endswith(".npy")]
        events = sorted(events)
        events_npy = [np.load(join(frame_event_folder, f)) for f in events]
        events_npy = np.concatenate(events_npy, axis=0)
        t = events_npy[:, 0]

        plt.figure(figsize=(18, 5))
        sns.kdeplot(t, shade=True, bw_adjust=0.5)
        plt.title("Density of Event Timestamps")
        plt.xlabel("Timestamp")
        plt.ylabel("Density")

        plt.savefig(join(video_folder_path, "events_timestamp_histogram.png"))


if __name__ == "__main__":
    # pudb.set_trace()
    app.run(main)
