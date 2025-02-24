import os
import pickle
from datetime import datetime, timedelta
from os import listdir, makedirs
from os.path import isdir, join

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pudb
from absl import app, logging
from absl.logging import info
from dv import AedatFile
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

mpl.rcParams["agg.path.chunksize"] = 10000

FLAGS = app.flags.FLAGS

app.flags.DEFINE_string("video_root", "", "Path to the video folder")


def plt_imu_2d(
    video_folder_path,
    fname,
    timestamp,
    accelerometer_x,
    accelerometer_y,
    accelerometer_z,
    gyroscope_x,
    gyroscope_y,
    gyroscope_z,
):
    # 可视化数据
    fig, axs = plt.subplots(2, 1, figsize=(18, 9))
    # Accelerometer data
    axs[0].plot(timestamp, accelerometer_x, label="X")
    axs[0].plot(timestamp, accelerometer_y, label="Y")
    axs[0].plot(timestamp, accelerometer_z, label="Z")
    axs[0].set_title("Accelerometer Data")
    axs[0].set_xlabel("Timestamp")
    axs[0].set_ylabel("Acceleration (g)")
    axs[0].legend()

    # Gyroscope data
    axs[1].plot(timestamp, gyroscope_x, label="X")
    axs[1].plot(timestamp, gyroscope_y, label="Y")
    axs[1].plot(timestamp, gyroscope_z, label="Z")
    axs[1].set_title("Gyroscope Data")
    axs[1].set_xlabel("Timestamp")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].legend()

    plt.tight_layout()

    plt.savefig(join(video_folder_path, f"{fname}.png"))


def plt_imu_3d(
    video_folder_path,
    fname,
    timestamp,
    accelerometer_x,
    accelerometer_y,
    accelerometer_z,
    gyroscope_x,
    gyroscope_y,
    gyroscope_z,
):
    fig = plt.figure(figsize=(18, 9))
    # Accelerometer 3D plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(accelerometer_x, accelerometer_y, accelerometer_z)
    ax1.set_title("Accelerometer Data")
    ax1.set_xlabel("X (g)")
    ax1.set_ylabel("Y (g)")
    ax1.set_zlabel("Z (g)")

    # Gyroscope 3D plot
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(gyroscope_x, gyroscope_y, gyroscope_z)
    ax2.set_title("Gyroscope Data")
    ax2.set_xlabel("X (rad/s)")
    ax2.set_ylabel("Y (rad/s)")
    ax2.set_zlabel("Z (rad/s)")

    plt.tight_layout()
    plt.savefig(join(video_folder_path, f"{fname}.png"))


def plot_imu(root):
    videos = sorted(listdir(root))

    for video_folder in videos:
        video_folder_path = join(root, video_folder)
        if not isdir(video_folder_path):
            continue
        # load IMU data
        imu_data_path = join(video_folder_path, "imu.npy")
        imu = np.load(imu_data_path)

        # imu = imu[imu[:, 0].argsort()]
        print(imu.shape)

        timestamp = imu[:, 0]
        accelerometer_x = imu[:, 1]
        accelerometer_y = imu[:, 2]
        accelerometer_z = imu[:, 3]
        gyroscope_x = imu[:, 4]
        gyroscope_y = imu[:, 5]
        gyroscope_z = imu[:, 6]
        temperature = imu[:, 7]

        plt_imu_2d(
            video_folder_path,
            "imu_2d",
            timestamp,
            accelerometer_x,
            accelerometer_y,
            accelerometer_z,
            gyroscope_x,
            gyroscope_y,
            gyroscope_z,
        )

        plt_imu_3d(
            video_folder_path,
            "imu_3d",
            timestamp,
            accelerometer_x,
            accelerometer_y,
            accelerometer_z,
            gyroscope_x,
            gyroscope_y,
            gyroscope_z,
        )


def main(argv):
    # root = "testdata/IMURegistration/"
    root = FLAGS.video_root
    # videos = ["indoor-22-2024_06_20_16_35_04", "indoor-22-2024_06_20_16_36_08"]
    for group in tqdm(sorted(listdir(root))):
        video_root = join(root, group)
        if not isdir(video_root):
            continue
        plot_imu(video_root)

    # plot_imu(root)


if __name__ == "__main__":
    # pudb.set_trace()
    app.run(main)
