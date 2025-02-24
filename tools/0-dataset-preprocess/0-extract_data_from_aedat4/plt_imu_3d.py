import os
import pickle
from datetime import datetime, timedelta
from os import listdir, makedirs
from os.path import isdir, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, logging
from absl.logging import info
from dv import AedatFile
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class IMU:
    def __init__(video_folder_path):
        # load IMU data
        self.name = video_folder_path.split("/")[-1].split("_")[-1]
        imu_data_path = join(video_folder_path, "imu.npy")
        imu = np.load(imu_data_path)
        self.timestamp = imu[:, 0]
        self.accelerometer_x = imu[:, 1]
        self.accelerometer_y = imu[:, 2]
        self.accelerometer_z = imu[:, 3]
        self.gyroscope_x = imu[:, 4]
        self.gyroscope_y = imu[:, 5]
        self.gyroscope_z = imu[:, 6]
        self.temperature = imu[:, 7]

    @property
    def ts(self):
        return self.timestamp

    @property
    def acc(self):
        return self.accelerometer_x, self.accelerometer_y, self.accelerometer_z

    @property
    def gyro(self):
        return self.gyroscope_x, self.gyroscope_y, self.gyroscope_z

    @property
    def temp(self):
        return self.temperature


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def calculate_trajectory(imu_obj, cutoff=5.0, fs=100.0, low_pass=False):
    timestamp = imu_obj.ts
    accelerometer_x, accelerometer_y, accelerometer_z = imu_obj.acc
    gyroscope_x, gyroscope_y, gyroscope_z = imu_obj.gyro

    if low_pass:
        accelerometer_x = lowpass_filter(accelerometer_x, cutoff, fs)
        accelerometer_y = lowpass_filter(accelerometer_y, cutoff, fs)
        accelerometer_z = lowpass_filter(accelerometer_z, cutoff, fs)
        gyroscope_x = lowpass_filter(gyroscope_x, cutoff, fs)
        gyroscope_y = lowpass_filter(gyroscope_y, cutoff, fs)
        gyroscope_z = lowpass_filter(gyroscope_z, cutoff, fs)

    initial_orientation = R.from_euler("xyz", [0, 0, 0], degrees=True)
    angular_velocity = np.vstack((gyroscope_x, gyroscope_y, gyroscope_z)).T
    delta_t = np.diff(timestamp)
    orientation = [initial_orientation]
    for i in range(1, len(timestamp)):
        delta_angle = angular_velocity[i] * delta_t[i - 1]
        orientation.append(orientation[-1] * R.from_rotvec(delta_angle))
    orientation = np.array([r.as_matrix() for r in orientation])
    gravity = np.array([0, 0, 9.81])
    accelerometer = np.vstack((accelerometer_x, accelerometer_y, accelerometer_z)).T - gravity
    position = [np.zeros(3)]
    velocity = [np.zeros(3)]
    for i in range(1, len(timestamp)):
        dt = delta_t[i - 1]
        acc_world = orientation[i - 1].dot(accelerometer[i])
        velocity.append(velocity[-1] + acc_world * dt)
        position.append(position[-1] + velocity[-1] * dt + 0.5 * acc_world * dt**2)
    position = np.array(position)
    return position


def plot_trajectory(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.show()


def main(argv):
    import pudb

    pudb.set_trace()

    root = "dataset/3-ComplexLight/VIDEOS/"
    videos = sorted(listdir(root))
    videos = ["indoor-10-2024_06_18_20_51_50", "indoor-10-2024_06_18_20_52_38", "indoor-10-2024_06_18_20_53_28"]
    imu1 = IMU(join(root, videos[0]))
    imu2 = IMU(join(root, videos[1]))
    imu3 = IMU(join(root, videos[2]))
    positions1 = calculate_trajectory(imu1, low_pass=True)
    positions2 = calculate_trajectory(imu2, low_pass=True)
    positions3 = calculate_trajectory(imu3, low_pass=True)
    plot_trajectory(positions1)
    plot_trajectory(positions2)
    plot_trajectory(positions3)


if __name__ == "__main__":
    app.run(main)
