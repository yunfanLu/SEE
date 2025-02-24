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
from tqdm import tqdm

logging.set_verbosity(logging.WARN)

FLAGS = app.flags.FLAGS

app.flags.DEFINE_string("aedat_folder", None, "The folder of the aedat4 files.")
app.flags.DEFINE_string("video_folder", None, "The folder to save the extracted data.")


def load_aedat4(aedat_file_path, video_folder):
    with AedatFile(aedat_file_path) as f:
        # Access IMU data
        (
            timestamps,
            accelerometer_x,
            accelerometer_y,
            accelerometer_z,
            gyroscope_x,
            gyroscope_y,
            gyroscope_z,
            temperatures,
        ) = ([], [], [], [], [], [], [], [])
        imu_count = 0
        imu = []
        for packet in f["imu"]:
            # timestamp, temperature, acceleration, gyro, magnetometer
            timestamp = packet.timestamp
            temperature = packet.temperature
            accelerometer = packet.accelerometer
            gyroscope = packet.gyroscope
            magnetometer = packet.magnetometer
            info(f"IMU data")
            info(f"  timestamp    : {timestamp}")
            info(f"  temperature  : {temperature}")
            info(f"  accelerometer: {accelerometer}")
            info(f"  gyroscope    : {gyroscope}")
            info(f"  magnetometer : {magnetometer}")
            imu_count += 1

            timestamps.append(timestamp)
            accelerometer_x.append(accelerometer[0])
            accelerometer_y.append(accelerometer[1])
            accelerometer_z.append(accelerometer[2])
            gyroscope_x.append(gyroscope[0])
            gyroscope_y.append(gyroscope[1])
            gyroscope_z.append(gyroscope[2])
            temperatures.append(temperature)

        # save all the imu data in a numpy file:
        imu_path = join(video_folder, "imu.npy")
        imu = np.stack(
            [
                timestamps,
                accelerometer_x,
                accelerometer_y,
                accelerometer_z,
                gyroscope_x,
                gyroscope_y,
                gyroscope_z,
                temperatures,
            ],
            axis=1,
        )
        np.save(imu_path, imu)


def extract(aedat_folder, video_folder):
    # list all the aedat4 files
    aedat_files = [f for f in listdir(aedat_folder) if f.endswith(".aedat4")]
    # aedat_files = sorted(aedat_files)
    for aedat_file in aedat_files:
        aedat_file_path = join(aedat_folder, aedat_file)
        video_folder_tmp = join(video_folder, aedat_file.split(".")[0])
        makedirs(video_folder_tmp, exist_ok=True)
        print(f"Loading aedat4 file: {aedat_file_path}")
        load_aedat4(aedat_file_path, video_folder_tmp)


def main(args):
    # import pudb

    # pudb.set_trace()
    # load aedat4 file
    aedat_folder = FLAGS.aedat_folder
    video_folder = FLAGS.video_folder
    extract(aedat_folder=aedat_folder, video_folder=video_folder)


if __name__ == "__main__":
    app.run(main)
