import os
import pickle
from datetime import datetime, timedelta
from multiprocessing import Pool
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

HAS_ACC = False

app.flags.DEFINE_string("aedat_folder", None, "The folder of the aedat4 files.")
app.flags.DEFINE_string("video_folder", None, "The folder to save the extracted data.")


def vis_events(events, width, height):
    vis = np.zeros((height, width, 3), dtype=np.uint8) + 255
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    # set p == 0, x y to red
    # set p == 1, x y to blue
    vis[y[p == 0], x[p == 0]] = [255, 0, 0]
    vis[y[p == 1], x[p == 1]] = [0, 0, 255]
    return vis


def _save_frame(packet, frame_event_folder, tag):
    # timestamp, timestamp_start_of_frame, timestamp_end_of_frame, timestamp_start_of_exposure, timestamp_end_of_exposure
    # size, position
    timestamp = packet.timestamp
    timestamp_start_of_frame = packet.timestamp_start_of_frame
    timestamp_end_of_frame = packet.timestamp_end_of_frame
    timestamp_start_of_exposure = packet.timestamp_start_of_exposure
    timestamp_end_of_exposure = packet.timestamp_end_of_exposure
    size = packet.size
    position = packet.position
    rgb_frame = packet.image
    info(f"RGB frame shape              : {rgb_frame.shape}")
    info(f"  timestamp                  : {timestamp}")
    info(f"  timestamp_start_of_frame   : {timestamp_start_of_frame}")
    info(f"  timestamp_end_of_frame     : {timestamp_end_of_frame}")
    info(f"  timestamp_start_of_exposure: {timestamp_start_of_exposure}")
    info(f"  timestamp_end_of_exposure  : {timestamp_end_of_exposure}")
    info(f"  size                       : {size}")
    info(f"  position                   : {position}")
    if tag:
        frame_name = f"{timestamp}_{timestamp_start_of_frame}_{timestamp_end_of_frame}_{timestamp_start_of_exposure}_{timestamp_end_of_exposure}_{tag}.png"
    else:
        frame_name = f"{timestamp}_{timestamp_start_of_frame}_{timestamp_end_of_frame}_{timestamp_start_of_exposure}_{timestamp_end_of_exposure}.png"
    frame_path = join(frame_event_folder, frame_name)
    cv2.imwrite(frame_path, rgb_frame)


def load_aedat4(aedat_file_path, video_folder):
    frame_event_folder = join(video_folder, "frame_event")
    makedirs(frame_event_folder, exist_ok=True)
    with AedatFile(aedat_file_path) as f:
        # load rgb frame, events, imu, and rgb histogram.
        # ['events', 'frames', 'imu', 'triggers are the keys of the file
        frame_timestamps = []
        if HAS_ACC:
            # Access RGB frames
            for packet in f["frames"]:
                _save_frame(packet, frame_event_folder, "acc")
            for packet in f["frames_1"]:
                timestamp = packet.timestamp
                frame_timestamps.append(timestamp)
                _save_frame(packet, frame_event_folder, None)
        else:
            # Access RGB frames
            for packet in f["frames"]:
                timestamp = packet.timestamp
                frame_timestamps.append(timestamp)
                _save_frame(packet, frame_event_folder, None)

        # Access dimensions of the event stream
        height, width = f["events"].size
        events = np.hstack([packet for packet in f["events"].numpy()])
        # t: int64, x y: int16, p: int8
        t, x, y, p = (events["timestamp"], events["x"], events["y"], events["polarity"])

        for i in range(len(frame_timestamps) - 1):
            time_start = frame_timestamps[i]
            time_end = frame_timestamps[i + 1]
            # select t, x, y, p in [time_start, time_end]
            idx = (t >= time_start) & (t < time_end)
            t_, x_, y_, p_ = t[idx], x[idx], y[idx], p[idx]
            # save t_, x_, y_, p_ in numpy file with dict
            event_shot = np.stack([t_, x_, y_, p_], axis=1)
            # sorte by timestamp
            event_shot = event_shot[event_shot[:, 0].argsort()]
            # timestamp all has 16 digits
            event_name = f"{time_start}_{time_end}.npy"
            event_path = join(frame_event_folder, event_name)
            np.save(event_path, event_shot)
            # visualize the events
            vis = vis_events(event_shot, width, height)
            vis_path = join(frame_event_folder, f"{time_start}_{time_end}_vis.png")
            cv2.imwrite(vis_path, vis)

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
    print(f"IMU count: {imu_count}")


def extract(aedat_folder, video_folder):
    JOBS = 16
    pool = Pool(JOBS)
    # list all the aedat4 files
    aedat_files = sorted([f for f in listdir(aedat_folder) if f.endswith(".aedat4")])
    aedat_files = sorted(aedat_files)
    for aedat_file in aedat_files:
        aedat_file_path = join(aedat_folder, aedat_file)
        video_folder_tmp = join(video_folder, aedat_file.split(".")[0])
        makedirs(video_folder_tmp, exist_ok=True)
        print(f"Loading aedat4 file: {aedat_file_path}")
        load_aedat4(aedat_file_path, video_folder_tmp)
        # pool.apply_async(load_aedat4, args=(aedat_file_path, video_folder_tmp))


def main(args):
    # import pudb

    # pudb.set_trace()
    # load aedat4 file
    aedat_folder = FLAGS.aedat_folder
    video_folder = FLAGS.video_folder
    extract(aedat_folder=aedat_folder, video_folder=video_folder)


if __name__ == "__main__":
    app.run(main)
