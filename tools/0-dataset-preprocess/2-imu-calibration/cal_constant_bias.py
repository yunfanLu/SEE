from os import listdir
from os.path import isdir, join

import numpy as np
from absl import app, flags
from absl.logging import info


def main(args):
    CALIBRATION_ROOT = "dataset/3-ComplexLight/IMUCalibrationRegistration/Calibration/"

    videos = sorted(listdir(CALIBRATION_ROOT))

    for video_folder in videos:
        video_folder_path = join(CALIBRATION_ROOT, video_folder)
        if not isdir(video_folder_path):
            continue
        # load IMU data
        imu_data_path = join(video_folder_path, "imu.npy")
        imu = np.load(imu_data_path)

        timestamp = imu[:, 0]
        accelerometer_x = imu[:, 1]
        accelerometer_y = imu[:, 2]
        accelerometer_z = imu[:, 3]
        gyroscope_x = imu[:, 4]
        gyroscope_y = imu[:, 5]
        gyroscope_z = imu[:, 6]
        temperature = imu[:, 7]

        info(f"Video: {video_folder}")
        info(f"  Accelerometer X: {accelerometer_x.mean()}, var: {accelerometer_x.var()}, std: {accelerometer_x.std()}")
        info(f"  Accelerometer Y: {accelerometer_y.mean()}, var: {accelerometer_y.var()}, std: {accelerometer_y.std()}")
        info(f"  Accelerometer Z: {accelerometer_z.mean()}, var: {accelerometer_z.var()}, std: {accelerometer_z.std()}")
        info(f"  Gyroscope X    : {gyroscope_x.mean()}, var: {gyroscope_x.var()}, std: {gyroscope_x.std()}")
        info(f"  Gyroscope Y    : {gyroscope_y.mean()}, var: {gyroscope_y.var()}, std: {gyroscope_y.std()}")
        info(f"  Gyroscope Z    : {gyroscope_z.mean()}, var: {gyroscope_z.var()}, std: {gyroscope_z.std()}")

        info(f"  Temperature    : {temperature.mean()}, std: {temperature.std()}")
        time_duration = (timestamp[-1] - timestamp[0]) / 1e6
        info(f"  Duration       : {time_duration} s")


if __name__ == "__main__":
    app.run(main)
