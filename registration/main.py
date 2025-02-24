from absl import app, flags, logging
from os.path import abspath, isdir
from absl.flags import FLAGS
from absl.logging import info
from os import listdir
from os.path import join
import numpy as np
import sys
from tqdm import tqdm
import json

from registration.IMU import IMU, plt_imu_2d, registrate, make_imu_file, plt_comp_imu
from registration.IMU import savitzky_golay_filter, kalman_filter, down_sample, plt_two_imu
from registration.calibration import Q, R
from registration.visualization import make_registration_visualization_of_each_group

FLAGS = flags.FLAGS

logging.set_verbosity(logging.INFO)

VISUALIZATION = True

flags.DEFINE_string("video_group_root", "", "Root of the video group")


def registrate_two_imus(S, T):
    # info("Denosing with Savitzky-Golay filter")
    # # acc_bais_svg, gyr_bais_svg, acc_match_ll_svg, gyr_match_ll_svg = registrate(T, S, denoise_type="savitzky_golay")
    # # info(f"  Registrate S to T: acc_bais={acc_bais_svg:>10}, gyr_bais={gyr_bais_svg:>10}, acc_match_ll={acc_match_ll_svg:>10}, gyr_match_ll={gyr_match_ll_svg:>10}")
    # acc_bais_svg, gyr_bais_svg, acc_match_ll_svg, gyr_match_ll_svg = registrate(S, T, denoise_type="savitzky_golay")
    # info(f"  acc_bais={acc_bais_svg:>10}, gyr_bais={gyr_bais_svg:>10}, acc_match_ll={acc_match_ll_svg:>10}, gyr_match_ll={gyr_match_ll_svg:>10}")
    info("Denosing with Kalman filter")
    # acc_bais_klm, gyr_bais_klm, acc_match_ll_klm, gyr_match_ll_klm = registrate(T, S, denoise_type="kalman")
    # info(f"  Registrate S to T: acc_bais={acc_bais_klm:>10}, gyr_bais={gyr_bais_klm:>10}, acc_match_ll={acc_match_ll_klm:>10}, gyr_match_ll={gyr_match_ll_klm:>10}")
    acc_bais_klm, gyr_bais_klm, acc_match_ll_klm, gyr_match_ll_klm = registrate(S, T, denoise_type="kalman")
    info(
        f"  acc_bais={acc_bais_klm:>10}, gyr_bais={gyr_bais_klm:>10}, acc_match_ll={acc_match_ll_klm:>10}, gyr_match_ll={gyr_match_ll_klm:>10}"
    )
    # info("Without denosing")
    # # acc_bais_wod, gyr_bais_wod, acc_match_ll_wod, gyr_match_ll_wod = registrate(T, S, denoise_type=None)
    # # info(f"  Registrate S to T: acc_bais={acc_bais_wod:>10}, gyr_bais={gyr_bais_wod:>10}, acc_match_ll={acc_match_ll_wod:>10}, gyr_match_ll={gyr_match_ll_wod:>10}")
    # acc_bais_wod, gyr_bais_wod, acc_match_ll_wod, gyr_match_ll_wod = registrate(S, T, denoise_type=None)
    # info(f"  acc_bais={acc_bais_wod:>10}, gyr_bais={gyr_bais_wod:>10}, acc_match_ll={acc_match_ll_wod:>10}, gyr_match_ll={gyr_match_ll_wod:>10}")
    return acc_bais_klm, gyr_bais_klm, acc_match_ll_klm, gyr_match_ll_klm


def registrate_one_group(GROUP_ROOT):
    GROUP_ROOT = abspath(GROUP_ROOT)
    folder_log_file = join(GROUP_ROOT, "registration.log")
    logging.get_absl_handler().use_absl_log_file(folder_log_file)
    # logging.get_absl_handler().python_handler.stream = sys.stdout

    registrate_result = {}

    # Registration of IMU data
    videos = sorted([f for f in listdir(GROUP_ROOT) if isdir(join(GROUP_ROOT, f))])
    source_root = join(GROUP_ROOT, videos[0], "imu.npy")
    source = make_imu_file(source_root)
    info(f"Source IMU {source_root} with {len(source)} samples")
    # Record each the "bias" of video_i to video_0
    bias_and_match = []
    for i in range(1, len(videos)):
        info(f"Registrate [{videos[i]}] to [{videos[0]}]:")
        target = make_imu_file(join(GROUP_ROOT, videos[i], "imu.npy"))
        acc_bais, gyr_bais, acc_match_ll, gyr_match_ll = registrate_two_imus(source, target)
        bias_and_match.append((acc_bais, gyr_bais, acc_match_ll, gyr_match_ll))

    # Merge the bias.
    S_start_merged = 0
    S_end_merged = len(source) - 1
    for i in range(1, len(videos)):
        info(f"Plot the IMU data: Video-{i} to Video-0")
        acc_bais, gyr_bais, acc_match_ll, gyr_match_ll = bias_and_match[i - 1]
        info(
            f"  acc_bais={acc_bais:>10}, gyr_bais={gyr_bais:>10}, acc_match_ll={acc_match_ll:>10}, gyr_match_ll={gyr_match_ll:>10}"
        )
        target = make_imu_file(join(GROUP_ROOT, videos[i], "imu.npy"))
        if acc_bais > 0:
            S_start = acc_bais
            S_end = acc_bais + acc_match_ll
        else:
            S_start = 0
            S_end = min(len(source) - 1, acc_match_ll)

        S_start_merged = max(S_start_merged, S_start)
        S_end_merged = min(S_end_merged, S_end)
        plt_path = join(GROUP_ROOT, f"V-{i}-to-{0}.png")
        if VISUALIZATION:
            plt_comp_imu(source, target, S_start, S_end, plt_path, level="L0", bias=acc_bais, context="")
    info(f"Merge the IMU data: Video-0 [{S_start_merged}, {S_end_merged}]")

    registrate_result[videos[0]] = {
        "start_index_of_imu": S_start_merged,
        "end_index_of_imu": S_end_merged,
        "start_timestamp": source.ts[S_start_merged],
        "end_timestamp": source.ts[S_end_merged],
        "time_duration": source.ts[S_end_merged] - source.ts[S_start_merged],
    }

    info(f"--------- Plot the merged IMU data ---------")
    # Plot the merged IMU data
    for i in range(1, len(videos)):
        info(f"Plot the IMU data: Video-{i} to Video-0")
        acc_bais, gyr_bais, acc_match_ll, gyr_match_ll = bias_and_match[i - 1]
        info(
            f"  acc_bais={acc_bais:>10}, gyr_bais={gyr_bais:>10}, acc_match_ll={acc_match_ll:>10}, gyr_match_ll={gyr_match_ll:>10}"
        )
        target = make_imu_file(join(GROUP_ROOT, videos[i], "imu.npy"))
        S = source.D(S_start_merged, S_end_merged)
        ll = S_end_merged - S_start_merged

        T_start = S_start_merged - acc_bais
        T_end = S_end_merged - acc_bais

        info(f"  S({len(source)}): {S_start_merged} -> {S_end_merged} ({ll})")
        info(f"  T({len(target)}): {T_start} -> {T_end} ({ll})")
        T = target.D(T_start, T_end)

        registrate_result[videos[i]] = {
            "start_index_of_imu": T_start,
            "end_index_of_imu": T_end,
            "start_timestamp": target.ts[T_start],
            "end_timestamp": target.ts[T_end - 1],
            "time_duration": target.ts[T_end - 1] - target.ts[T_start],
        }

        ax1, ay1, az1, gx1, gy1, gz1 = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], S[:, 5]
        ax2, ay2, az2, gx2, gy2, gz2 = T[:, 0], T[:, 1], T[:, 2], T[:, 3], T[:, 4], T[:, 5]
        info(f"  ax1.L:{len(ax1)}, ax2.L:{len(ax2)}")
        ts_1 = np.arange(len(ax1))
        ts_2 = np.arange(len(ax2))
        plt_path = join(GROUP_ROOT, f"V-{i}-to-{0}-merged.png")
        if VISUALIZATION:
            plt_two_imu(
                ax1,
                ay1,
                az1,
                gx1,
                gy1,
                gz1,
                ts_1,
                ax2,
                ay2,
                az2,
                gx2,
                gy2,
                gz2,
                ts_2,
                0,
                len(ax1),
                plt_path,
                level="L0",
                context="Merged",
            )
    return registrate_result


def main(args):
    # GROUP_ROOT = "dataset/3-ComplexLight/IMUCalibrationRegistration/Registration/IMU-Test-1/"
    # registrate_result = registrate_one_group(GROUP_ROOT)
    # with open(join(GROUP_ROOT, "registrate_result.json"), "w") as f:
    #     json.dump(registrate_result, f, indent=4)

    # VIDEO_GROUP_ROOT = "dataset/3-ComplexLight/RoboticArm/DEBUG/"
    VIDEO_GROUP_ROOT = FLAGS.video_group_root

    for group in tqdm(sorted(listdir(VIDEO_GROUP_ROOT))):
        GROUP_ROOT = join(VIDEO_GROUP_ROOT, group)
        if not isdir(GROUP_ROOT):
            continue
        registrate_result = registrate_one_group(GROUP_ROOT)
        with open(join(GROUP_ROOT, "registrate_result.json"), "w") as f:
            json.dump(registrate_result, f, indent=4)
        make_registration_visualization_of_each_group(GROUP_ROOT)


if __name__ == "__main__":
    import pudb

    # pudb.set_trace()
    app.run(main)
