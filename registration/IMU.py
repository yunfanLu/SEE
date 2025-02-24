from scipy.signal import savgol_filter
import numpy as np
import copy
from numba import jit
from copy import deepcopy
from os.path import join
import matplotlib.pyplot as plt
from absl.logging import info, debug, warn
from registration import calibration as CALB


VISUALIZATION = True

ZOOM_SCALE = 10


def savitzky_golay_filter(imu, win_len=101, polyorder=2):
    imu = deepcopy(imu)
    imu.acc_x = savgol_filter(imu.acc_x, win_len, polyorder)
    imu.acc_y = savgol_filter(imu.acc_y, win_len, polyorder)
    imu.acc_z = savgol_filter(imu.acc_z, win_len, polyorder)
    imu.gyr_x = savgol_filter(imu.gyr_x, win_len, polyorder)
    imu.gyr_y = savgol_filter(imu.gyr_y, win_len, polyorder)
    imu.gyr_z = savgol_filter(imu.gyr_z, win_len, polyorder)
    imu.tmp = savgol_filter(imu.tmp, win_len, polyorder)
    return imu


def kalman_filter(imu, Q=None, R=None):
    if Q is None:
        Q = CALB.Q
    if R is None:
        R = CALB.R

    imu = deepcopy(imu)
    # 初始化状态向量 x 和协方差矩阵 P
    x = np.zeros((6, len(imu)))
    P = np.zeros((6, 6, len(imu)))
    # 初始状态
    x[:, 0] = [imu.acc_x[0], imu.acc_y[0], imu.acc_z[0], imu.gyr_x[0], imu.gyr_y[0], imu.gyr_z[0]]
    P[:, :, 0] = np.eye(6)
    # 状态转移矩阵 F 和观测矩阵 H
    F = np.eye(6)
    H = np.eye(6)
    # 噪声协方差矩阵 Q 和测量噪声协方差矩阵 R
    Q = np.eye(6) * Q
    R = np.eye(6) * R

    for t in range(1, len(imu)):
        # 时间更新（预测）
        x[:, t] = F @ x[:, t - 1]
        P[:, :, t] = F @ P[:, :, t - 1] @ F.T + Q
        # 观测
        z = np.array([imu.acc_x[t], imu.acc_y[t], imu.acc_z[t], imu.gyr_x[t], imu.gyr_y[t], imu.gyr_z[t]])
        # 计算卡尔曼增益
        S = H @ P[:, :, t] @ H.T + R
        K = P[:, :, t] @ H.T @ np.linalg.inv(S)
        # 测量更新（校正）
        x[:, t] = x[:, t] + K @ (z - H @ x[:, t])
        P[:, :, t] = (np.eye(6) - K @ H) @ P[:, :, t]

    # 将滤波后的数据赋值回 imu 对象
    imu.acc_x = x[0, :]
    imu.acc_y = x[1, :]
    imu.acc_z = x[2, :]
    imu.gyr_x = x[3, :]
    imu.gyr_y = x[4, :]
    imu.gyr_z = x[5, :]
    return imu


def down_sample(imu, factor):
    # average down sample with factor
    imu = deepcopy(imu)
    length = len(imu) // factor * factor
    imu.ts = imu.ts[:length:factor]
    imu.acc_x = np.mean(imu.acc_x[:length].reshape(-1, factor), axis=1)
    imu.acc_y = np.mean(imu.acc_y[:length].reshape(-1, factor), axis=1)
    imu.acc_z = np.mean(imu.acc_z[:length].reshape(-1, factor), axis=1)
    imu.gyr_x = np.mean(imu.gyr_x[:length].reshape(-1, factor), axis=1)
    imu.gyr_y = np.mean(imu.gyr_y[:length].reshape(-1, factor), axis=1)
    imu.gyr_z = np.mean(imu.gyr_z[:length].reshape(-1, factor), axis=1)
    imu.tmp = np.mean(imu.tmp[:length].reshape(-1, factor), axis=1)
    return imu


def plt_imu_2d(imu, plt_path):
    timestamp = imu.ts
    accelerometer_x = imu.acc_x
    accelerometer_y = imu.acc_y
    accelerometer_z = imu.acc_z
    gyroscope_x = imu.gyr_x
    gyroscope_y = imu.gyr_y
    gyroscope_z = imu.gyr_z

    # 可视化数据
    fig, axs = plt.subplots(2, 1, figsize=(18, 9))
    # Accelerometer data
    axs[0].plot(timestamp, accelerometer_x, label="X")
    axs[0].plot(timestamp, accelerometer_y, label="Y")
    axs[0].plot(timestamp, accelerometer_z, label="Z")
    axs[0].set_title("Accelerometer Data")
    axs[0].set_xlabel("Timestamp")
    axs[0].set_ylabel("Acceleration (g)")
    axs[0].set_ylim(-1, 1)
    axs[0].legend()

    # Gyroscope data
    index = np.arange(0, len(timestamp))
    axs[1].plot(index, gyroscope_x, label="X")
    axs[1].plot(index, gyroscope_y, label="Y")
    axs[1].plot(index, gyroscope_z, label="Z")
    axs[1].set_title("Gyroscope Data")
    axs[1].set_xlabel("Timestamp")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    axs[1].set_ylim(-10, 10)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(plt_path)
    plt.close()


def plt_two_imu(
    ax1, ay1, az1, gx1, gy1, gz1, ts_1, ax2, ay2, az2, gx2, gy2, gz2, ts_2, S_start, S_end, plt_path, level, context=""
):
    fig, axs = plt.subplots(2, 1, figsize=(18, 9))
    # Accelerometer data, IMU 1 with blod line and alpha
    axs[0].plot(ts_1, ax1, label="X-1", color="red", linestyle="-", alpha=0.5, linewidth=5)
    axs[0].plot(ts_1, ay1, label="Y-1", color="green", linestyle="-", alpha=0.5, linewidth=5)
    axs[0].plot(ts_1, az1, label="Z-1", color="blue", linestyle="-", alpha=0.5, linewidth=5)
    axs[0].axvline(S_start, color="black", linestyle="--")
    axs[0].axvline(S_end, color="black", linestyle="--")

    axs[0].plot(ts_2, ax2, label="X-2", color="darkred", linestyle="-", linewidth=0.5)
    axs[0].plot(ts_2, ay2, label="Y-2", color="darkgreen", linestyle="-", linewidth=0.5)
    axs[0].plot(ts_2, az2, label="Z-2", color="darkblue", linestyle="-", linewidth=0.5)
    axs[0].set_title("Accelerometer Data")
    axs[0].set_xlabel("Timestamp")
    axs[0].set_ylabel("Acceleration (g)")
    # axs[0].set_ylim(-1, 1)
    axs[0].legend()

    # Gyroscope data
    axs[1].plot(ts_1, gx1, label="X-1", color="red", linestyle="-", alpha=0.3, linewidth=5)
    axs[1].plot(ts_1, gy1, label="Y-1", color="green", linestyle="-", alpha=0.3, linewidth=5)
    axs[1].plot(ts_1, gz1, label="Z-1", color="blue", linestyle="-", alpha=0.3, linewidth=5)
    axs[1].axvline(S_start, color="black", linestyle="--")
    axs[1].axvline(S_end, color="black", linestyle="--")

    axs[1].plot(ts_2, gx2, label="X-2", color="darkred", linestyle="-", linewidth=0.5)
    axs[1].plot(ts_2, gy2, label="Y-2", color="darkgreen", linestyle="-", linewidth=0.5)
    axs[1].plot(ts_2, gz2, label="Z-2", color="darkblue", linestyle="-", linewidth=0.5)
    axs[1].set_title("Gyroscope Data")
    axs[1].set_xlabel("Timestamp")
    axs[1].set_ylabel("Angular Velocity (rad/s)")
    # axs[1].set_ylim(-10, 10)
    axs[1].legend()

    plt.suptitle(f"{level} - {context}")
    plt.tight_layout()
    plt.savefig(plt_path)
    plt.close()


def plt_comp_imu(imu_1, imu_2, S_start, S_end, plt_path, level, bias=0, context=""):
    # ts_1 = imu_1.ts
    ts_1 = np.arange(imu_1.ts.shape[0])
    ax1, ay1, az1, gx1, gy1, gz1 = imu_1.acc_x, imu_1.acc_y, imu_1.acc_z, imu_1.gyr_x, imu_1.gyr_y, imu_1.gyr_z
    # ts_2 = imu_2.ts + bias
    ts_2 = np.arange(imu_2.ts.shape[0]) + bias
    ax2, ay2, az2, gx2, gy2, gz2 = imu_2.acc_x, imu_2.acc_y, imu_2.acc_z, imu_2.gyr_x, imu_2.gyr_y, imu_2.gyr_z
    #
    plt_two_imu(
        ax1, ay1, az1, gx1, gy1, gz1, ts_1, ax2, ay2, az2, gx2, gy2, gz2, ts_2, S_start, S_end, plt_path, level, context
    )


def distance(S, T):
    S_acc = S[:, 0:3]
    S_gyr = S[:, 3:6]
    T_acc = T[:, 0:3]
    T_gyr = T[:, 3:6]
    acc_dif = np.linalg.norm(S_acc - T_acc, axis=1)
    gyr_dif = np.linalg.norm(S_gyr - T_gyr, axis=1)
    acc_diff = np.mean(acc_dif)
    gyr_diff = np.mean(gyr_dif)
    return acc_diff, gyr_diff


def plt_dif_change(acc_dif_list, gyr_dif_list, bias, level, plt_path):
    fig, axs = plt.subplots(2, 1, figsize=(18, 9))
    axs[0].plot(bias, acc_dif_list, label="acc_diff")
    axs[0].set_title("acc_diff change with bias")
    axs[0].set_xlabel("bias")
    axs[0].set_ylabel("acc_diff")
    axs[0].legend()

    axs[1].plot(bias, gyr_dif_list, label="gyr_diff")
    axs[1].set_title("gyr_diff change with bias")
    axs[1].set_xlabel("bias")
    axs[1].set_ylabel("gyr_diff")
    axs[1].legend()

    plt.suptitle(f"{level} - dif change with bias")
    plt.tight_layout()
    plt.savefig(plt_path)
    plt.close()


def find_min_bais(S, T, Nbais, Pbais, level):
    info(f"Find min bais: S.length: {len(S)}, T.length: {len(T)}")
    acc_bais, gyr_bais = 0, 0
    acc_match_ll, gyr_match_ll = 0, 0

    LS, LT = len(S), len(T)
    lmg = min(LS, LT)  # length min
    Nbais = max(Nbais, -lmg + 10)
    Pbais = min(Pbais, lmg - 10)
    acc_match_ll, gyr_match_ll = lmg, lmg
    acc_diff_min, gyr_diff_min = distance(S.D(0, lmg), T.D(0, lmg))
    info(f"  Init  acc_diff_min: {acc_diff_min:.16f}, gyr_diff_min: {gyr_diff_min:.16f}")

    acc_dif_list, gyr_dif_list = [], []
    for idx, bias in enumerate(range(Nbais, Pbais)):
        if bias < 0:
            ll = min(LT + bias, LS)
            _S = S.D(0, ll)
            S_start = 0
            S_end = ll
            _T = T.D(-bias, -bias + ll)
        else:
            ll = min(LS - bias, LT)
            _S = S.D(bias, bias + ll)
            S_start = bias
            S_end = bias + ll
            _T = T.D(0, ll)
        # calculate distance
        acc_dif, gyr_dif = distance(_S, _T)
        acc_dif_list.append(acc_dif)
        gyr_dif_list.append(gyr_dif)

        if acc_dif < acc_diff_min:
            acc_diff_min = acc_dif
            acc_bais = bias
            acc_match_ll = ll
            debug(f"  Update bias: {bias}, acc_bais: {acc_bais}, acc_diff_min: {acc_diff_min:.18f}")
        if gyr_dif < gyr_diff_min:
            gyr_diff_min = gyr_dif
            gyr_bais = bias
            gyr_match_ll = ll
            debug(f"  Update bias: {bias}, gyr_bais: {gyr_bais}, gyr_diff_min: {gyr_diff_min:.18f}")

        if VISUALIZATION:
            plt_comp_imu(
                S,
                T,
                S_start,
                S_end,
                join("testdata/RegistrateVisualization/", f"{level}-{str(idx).zfill(4)}.png"),
                level,
                bias,
                f"bias: {bias}, acc_diff: {acc_dif:.6f}, gyr_diff: {gyr_dif:.6f}",
            )
    if VISUALIZATION:
        # plt the dif change with bias
        plt_dif_change(
            acc_dif_list,
            gyr_dif_list,
            range(Nbais, Pbais),
            level,
            join("testdata/RegistrateVisualization/", f"Dif-change-{level}.png"),
        )

    if acc_bais != gyr_bais:
        debug(f"Warning: acc_bais: {acc_bais}, gyr_bais: {gyr_bais}")
    debug(f"  Final acc_diff_min: {acc_diff_min:.16f}, gyr_diff_min: {gyr_diff_min:.16f}")
    debug(f"  Final acc_bais    : {acc_bais},          gyr_bais    : {gyr_bais}")
    debug(f"  Final acc_match_ll: {acc_match_ll},      gyr_match_ll: {gyr_match_ll}")
    return acc_bais, gyr_bais, acc_match_ll, gyr_match_ll


def registrate(source, target, denoise_type):
    if denoise_type == "savitzky_golay":
        S = savitzky_golay_filter(source, win_len=31, polyorder=3)
        T = savitzky_golay_filter(target, win_len=31, polyorder=3)
    elif denoise_type == "kalman":
        S = kalman_filter(source)
        T = kalman_filter(target)
    else:
        S = source
        T = target

    S_d1 = down_sample(S, ZOOM_SCALE)
    T_d1 = down_sample(T, ZOOM_SCALE)

    S_d2 = down_sample(S_d1, ZOOM_SCALE)
    T_d2 = down_sample(T_d1, ZOOM_SCALE)

    debug(f"Source: {len(S)} -> {len(S_d1)} -> {len(S_d2)}")
    debug(f"Target: {len(T)} -> {len(T_d1)} -> {len(T_d2)}")

    d2_acc_bais, d2_gyr_bais, d2_acc_match_ll, d2_gyr_match_ll = find_min_bais(S_d2, T_d2, -100, 100, level="L2")

    Nbias = min(d2_acc_bais, d2_gyr_bais) * ZOOM_SCALE - ZOOM_SCALE * 10
    Pbias = max(d2_acc_bais, d2_gyr_bais) * ZOOM_SCALE + ZOOM_SCALE * 10
    d1_acc_bais, d1_gyr_bais, d1_acc_match_ll, d1_gyr_match_ll = find_min_bais(S_d1, T_d1, Nbias, Pbias, level="L1")

    Nbias = min(d1_acc_bais, d1_gyr_bais) * ZOOM_SCALE - ZOOM_SCALE * 10
    Pbias = max(d1_acc_bais, d1_gyr_bais) * ZOOM_SCALE + ZOOM_SCALE * 10
    acc_bais, gyr_bais, acc_match_ll, gyr_match_ll = find_min_bais(S, T, Nbias, Pbias, level="L0")

    debug(
        f"  d2_acc_bais: {d2_acc_bais:>10}, d2_gyr_bais: {d2_gyr_bais:>10}, d2_acc_match_ll: "
        f"{d2_acc_match_ll:>10}, d2_gyr_match_ll: {d2_gyr_match_ll:>10}"
    )
    debug(
        f"  d1_acc_bais: {d1_acc_bais:>10}, d1_gyr_bais: {d1_gyr_bais:>10}, d1_acc_match_ll: "
        f"{d1_acc_match_ll:>10}, d1_gyr_match_ll: {d1_gyr_match_ll:>10}"
    )
    debug(
        f"  acc_bais   : {acc_bais:>10}, gyr_bais   : {gyr_bais:>10}, acc_match_ll   : "
        f"{acc_match_ll:>10}, gyr_match_ll   : {gyr_match_ll:>10}"
    )

    return acc_bais, gyr_bais, acc_match_ll, gyr_match_ll


def make_imu_file(file):
    imu = np.load(file)
    return IMU(imu)


class IMU:
    def __init__(self, imu=None):
        self.ts = imu[:, 0]
        self.index = np.arange(imu.shape[0])
        self.acc_x = imu[:, 1]
        self.acc_y = imu[:, 2]
        self.acc_z = imu[:, 3]
        self.gyr_x = imu[:, 4]
        self.gyr_y = imu[:, 5]
        self.gyr_z = imu[:, 6]
        self.tmp = imu[:, 7]

    def D(self, s, e):
        # return acc and gry
        return np.stack(
            [self.acc_x[s:e], self.acc_y[s:e], self.acc_z[s:e], self.gyr_x[s:e], self.gyr_y[s:e], self.gyr_z[s:e]],
            axis=1,
        )

    def __len__(self):
        return self.ts.shape[0]
