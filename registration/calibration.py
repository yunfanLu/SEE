import numpy as np

"""
I0716 17:16:22.405374 140216232690880 cal_constant_bias.py:51] Video: IMU-Record-2024_07_08_20_45_00
I0716 17:16:22.967735 140216232690880 cal_constant_bias.py:52]   Accelerometer X: -0.009256236451445937, var: 5.835531070846904e-06, std: 0.002415684389742771
I0716 17:16:23.115411 140216232690880 cal_constant_bias.py:53]   Accelerometer Y: 0.9933439412483517, var: 6.196347306008374e-06, std: 0.002489246332930587
I0716 17:16:23.264492 140216232690880 cal_constant_bias.py:54]   Accelerometer Z: -0.04862174019695992, var: 1.3480731158043108e-05, std: 0.0036716115205782745
I0716 17:16:23.426353 140216232690880 cal_constant_bias.py:55]   Gyroscope X    : 1.0817807272159374, var: 0.010549644044962857, std: 0.10271146014424513
I0716 17:16:23.580387 140216232690880 cal_constant_bias.py:56]   Gyroscope Y    : -1.7912233357821072, var: 0.011101866467753914, std: 0.10536539502015789
I0716 17:16:23.730314 140216232690880 cal_constant_bias.py:57]   Gyroscope Z    : -0.6972366947574563, var: 0.01135962615495558, std: 0.10658154697205131
I0716 17:16:23.815169 140216232690880 cal_constant_bias.py:59]   Temperature    : 39.25108602008115, std: 1.0581819647863697
I0716 17:16:23.815385 140216232690880 cal_constant_bias.py:61]   Duration       : 4017.899833 s
I0716 17:16:24.086990 140216232690880 cal_constant_bias.py:51] Video: IMU-Record-2024_07_09_09_55_14
I0716 17:16:24.164375 140216232690880 cal_constant_bias.py:52]   Accelerometer X: -0.008570125918900922, var: 6.838885595344954e-06, std: 0.002615126305811051
I0716 17:16:24.202838 140216232690880 cal_constant_bias.py:53]   Accelerometer Y: 0.9921554891372245, var: 5.3432721591862495e-06, std: 0.0023115518941149147
I0716 17:16:24.244724 140216232690880 cal_constant_bias.py:54]   Accelerometer Z: -0.04536640425921547, var: 1.2980193771339697e-05, std: 0.0036028035987741127
I0716 17:16:24.284633 140216232690880 cal_constant_bias.py:55]   Gyroscope X    : 1.0462918451523149, var: 0.01145653587312574, std: 0.10703520856767525
I0716 17:16:24.325329 140216232690880 cal_constant_bias.py:56]   Gyroscope Y    : -1.792027798291795, var: 0.010695547981573208, std: 0.10341928244565038
I0716 17:16:24.364032 140216232690880 cal_constant_bias.py:57]   Gyroscope Z    : -0.7789281668335143, var: 0.012578705730285227, std: 0.11215482927758941
I0716 17:16:24.386549 140216232690880 cal_constant_bias.py:59]   Temperature    : 35.21148055461509, std: 2.3129380151491015
I0716 17:16:24.386746 140216232690880 cal_constant_bias.py:61]   Duration       : 932.508673 s
"""

var_acc_x = 5.835531070846904e-06
var_acc_y = 6.196347306008374e-06
var_acc_z = 1.3480731158043108e-05
var_gyr_x = 0.010549644044962857
var_gyr_y = 0.011101866467753914
var_gyr_z = 0.01135962615495558
# 构建测量噪声协方差矩阵 R
R = np.diag([var_acc_x, var_acc_y, var_acc_z, var_gyr_x, var_gyr_y, var_gyr_z])

# 初始设定的过程噪声方差
q_acc = 1e-3  # 假设的加速度计过程噪声
q_gyr = 1e-3  # 假设的陀螺仪过程噪声
# 构建过程噪声协方差矩阵 Q
Q = np.diag([q_acc, q_acc, q_acc, q_gyr, q_gyr, q_gyr])
