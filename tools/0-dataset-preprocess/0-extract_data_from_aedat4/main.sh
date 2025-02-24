#!/bin/bash
#SBATCH -p i64m512u   # 申请的分区计算资源
#SBATCH -J myjob      # 作业名称
#SBATCH --ntasks-per-node=64 # 每个计算节点上运行 task 的数量
#SBATCH  -n 128   # -n: 任务运行多少核;也可以通过-N 来指定申请的节点，如2个节点，就可以用-N 2来指定;
#SBATCH -o job.%j.out   # 作业运行log输出文件
#SBATCH -e job.%j.err   # 作业错误信息log输出文件

python tools/0-extract_data_from_aedat4/extract_aedat4.py \
    --aedat_folder="dataset/3-ComplexLight/柳工实际采集的数据/" \
    --video_folder="dataset/3-ComplexLight/DRIVE/VIDEOS/LIU_GONG/"

python tools/0-extract_data_from_aedat4/plt_imu_2d_3d.py \
    --video_root="dataset/3-ComplexLight/RoboticArm/VIDEOS-0722-R/"

python tools/0-extract_data_from_aedat4/plt_imu_2d_3d.py \
    --video_root="dataset/3-ComplexLight/IMUCalibrationRegistration/Registration/"

python tools/0-extract_data_from_aedat4/make_group_in_a_video_for_visualization.py \
    --aedat_folder="dataset/3-ComplexLight/RoboticArm/AEDTA4-0722/" \
    --video_folder="dataset/3-ComplexLight/IMUCalibrationRegistration/Registration/"

python tools/0-extract_data_from_aedat4/make_group_in_a_video_for_visualization.py \
    --aedat_folder="dataset/3-ComplexLight/柳工实际采集的数据/" \
    --video_folder="dataset/3-ComplexLight/TestingScenes/VIDEOS/"