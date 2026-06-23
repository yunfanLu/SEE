set -e

cd "/Users/yunfanlu/Documents/2-MyPublish/1.1-1-ECCV-Workshop/2-SEE-Challenges/SEE"
export PYTHONPATH="$PWD:$PYTHONPATH"

# 1. Extract the frames from the videos and the events from the DVS recordings
# python -m pip install -U dv-processing

# AEDAT_DIR="/Users/yunfanlu/Documents/2-MyPublish/1.1-1-ECCV-Workshop/2-SEE-Challenges/SEE/additional_data/DVS346-机械臂镜子"
# OUT_DIR="/Users/yunfanlu/Documents/2-MyPublish/1.1-1-ECCV-Workshop/2-SEE-Challenges/SEE/additional_data/DVS346-机械臂镜子"

# mkdir -p "$OUT_DIR"

# python tools/IMU-Registration-Tool/registration/extract_aedat4.py \
#   --aedat_folder="$AEDAT_DIR" \
#   --video_folder="$OUT_DIR"

# # 2. Plat the IMU data
# python tools/IMU-Registration-Tool/registration/plt_imu_2d_3d.py --video_root="additional_data/DVS346-Robotic-Arm-Mirror/"

# # 3. Make video without registration
# python tools/IMU-Registration-Tool/registration/make_group_in_a_video_for_visualization.py \
#     --video_folder="additional_data/DVS346-Robotic-Arm-Mirror/"

# # 4. Registration
# python ./registration/main.py \
#     --video_group_root="additional_data/DVS346-Robotic-Arm-Mirror/"

python registration/exposure_classifier.py --root="additional_data/DVS346-Robotic-Arm-Mirror/"