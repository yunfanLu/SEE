import glob
import os
import subprocess

import cv2


def create_video_from_images(image_folder, output_video):
    # 生成一个文件列表文件
    file_list_path = []
    for prefix in ["L2", "L1", "L0"]:
        i = 0
        image_path = glob.glob(os.path.join(image_folder, f"{prefix}-*.png"))
        image_path = sorted(image_path)
        file_list_path.extend(image_path)

    # 读取第一张图片，获取其尺寸
    frame = cv2.imread(file_list_path[0])
    height, width, layers = frame.shape

    # 创建视频写入对象
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))

    for image_file in file_list_path:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()


# 示例用法
image_folder = "testdata/RegistrateVisualization/"
output_video = "testdata/RegistrateVisualization-demo.mp4"
create_video_from_images(image_folder, output_video)
