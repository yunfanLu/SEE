import os
from os.path import isfile

import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from absl.logging import info
from tqdm import tqdm

app.flags.DEFINE_string("aedat_folder", None, "The folder of the aedat4 files.")
app.flags.DEFINE_string("video_folder", None, "The folder to save the extracted data.")


def find_video_in_aedat4(video_name, aedat4_files):
    for aedat4 in aedat4_files:
        if video_name in aedat4:
            return aedat4
    return None


def mv_aedat4_to_video_folder(AEDAT_ROOT, VIDEO_ROOT):
    # list all aedat4 files in AEDAT_ROOT
    aedat4_files = [f for f in os.listdir(AEDAT_ROOT) if f.endswith(".aedat4")]
    groups = [f for f in os.listdir(VIDEO_ROOT) if os.path.isdir(os.path.join(VIDEO_ROOT, f))]
    for g in groups:
        info(f"Group: {g}")
        group_folder = os.path.join(VIDEO_ROOT, g)
        videos = [f for f in os.listdir(group_folder) if os.path.isdir(os.path.join(group_folder, f))]
        for v in videos:
            info(f"--VIDEO:{v}")
            video_folder = os.path.join(group_folder, v)
            aedat4_file = find_video_in_aedat4(v, aedat4_files)
            if aedat4_file is not None:
                info(f"--AEDAT:{aedat4_file}")
                aedat4_file_path = os.path.join(AEDAT_ROOT, aedat4_file)
                mv_command = f"mv {aedat4_file_path} {video_folder}"
                info(f"--{mv_command}")
                os.system(mv_command)
            else:
                info(f"--AEDAT: Not found")


def make_video_in_a_group(group_folder, videos, event_frames):
    info(f"Group: {group_folder}")
    info(f"  Videos: {videos}")
    length_min = 1000000
    for frames, events in event_frames:
        length_min = min(length_min, len(frames), len(events))
        info(f"  Frames: {len(frames)}, Events: {len(events)}")
    info(f"  Min length: {length_min}")
    video_count = len(videos)
    H, W = 290, 346

    output_path = os.path.join(group_folder, f"video-{length_min}.avi")
    if isfile(output_path):
        info(f"  Output exists: {output_path}")
        return

    info(f"  Output: {output_path}")
    # 指定编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 定义视频编码器
    out = cv2.VideoWriter(output_path, fourcc, 10, (W * 2, H * video_count))  # 定义视频写入对象

    # make a video with this frames and events
    for i in tqdm(range(length_min)):
        # make a frame
        frame_in_video = np.zeros((H * video_count, W * 2, 3), dtype=np.uint8)
        for j in range(video_count):
            frame_path = os.path.join(group_folder, videos[j], "frame_event", event_frames[j][0][i])
            event_path = os.path.join(group_folder, videos[j], "frame_event", event_frames[j][1][i])
            frame = cv2.imread(frame_path)
            event = cv2.imread(event_path)
            frame_with_text = np.zeros((H, W * 2, 3), dtype=np.uint8)
            frame_with_text[:260, :W, :] = frame
            frame_with_text[:260, W:, :] = event
            frame_with_text = cv2.putText(
                frame_with_text,
                f"{videos[j]}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            frame_in_video[j * H : (j + 1) * H, :, :] = frame_with_text
        out.write(frame_in_video)
    out.release()
    # exit()


def make_video_for_each_group(VIDEO_ROOT):
    groups = sorted([f for f in os.listdir(VIDEO_ROOT) if os.path.isdir(os.path.join(VIDEO_ROOT, f))])
    for g in groups:
        info(f"Group: {g}")
        group_folder = os.path.join(VIDEO_ROOT, g)
        videos = sorted([f for f in os.listdir(group_folder) if os.path.isdir(os.path.join(group_folder, f))])
        event_frames = []
        for v in videos:
            video_folder = os.path.join(group_folder, v)
            frame_event_folder = os.path.join(video_folder, "frame_event")
            frame_event_files = [[], []]
            png_files = [f for f in os.listdir(frame_event_folder) if f.endswith(".png")]
            png_files = sorted(png_files)
            for png_file in png_files:
                if "_vis.png" in png_file:
                    frame_event_files[1].append(png_file)
                else:
                    frame_event_files[0].append(png_file)
            info(f"-- {v}. frames:{len(frame_event_files[0])}, events:{len(frame_event_files[1])}")
            event_frames.append(frame_event_files)
        # make video
        make_video_in_a_group(group_folder, videos, event_frames)
        # exit()


def main(argv):
    # AEDAT_ROOT = "dataset/3-ComplexLight/RoboticArm/AEDAT4-0720/"
    # VIDEO_ROOT = "dataset/3-ComplexLight/RoboticArm/VIDOES-0720-R/"
    AEDAT_ROOT = FLAGS.aedat_folder
    VIDEO_ROOT = FLAGS.video_folder
    mv_aedat4_to_video_folder(AEDAT_ROOT, VIDEO_ROOT)
    make_video_for_each_group(VIDEO_ROOT)


if __name__ == "__main__":
    app.run(main)
