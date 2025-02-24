from os.path import join
import json
from absl import app
from absl.flags import FLAGS
from absl.logging import info
from os import listdir
from os.path import isdir
import os
import cv2
from tqdm import tqdm
import numpy as np


def make_video_in_a_group(group_folder, videos, event_frames):
    info(f"Group: {group_folder}")
    info(f"  Videos: {videos}")
    length_min = 1234567890
    for frames, events in event_frames:
        length_min = min(length_min, len(frames), len(events))
        info(f"  Frames: {len(frames)}, Events: {len(events)}")
    info(f"  Min length: {length_min}")
    video_count = len(videos)
    H, W = 290, 346

    output_path = os.path.join(group_folder, f"video-{length_min}-registrated.avi")
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


def make_registration_visualization_of_each_group(GROUP_ROOT):
    with open(join(GROUP_ROOT, "registrate_result.json"), "r") as f:
        registrate_result = json.load(f)
    event_frames = []
    videos = []
    for key, value in registrate_result.items():
        start_timestamp = value["start_timestamp"]
        end_timestamp = value["end_timestamp"]
        frame_event_folder = join(GROUP_ROOT, key, "frame_event")
        # f ends with png
        files = [f for f in listdir(frame_event_folder) if f.endswith(".png")]
        files = sorted(files)
        # timestamp is the first number of files
        frame_event_files = [[], []]
        for f in files:
            timestamp = float(f.split("_")[0])
            if start_timestamp <= timestamp <= end_timestamp:
                if "_vis" in f:
                    frame_event_files[1].append(f)
                else:
                    frame_event_files[0].append(f)
        videos.append(key)
        event_frames.append(frame_event_files)
    make_video_in_a_group(GROUP_ROOT, videos, event_frames)


if __name__ == "__main__":
    import pudb

    app.run(main)
