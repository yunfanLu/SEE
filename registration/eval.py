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
from absl import flags
import matplotlib.pyplot as plt
import seaborn as sns

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "root", "testdata/IMURegistration", "root folder of registration result"
)


def cal_pixel_distance(base_frame, frame):
    gray_base = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_base, descriptors_base = sift.detectAndCompute(gray_base, None)
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors_base, descriptors_frame, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    src_pts = np.float32([keypoints_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Transform matrix H:")
    print(H)
    # Randomly select 100 point in the base image
    point_base = np.random.randint(0, 256, (100, 2)).astype(np.float32).reshape(-1, 1, 2)
    point_transformed = cv2.perspectiveTransform(point_base, H)
    print(f"Point in Source: {point_base[0][0]}")
    print(f"Point in Target: {point_transformed[0][0]}")
    distance = np.linalg.norm(point_base[0][0] - point_transformed[0][0])
    print(f"L1 Distance: {distance}")
    return distance


def make_video_in_a_group_visualziation_evaluation(group_folder, videos, event_frames):
    info(f"Group: {group_folder}")
    info(f"  Videos: {videos}")
    length_min = 1234567890
    for frames, events in event_frames:
        length_min = min(length_min, len(frames), len(events))
        info(f"  Frames: {len(frames)}, Events: {len(events)}")
    info(f"  Min length: {length_min}")
    video_count = len(videos)
    H, W = 290, 346

    output_path = join(group_folder, f"video-{length_min}-registrated-eval-pixel-distance.avi")
    info(f"  Output: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 10, (W * 3, H * video_count))

    PSNR_average = 0
    L1_average = 0
    pixel_distance_average = 0
    pixel_distance_list = []
    count = 0
    # make a video with this frames and events
    for i in tqdm(range(length_min)):
        # make a frame
        frame_in_video = np.zeros((H * video_count, W * 3, 3), dtype=np.uint8)
        for j in range(video_count):
            frame_path = join(group_folder, videos[j], "frame_event", event_frames[j][0][i])
            event_path = join(group_folder, videos[j], "frame_event", event_frames[j][1][i])
            frame = cv2.imread(frame_path)
            event = cv2.imread(event_path)
            frame_with_text = np.zeros((H, W * 3, 3), dtype=np.uint8) + 255
            frame_with_text[:260, :W, :] = frame
            frame_with_text[:260, W : W * 2, :] = event
            frame_with_text = cv2.putText(
                frame_with_text,
                f"Time: {videos[j][18:].replace('_',':')}:{frame_path.split('/')[-1].split('_')[0]}",
                (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            # add evaluation
            if j > 0:
                base_frame = cv2.imread(join(group_folder, videos[0], "frame_event", event_frames[0][0][i]))
                diff = cv2.absdiff(base_frame, frame)
                # using gamma to enhance the difference visualization
                diff_vis = np.power(diff / 255.0, 1.0 / 1.2) * 255.0
                diff_vis = np.clip(diff, 0, 255).astype(np.uint8)
                frame_with_text[:260, W * 2 : W * 3, :] = diff_vis
                pixel_distance = cal_pixel_distance(base_frame, frame)
                PSNR = cv2.PSNR(base_frame, frame)
                L1 = np.mean(np.abs(diff))
                pixel_distance_average += pixel_distance
                pixel_distance_list.append(pixel_distance)
                PSNR_average += PSNR
                L1_average += L1
                count += 1
                frame_with_text = cv2.putText(
                    frame_with_text,
                    f"Distance:{pixel_distance:.2f} pixel. PSNR:{PSNR:.2f}, L1:{L1:.2f}.",
                    (W * 2 - 340, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            frame_in_video[j * H : (j + 1) * H, :, :] = frame_with_text
        out.write(frame_in_video)
        # cv2.imwrite("test.png", frame_in_video)
        # exit()
    out.release()
    PSNR_average /= count
    L1_average /= count
    pixel_distance_average /= count
    info(f"  Pixel Distance: {pixel_distance_average}, PSNR average: {PSNR_average:.6f}, L1 average: {L1_average:.6f}")
    return pixel_distance_list

def make_registration_visualization_and_evaluation_of_each_group(GROUP_ROOT):
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
    pixel_distance_list = make_video_in_a_group_visualziation_evaluation(GROUP_ROOT, videos, event_frames)
    visualize_pixel_distance(pixel_distance_list, GROUP_ROOT)

def visualize_pixel_distance(pixel_distance_list, GROUP_ROOT):
    # 自定义调色板，使用您提供的颜色
    sns.set_theme(style="whitegrid")  # 可选主题："darkgrid", "whitegrid", "dark", "white", "ticks"
    plt.figure(figsize=(3, 3))  # 设置图表尺寸
    sns.histplot(pixel_distance_list, bins=500, kde=True, color='b', alpha=0.5)
    plt.title("Pixel Distance Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Pixel Distance", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # 自动调整子图参数以适应图表
    plt.savefig(join(GROUP_ROOT, "pixel_distance.png"), dpi=900)
    # plt.show()
    plt.close()

    plt.figure(figsize=(4, 4))  # 设置图表尺寸
    time = [i / 24.0 for i in range(len(pixel_distance_list))]
    sns.lineplot(x=time, y=pixel_distance_list, alpha=0.5)
    plt.axhline(y=np.mean(pixel_distance_list), color='b', linestyle='--', linewidth=1)
    plt.text(5, np.mean(pixel_distance_list), f"Mean: {np.mean(pixel_distance_list):.4f}", color='black', fontsize=18)
    plt.title("Pixel Distance Change", fontsize=16, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Pixel Distance", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # 自动调整子图参数以适应图表
    plt.savefig(join(GROUP_ROOT, "pixel_distance_change.png"), dpi=900)
    # plt.show()
    plt.close()


def main(args):
    for group in sorted(listdir(FLAGS.root)):
        group_folder = join(FLAGS.root, group)
        if isdir(group_folder):
            make_registration_visualization_and_evaluation_of_each_group(group_folder)


if __name__ == "__main__":
    import pudb
    pudb.set_trace()
    app.run(main)
