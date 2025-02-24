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

from matplotlib import pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string("root", "dataset/3-ComplexLight/RoboticArm/VIDEOS-ALL/", "root folder of registration result")


def analyze_exposure(histogram, total_pixels):
    L, O = int(255 * 0.1), int(255 * 0.9)
    dark_pixels = np.sum(histogram[:L])
    mid_pixels = np.sum(histogram[L:O])
    bright_pixels = np.sum(histogram[O:])
    if dark_pixels > 0.5 * total_pixels:
        exposure = -1
    elif bright_pixels > 0.5 * total_pixels:
        exposure = 1
    else:
        exposure = 0
    return exposure


def exposure_classifier_for_each_group(GROUP_ROOT):
    with open(join(GROUP_ROOT, "registrate_result.json"), "r") as f:
        registrate_result = json.load(f)
    exposure_state = {}
    for video_name, value in registrate_result.items():
        start_timestamp = value["start_timestamp"]
        end_timestamp = value["end_timestamp"]
        frame_event_folder = join(GROUP_ROOT, video_name, "frame_event")
        files = sorted([f for f in listdir(frame_event_folder) if f.endswith(".png")])
        # timestamp is the first number of files
        exposure_state[video_name] = {"mean_value": [], "exposure_state": [], "timestamp": []}
        for f in files:
            if "_vis" in f:
                continue
            timestamp = float(f.split("_")[0])
            if start_timestamp <= timestamp <= end_timestamp:
                frame_path = join(frame_event_folder, f)
                frame = cv2.imread(frame_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
                total_pixels = gray.size
                exposure = analyze_exposure(histogram, total_pixels)
                exposure_state[video_name]["mean_value"].append(gray.mean())
                exposure_state[video_name]["exposure_state"].append(exposure)
                exposure_state[video_name]["timestamp"].append(timestamp)
    return exposure_state


def average_exposure_state_value_to_string(average_exposure_state):
    if average_exposure_state < -0.333333:
        return "low-light"
    elif average_exposure_state > 0.333333:
        return "high-light"
    else:
        return "normal-light"


def store_exposure_state(GROUP_ROOT, exposure_state):
    # plot the all gray mean value in one figure with size 8 16
    plt.figure(figsize=(16, 8))
    for video_name, value in exposure_state.items():
        plt.plot(value["mean_value"], label=video_name)
    plt.legend()
    plt.title("Gray mean value of each frame")
    plt.savefig(join(GROUP_ROOT, "gray_mean_value.png"))
    # count the exposure state
    video_to_exposure_state = {}
    for video_name, value in exposure_state.items():
        states = value["exposure_state"]
        average_exposure_state = np.mean(states)
        video_to_exposure_state[video_name] = {
            "exposure_state": average_exposure_state_value_to_string(average_exposure_state),
            "average_exposure_state": average_exposure_state,
            "all_count": len(states),
            "dark": states.count(-1),
            "normal": states.count(0),
            "bright": states.count(1),
        }
    with open(join(GROUP_ROOT, "exposure_state.json"), "w") as f:
        json.dump(video_to_exposure_state, f, indent=4)


def main(args):
    for group in tqdm(sorted(listdir(FLAGS.root))):
        group_folder = join(FLAGS.root, group)
        if isdir(group_folder):
            info(f"Analyzing group: {group}")
            exposure_state = exposure_classifier_for_each_group(group_folder)
            store_exposure_state(group_folder, exposure_state)


if __name__ == "__main__":
    app.run(main)
