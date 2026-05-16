import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
from absl.logging import info
import cv2
import numpy as np


"""
Root
    vis
        group_name-[normal-normal OR low-normal OR high-normal]-input_videoname-normal_videoname
            *_l0_*.png # Low Quality Image (Input)
            *_p0_*.png # Prediction (Output)
GT Root
    group_name-[normal-normal OR low-normal OR high-normal]-input_videoname-normal_videoname
        *_n0_*.png     # Ground Truth
"""

FLAGS = flags.FLAGS

flags.DEFINE_string("root", None, "Evaluation root directory containing a vis subfolder")
flags.DEFINE_string("gt_root", None, "Ground-truth root directory")


class Eval:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.psnr = 0
        self.ssim = 0
        self.npsnr = 0
        self.l1 = 0

    def add(self, psnr, ssim, npsnr, l1):
        self.count += 1
        self.psnr += psnr
        self.ssim += ssim
        self.npsnr += npsnr
        self.l1 += l1

    def get(self):
        if self.count == 0:
            return 0, 0, 0, 0
        return (
            self.psnr / self.count,
            self.ssim / self.count,
            self.npsnr / self.count,
            self.l1 / self.count,
        )


def _load_image(image_path):
    import torch

    frame = cv2.imread(str(image_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    frame = torch.from_numpy(frame)
    return frame


def _iter_video_folders(vis_root: Path):
    for video_folder_name in sorted(listdir(vis_root)):
        video_folder = vis_root / video_folder_name
        if video_folder.is_dir():
            yield video_folder_name, video_folder


def _sample_key_from_name(img_name):
    parts = img_name.split("_")
    if len(parts) < 6:
        return None
    return "_".join(parts[:5])


def _build_gt_index(gt_video_folder: Path):
    gt_index = {}
    for img_name in sorted(listdir(gt_video_folder)):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        parts = img_name.split("_")
        if len(parts) < 6 or parts[5][:2] != "n0":
            continue
        key = _sample_key_from_name(img_name)
        if key is not None:
            gt_index[key] = gt_video_folder / img_name
    return gt_index


def summarize_results(all_eval, low_nml_eval, hgh_nml_eval, nml_nml_eval):
    def pack(eval_obj):
        return {"count": eval_obj.count, "metrics": eval_obj.get()}

    return {
        "All": pack(all_eval),
        "Low-Normal": pack(low_nml_eval),
        "High-Normal": pack(hgh_nml_eval),
        "Normal-Normal": pack(nml_nml_eval),
    }


def collect_eval_pairs(vis_root, gt_root):
    vis_root = Path(vis_root)
    gt_root = Path(gt_root)
    pairs = []

    for video_folder_name, video_folder in _iter_video_folders(vis_root):
        gt_video_folder = gt_root / video_folder_name
        if not gt_video_folder.is_dir():
            info(f"Skip {video_folder_name}: GT folder not found at {gt_video_folder}")
            continue

        gt_index = _build_gt_index(gt_video_folder)
        for img_name in sorted(listdir(video_folder)):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            parts = img_name.split("_")
            if len(parts) < 6:
                continue
            tp = parts[5]
            if tp[:2] != "l0":
                continue

            pred_name = img_name.replace("l0", "p0", 1)
            sample_key = _sample_key_from_name(img_name)
            gt_path = gt_index.get(sample_key)

            input_path = video_folder / img_name
            pred_path = video_folder / pred_name

            if input_path.is_file() and pred_path.is_file() and gt_path is not None and gt_path.is_file():
                info(f"Low Quality Image: {img_name}")
                info(f"Prediction       : {pred_name}")
                info(f"Ground Truth     : {gt_path}")
                pairs.append(
                    {
                        "video_folder_name": video_folder_name,
                        "input_path": input_path,
                        "pred_path": pred_path,
                        "gt_path": gt_path,
                    }
                )
            else:
                info(
                    "Skip sample due to missing file(s): "
                    f"input={input_path.is_file()} pred={pred_path.is_file()} gt={bool(gt_path and gt_path.is_file())}"
                )
    return pairs


def main(_):
    import torch

    from see.losses.psnr import PSNR, NormalizedPSNR
    from see.losses.ssim import SSIM

    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()

    vis_folder = Path(FLAGS.root) / "vis"
    gt_root = Path(FLAGS.gt_root)

    f_psnr = PSNR().cuda()
    f_ssim = SSIM().cuda()
    f_npsnr = NormalizedPSNR("linear").cuda()
    f_l1 = torch.nn.L1Loss().cuda()

    all_eval = Eval("all")
    low_nml_eval = Eval("low_nml")
    hgh_nml_eval = Eval("hgh_nml")
    nml_nml_eval = Eval("nml_nml")

    for pair in collect_eval_pairs(vis_folder, gt_root):
        oi = _load_image(pair["pred_path"]).cuda().unsqueeze(0)
        gi = _load_image(pair["gt_path"]).cuda().unsqueeze(0)
        psnr = f_psnr(oi, gi).item()
        ssim = f_ssim(oi, gi).item()
        npsnr = f_npsnr(oi, gi).item()
        l1 = f_l1(oi, gi).item()

        all_eval.add(psnr, ssim, npsnr, l1)

        video_folder_name = pair["video_folder_name"]
        if "high-normal" in video_folder_name:
            hgh_nml_eval.add(psnr, ssim, npsnr, l1)
            info(f"high-normal  : {hgh_nml_eval.get()}")
        elif "normal-normal" in video_folder_name:
            nml_nml_eval.add(psnr, ssim, npsnr, l1)
            info(f"normal-normal: {nml_nml_eval.get()}")
        elif "low-normal" in video_folder_name:
            low_nml_eval.add(psnr, ssim, npsnr, l1)
            info(f"low-normal   : {low_nml_eval.get()}")

    summary = summarize_results(all_eval, low_nml_eval, hgh_nml_eval, nml_nml_eval)

    info(f"VIS ROOT      : {vis_folder}")
    info(f"GT ROOT       : {gt_root}")
    info(f"All          : count={summary['All']['count']} metrics={summary['All']['metrics']}")
    info(f"Low-Normal   : count={summary['Low-Normal']['count']} metrics={summary['Low-Normal']['metrics']}")
    info(f"High-Normal  : count={summary['High-Normal']['count']} metrics={summary['High-Normal']['metrics']}")
    info(f"Normal-Normal: count={summary['Normal-Normal']['count']} metrics={summary['Normal-Normal']['metrics']}")


if __name__ == "__main__":
    flags.mark_flag_as_required("root")
    flags.mark_flag_as_required("gt_root")
    app.run(main)
