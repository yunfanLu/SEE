# eval.py
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as calc_ssim


CATEGORIES = ["high-normal", "low-normal", "normal-normal"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PENALTY_PSNR = 0.0
PENALTY_SSIM = 0.0


class Meter:
    def __init__(self):
        self.count = 0
        self.psnr = 0.0
        self.ssim = 0.0

    def add(self, psnr, ssim):
        self.count += 1
        self.psnr += float(psnr)
        self.ssim += float(ssim)

    def avg(self):
        if self.count == 0:
            return {"PSNR": PENALTY_PSNR, "SSIM": PENALTY_SSIM}
        return {
            "PSNR": self.psnr / self.count,
            "SSIM": self.ssim / self.count,
        }


def warn(msg):
    print(msg, file=sys.stderr)


def sample_key(path: Path):
    parts = path.stem.split("_")
    if len(parts) < 5:
        return None
    return "_".join(parts[:5])


def detect_category(folder_name: str):
    for cat in CATEGORIES:
        if cat in folder_name:
            return cat
    return None


def build_index(folder: Path):
    index = {}

    if not folder.is_dir():
        return index

    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        key = sample_key(p)
        if key is None:
            warn(f"[skip] bad filename: {p}")
            continue

        if key in index:
            warn(f"[warning] duplicate key: {key}, keep first: {index[key]}")
            continue

        index[key] = p

    return index


def load_rgb(path: Path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return 100.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(pred, gt):
    return calc_ssim(gt, pred, channel_axis=-1, data_range=1.0)


def eval_pair(pred_path: Path, gt_path: Path):
    pred = load_rgb(pred_path)
    gt = load_rgb(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    return compute_psnr(pred, gt), compute_ssim(pred, gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_root", type=str)
    parser.add_argument("pred_root", type=str)
    parser.add_argument("--out", type=str, default="scores.json")
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)

    meters = {cat: Meter() for cat in CATEGORIES}

    num_gt = 0
    num_matched = 0
    num_missing = 0
    num_failed = 0

    for gt_folder in sorted(gt_root.iterdir()):
        if not gt_folder.is_dir():
            continue

        cat = detect_category(gt_folder.name)
        if cat is None:
            warn(f"[skip] unknown category folder: {gt_folder.name}")
            continue

        pred_folder = pred_root / gt_folder.name

        gt_index = build_index(gt_folder)
        pred_index = build_index(pred_folder)

        if not pred_folder.is_dir():
            warn(f"[missing folder] {pred_folder}")

        for key, gt_path in gt_index.items():
            num_gt += 1
            pred_path = pred_index.get(key)

            if pred_path is None:
                num_missing += 1
                meters[cat].add(PENALTY_PSNR, PENALTY_SSIM)
                warn(f"[missing pred] {gt_folder.name}/{gt_path.name}")
                continue

            try:
                psnr, ssim = eval_pair(pred_path, gt_path)
                meters[cat].add(psnr, ssim)
                num_matched += 1
            except Exception as e:
                num_failed += 1
                meters[cat].add(PENALTY_PSNR, PENALTY_SSIM)
                warn(f"[failed] pred={pred_path}, gt={gt_path}, error={e}")

    cat_avg = {cat: meters[cat].avg() for cat in CATEGORIES}

    final_psnr = sum(cat_avg[cat]["PSNR"] for cat in CATEGORIES) / len(CATEGORIES)
    final_ssim = sum(cat_avg[cat]["SSIM"] for cat in CATEGORIES) / len(CATEGORIES)

    scores = {
        "PSNR": final_psnr,
        "SSIM": final_ssim,

        "High_Norm_PSNR": cat_avg["high-normal"]["PSNR"],
        "High_Norm_SSIM": cat_avg["high-normal"]["SSIM"],

        "Low_Norm_PSNR": cat_avg["low-normal"]["PSNR"],
        "Low_Norm_SSIM": cat_avg["low-normal"]["SSIM"],

        "Norm_Norm_PSNR": cat_avg["normal-normal"]["PSNR"],
        "Norm_Norm_SSIM": cat_avg["normal-normal"]["SSIM"],

        "num_gt": num_gt,
        "num_matched": num_matched,
        "num_missing": num_missing,
        "num_failed": num_failed,
    }

    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2)

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()