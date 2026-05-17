# scoring.py
import sys
import os
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as calc_ssim
    SSIM_API = "new"
except ImportError:
    from skimage.measure import compare_ssim as calc_ssim
    SSIM_API = "old"


CATEGORIES = ["high-normal", "low-normal", "normal-normal"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

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
            return PENALTY_PSNR, PENALTY_SSIM
        return self.psnr / self.count, self.ssim / self.count


def log(msg):
    print(msg, flush=True)


def detect_category(folder_name):
    for cat in CATEGORIES:
        if cat in folder_name:
            return cat
    return None


def unwrap_root(root):
    """
    Codabench may unzip a submission as:
        res/submission_name/scene_folders
    This unwraps one redundant top-level folder.
    """
    if not os.path.isdir(root):
        return root

    items = [x for x in os.listdir(root) if not x.startswith(".")]
    if len(items) == 1:
        only_path = os.path.join(root, items[0])
        if os.path.isdir(only_path):
            return only_path

    return root


def list_scene_folders(root):
    folders = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            folders.append(path)
    return sorted(folders, key=lambda x: os.path.basename(x))


def list_images(folder, prefer_p0=False):
    if not os.path.isdir(folder):
        return []

    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
            files.append(path)

    files = sorted(files, key=lambda x: os.path.basename(x))

    # If the submitted folder is a visualization folder, only use predictions.
    if prefer_p0:
        p0_files = [p for p in files if "_p0_" in os.path.basename(p)]
        if len(p0_files) > 0:
            return p0_files

    return files


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(pred, gt):
    h, w = gt.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1

    if SSIM_API == "new":
        return float(
            calc_ssim(
                gt,
                pred,
                channel_axis=-1,
                data_range=1.0,
                win_size=win_size,
            )
        )

    return float(
        calc_ssim(
            gt,
            pred,
            multichannel=True,
            data_range=1.0,
            win_size=win_size,
        )
    )


def evaluate_pair(gt_path, pred_path):
    gt = load_rgb(gt_path)
    pred = load_rgb(pred_path)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: GT={gt.shape}, Pred={pred.shape}")

    psnr = compute_psnr(pred, gt)
    ssim = compute_ssim(pred, gt)

    return psnr, ssim


def write_scores(output_dir, scores):
    os.makedirs(output_dir, exist_ok=True)
    score_file = os.path.join(output_dir, "scores.json")
    with open(score_file, "w") as f:
        json.dump(scores, f, indent=2)
    log(json.dumps(scores, indent=2))


def main():
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: python scoring.py <input_dir> <output_dir>")

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    gt_root = os.path.join(input_dir, "ref")
    pred_root = os.path.join(input_dir, "res")

    gt_root = unwrap_root(gt_root)
    pred_root = unwrap_root(pred_root)

    log(f"Input dir : {input_dir}")
    log(f"Output dir: {output_dir}")
    log(f"GT root   : {gt_root}")
    log(f"Pred root : {pred_root}")
    log(f"SSIM API  : {SSIM_API}")

    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    if not os.path.isdir(pred_root):
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    meters = {cat: Meter() for cat in CATEGORIES}

    num_gt = 0
    num_matched = 0
    num_missing = 0
    num_failed = 0
    num_extra_pred = 0

    gt_folders = list_scene_folders(gt_root)

    if len(gt_folders) == 0:
        raise RuntimeError(f"No GT folders found in {gt_root}")

    for gt_folder in gt_folders:
        folder_name = os.path.basename(gt_folder)
        pred_folder = os.path.join(pred_root, folder_name)

        cat = detect_category(folder_name)
        if cat is None:
            log(f"Skip unknown category folder: {folder_name}")
            continue

        gt_files = list_images(gt_folder, prefer_p0=False)
        pred_files = list_images(pred_folder, prefer_p0=True)

        if len(gt_files) == 0:
            log(f"Skip empty GT folder: {folder_name}")
            continue

        if not os.path.isdir(pred_folder):
            log(f"Missing prediction folder: {folder_name}")
            num_gt += len(gt_files)
            num_missing += len(gt_files)
            for _ in gt_files:
                meters[cat].add(PENALTY_PSNR, PENALTY_SSIM)
            continue

        if len(pred_files) == 0:
            log(f"No prediction images found in folder: {folder_name}")
            num_gt += len(gt_files)
            num_missing += len(gt_files)
            for _ in gt_files:
                meters[cat].add(PENALTY_PSNR, PENALTY_SSIM)
            continue

        if len(pred_files) < len(gt_files):
            log(
                f"Warning: fewer predictions in {folder_name}: "
                f"GT={len(gt_files)}, Pred={len(pred_files)}"
            )

        if len(pred_files) > len(gt_files):
            extra = len(pred_files) - len(gt_files)
            num_extra_pred += extra
            log(
                f"Warning: more predictions in {folder_name}: "
                f"GT={len(gt_files)}, Pred={len(pred_files)}. Extra ignored."
            )

        pair_count = min(len(gt_files), len(pred_files))

        for i in range(pair_count):
            gt_path = gt_files[i]
            pred_path = pred_files[i]
            num_gt += 1

            try:
                psnr, ssim = evaluate_pair(gt_path, pred_path)
                meters[cat].add(psnr, ssim)
                num_matched += 1

                if num_matched <= 5:
                    log(
                        f"Pair check [{folder_name}]: "
                        f"GT={os.path.basename(gt_path)} | "
                        f"Pred={os.path.basename(pred_path)} | "
                        f"PSNR={psnr:.4f}, SSIM={ssim:.4f}"
                    )

                if num_matched % 20 == 0:
                    log(f"Evaluated {num_matched} matched images.")

            except Exception as e:
                num_failed += 1
                meters[cat].add(PENALTY_PSNR, PENALTY_SSIM)
                log(
                    f"Failed [{folder_name}]: "
                    f"GT={os.path.basename(gt_path)}, "
                    f"Pred={os.path.basename(pred_path)}, "
                    f"error={e}"
                )

        missing_count = len(gt_files) - pair_count
        if missing_count > 0:
            num_gt += missing_count
            num_missing += missing_count
            for _ in range(missing_count):
                meters[cat].add(PENALTY_PSNR, PENALTY_SSIM)

    high_psnr, high_ssim = meters["high-normal"].avg()
    low_psnr, low_ssim = meters["low-normal"].avg()
    norm_psnr, norm_ssim = meters["normal-normal"].avg()

    final_psnr = (high_psnr + low_psnr + norm_psnr) / 3.0
    final_ssim = (high_ssim + low_ssim + norm_ssim) / 3.0

    scores = {
        "PSNR": float(final_psnr),
        "SSIM": float(final_ssim),

        "High_Norm_PSNR": float(high_psnr),
        "High_Norm_SSIM": float(high_ssim),

        "Low_Norm_PSNR": float(low_psnr),
        "Low_Norm_SSIM": float(low_ssim),

        "Norm_Norm_PSNR": float(norm_psnr),
        "Norm_Norm_SSIM": float(norm_ssim),

        "num_gt": int(num_gt),
        "num_matched": int(num_matched),
        "num_missing": int(num_missing),
        "num_failed": int(num_failed),
        "num_extra_pred": int(num_extra_pred),
    }

    write_scores(output_dir, scores)
    log("Evaluation completed successfully.")


if __name__ == "__main__":
    main()