#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect SEE visualization predictions into a clean folder.

This is a DVS346-eval friendly variant of ``collect_codabench_pred.py``.
It copies prediction images from

    vis_root/<scene_folder>/*_p0_*.png

to

    out_root/<scene_folder>/*_p0_*.png

By default it collects all predictions. If ``--list`` is provided, it reads a
tree-style or plain relative-path list and copies only matching predictions by
sample key.
"""

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sample_key(name: str):
    """Return the timestamp/sample prefix before p0/n0/l0 tokens."""
    stem = Path(name).stem
    for token in ("_p0_", "_n0_", "_l0_", "_g0_", "_ev_"):
        if token in stem:
            return stem.split(token, 1)[0]
    parts = stem.split("_")
    if len(parts) < 1:
        return None
    return parts[0]


def clean_tree_line(line: str):
    line = line.strip()
    if not line or line == ".":
        return None
    if "──" in line:
        line = line.split("──", 1)[1].strip()
    return line


def read_optional_list(txt_path: Path):
    """Return [(folder_name, image_name), ...] from tree-style or relative paths."""
    pairs = []
    current_folder = None
    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            item = clean_tree_line(raw_line)
            if item is None:
                continue

            path = Path(item)
            if len(path.parts) >= 2 and path.suffix.lower() in IMAGE_EXTS:
                pairs.append((path.parts[-2], path.name))
                continue

            if path.suffix.lower() in IMAGE_EXTS:
                if current_folder is None:
                    raise ValueError(f"Image found before folder: {item}")
                pairs.append((current_folder, path.name))
                continue

            current_folder = item
    return pairs


def iter_scene_folders(vis_root: Path):
    for scene_folder in sorted(p for p in vis_root.iterdir() if p.is_dir()):
        yield scene_folder


def build_pred_index(scene_folder: Path, pred_token: str):
    index = {}
    if not scene_folder.is_dir():
        return index

    marker = f"_{pred_token}_"
    for path in sorted(scene_folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if marker not in path.name:
            continue
        key = sample_key(path.name)
        if key is None:
            continue
        index.setdefault(key, path)
    return index


def collect_all(vis_root: Path, out_root: Path, pred_token: str):
    rows = []
    num_scene = 0
    num_copied = 0
    marker = f"_{pred_token}_"

    for scene_folder in iter_scene_folders(vis_root):
        num_scene += 1
        dst_scene = out_root / scene_folder.name
        for pred_path in sorted(scene_folder.iterdir()):
            if not pred_path.is_file() or pred_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if marker not in pred_path.name:
                continue
            dst_scene.mkdir(parents=True, exist_ok=True)
            dst_path = dst_scene / pred_path.name
            shutil.copy2(pred_path, dst_path)
            rows.append({
                "scene_folder": scene_folder.name,
                "sample_key": sample_key(pred_path.name),
                "src_path": str(pred_path),
                "dst_path": str(dst_path),
            })
            num_copied += 1

    return {
        "mode": "all",
        "scene_folders": num_scene,
        "requested": num_copied,
        "copied": num_copied,
        "missing": 0,
    }, rows


def collect_from_list(vis_root: Path, out_root: Path, list_path: Path, pred_token: str):
    wanted_items = read_optional_list(list_path)
    folder_cache = {}
    rows = []
    num_missing = 0

    for folder_name, image_name in wanted_items:
        key = sample_key(image_name)
        if key is None:
            print(f"[skip] bad filename: {folder_name}/{image_name}")
            num_missing += 1
            continue

        if folder_name not in folder_cache:
            folder_cache[folder_name] = build_pred_index(vis_root / folder_name, pred_token)

        pred_path = folder_cache[folder_name].get(key)
        if pred_path is None:
            print(f"[missing] {folder_name}/{image_name}")
            num_missing += 1
            continue

        dst_scene = out_root / folder_name
        dst_scene.mkdir(parents=True, exist_ok=True)
        dst_path = dst_scene / pred_path.name
        shutil.copy2(pred_path, dst_path)
        rows.append({
            "scene_folder": folder_name,
            "sample_key": key,
            "src_path": str(pred_path),
            "dst_path": str(dst_path),
        })

    return {
        "mode": "list",
        "list": str(list_path),
        "scene_folders": len(folder_cache),
        "requested": len(wanted_items),
        "copied": len(rows),
        "missing": num_missing,
    }, rows


def write_manifest(out_root: Path, summary, rows):
    manifest_path = out_root / "collect_dvs346_pred_manifest.csv"
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = out_root / "collect_dvs346_pred_summary.txt"
    lines = [f"{key}: {value}" for key, value in summary.items()]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, manifest_path if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vis_root", type=str, help="Full inference visualization root")
    parser.add_argument("out_root", type=str, help="Output folder for collected predictions")
    parser.add_argument(
        "--list",
        type=str,
        default=None,
        help="Optional tree-style/plain list. If omitted, copy all p0 predictions.",
    )
    parser.add_argument("--pred-token", default="p0", help="Prediction token to collect")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output folder")
    args = parser.parse_args()

    vis_root = Path(args.vis_root)
    out_root = Path(args.out_root)
    list_path = Path(args.list) if args.list else None

    if not vis_root.is_dir():
        raise FileNotFoundError(f"vis_root does not exist: {vis_root}")
    if list_path is not None and not list_path.is_file():
        raise FileNotFoundError(f"list file does not exist: {list_path}")

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} already exists. Use --overwrite if needed.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if list_path is None:
        summary, rows = collect_all(vis_root, out_root, args.pred_token)
    else:
        summary, rows = collect_from_list(vis_root, out_root, list_path, args.pred_token)

    summary.update({
        "vis_root": str(vis_root),
        "out_root": str(out_root),
        "pred_token": args.pred_token,
    })
    summary_path, manifest_path = write_manifest(out_root, summary, rows)

    print("Done.")
    for key, value in summary.items():
        print(f"{key:14}: {value}")
    print(f"summary       : {summary_path}")
    if manifest_path is not None:
        print(f"manifest      : {manifest_path}")


if __name__ == "__main__":
    main()
