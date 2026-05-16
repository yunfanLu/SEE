import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sample_key(name: str):
    parts = Path(name).stem.split("_")
    if len(parts) < 5:
        return None
    return "_".join(parts[:5])


def clean_tree_line(line: str):
    line = line.strip()
    if not line or line == ".":
        return None

    if "──" in line:
        line = line.split("──", 1)[1].strip()

    return line


def read_mini_list(txt_path: Path):
    """
    Return:
        [(folder_name, gt_image_name), ...]
    Supports tree-style txt and plain relative paths.
    """
    pairs = []
    current_folder = None

    with open(txt_path, "r") as f:
        for raw_line in f:
            item = clean_tree_line(raw_line)
            if item is None:
                continue

            path = Path(item)

            # case 1: folder/image.png
            if len(path.parts) >= 2 and path.suffix.lower() in IMAGE_EXTS:
                folder = path.parts[-2]
                img_name = path.name
                pairs.append((folder, img_name))
                continue

            # case 2: tree-style image line
            if path.suffix.lower() in IMAGE_EXTS:
                if current_folder is None:
                    raise ValueError(f"Image found before folder: {item}")
                pairs.append((current_folder, path.name))
                continue

            # case 3: folder line
            current_folder = item

    return pairs


def build_pred_index(pred_folder: Path):
    """
    Index p0 prediction files by sample key.
    """
    index = {}

    if not pred_folder.is_dir():
        return index

    for p in sorted(pred_folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        if "_p0_" not in p.name:
            continue

        key = sample_key(p.name)
        if key is None:
            continue

        if key not in index:
            index[key] = p

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vis_root", type=str, help="Full inference visualization root")
    parser.add_argument("out_root", type=str, help="Output folder for Codabench submission")
    parser.add_argument(
        "--list",
        type=str,
        default="codabench/SEE_gt_mini.txt",
        help="GT mini file list",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output folder if it already exists",
    )
    args = parser.parse_args()

    vis_root = Path(args.vis_root)
    out_root = Path(args.out_root)
    list_path = Path(args.list)

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out_root} already exists. Use --overwrite if needed."
            )
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    gt_items = read_mini_list(list_path)

    folder_cache = {}

    num_total = 0
    num_copied = 0
    num_missing = 0

    for folder_name, gt_img_name in gt_items:
        num_total += 1

        key = sample_key(gt_img_name)
        if key is None:
            print(f"[skip] bad GT filename: {folder_name}/{gt_img_name}")
            num_missing += 1
            continue

        if folder_name not in folder_cache:
            pred_folder = vis_root / folder_name
            folder_cache[folder_name] = build_pred_index(pred_folder)

        pred_index = folder_cache[folder_name]
        pred_path = pred_index.get(key)

        if pred_path is None:
            print(f"[missing] {folder_name}/{gt_img_name}")
            num_missing += 1
            continue

        dst_folder = out_root / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        dst_path = dst_folder / pred_path.name
        shutil.copy2(pred_path, dst_path)

        num_copied += 1

    print("Done.")
    print(f"GT list      : {list_path}")
    print(f"VIS root     : {vis_root}")
    print(f"Output root  : {out_root}")
    print(f"Total        : {num_total}")
    print(f"Copied       : {num_copied}")
    print(f"Missing      : {num_missing}")


if __name__ == "__main__":
    main()