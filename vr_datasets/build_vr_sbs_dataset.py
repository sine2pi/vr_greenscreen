import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

@dataclass
class PairItem:
    image_path: Path
    mask_path: Path
    rel_key: str

def _collect_images(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files

def _find_mask_for_image(
    image_path: Path,
    images_root: Path,
    masks_root: Path,
    mask_suffix: str,
) -> Optional[Path]:
    rel = image_path.relative_to(images_root)
    stem = rel.stem
    parent = rel.parent

    candidates: List[Path] = []
    for ext in IMAGE_EXTS:
        candidates.append(masks_root / parent / f"{stem}{mask_suffix}{ext}")
        candidates.append(masks_root / parent / f"{stem}{ext}")

    for cand in candidates:
        if cand.exists():
            return cand
    return None

def _pair_images_and_masks(
    images_root: Path,
    masks_root: Path,
    mask_suffix: str,
) -> Tuple[List[PairItem], List[Path]]:
    image_files = _collect_images(images_root)
    pairs: List[PairItem] = []
    missing_masks: List[Path] = []

    for img in image_files:
        mask = _find_mask_for_image(img, images_root, masks_root, mask_suffix)
        if mask is None:
            missing_masks.append(img)
            continue
        rel_key = str(img.relative_to(images_root)).replace("\\", "/")
        pairs.append(PairItem(image_path=img, mask_path=mask, rel_key=rel_key))

    return pairs, missing_masks

def _mask_to_annotation(mask_path: Path, threshold: int) -> Optional[Dict[str, object]]:
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    fg = arr > threshold

    if not np.any(fg):
        return None

    ys, xs = np.where(fg)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    area = int(fg.sum())

    seg = [[
        float(x_min),
        float(y_min),
        float(x_min + w),
        float(y_min),
        float(x_min + w),
        float(y_min + h),
        float(x_min),
        float(y_min + h),
    ]]

    return {
        "bbox": [float(x_min), float(y_min), float(w), float(h)],
        "area": float(area),
        "segmentation": seg,
        "iscrowd": 0,
    }

def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def _build_split(
    items: Sequence[PairItem],
    split_dir: Path,
    category_id: int,
    category_name: str,
    threshold: int,
    keep_empty: bool,
) -> Dict[str, object]:
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    categories = [{"id": int(category_id), "name": category_name}]

    image_id = 1
    ann_id = 1

    copied_images = 0
    kept_empty = 0

    for item in items:
        with Image.open(item.image_path) as im:
            w, h = im.size

        ann_core = _mask_to_annotation(item.mask_path, threshold)
        if ann_core is None and not keep_empty:
            continue

        rel_img = Path("images") / Path(item.rel_key)
        dst_img = split_dir / rel_img
        _safe_copy(item.image_path, dst_img)

        images.append(
            {
                "id": image_id,
                "file_name": str(rel_img).replace("\\", "/"),
                "width": int(w),
                "height": int(h),
            }
        )
        copied_images += 1

        if ann_core is None:
            kept_empty += 1
        else:
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(category_id),
                **ann_core,
            }
            annotations.append(ann)
            ann_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_json = split_dir / "_annotations.coco.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    return {
        "json_path": str(out_json),
        "images": len(images),
        "annotations": len(annotations),
        "kept_empty_images": kept_empty,
        "copied_images": copied_images,
    }

def _split_items(items: List[PairItem], test_ratio: float, seed: int) -> Tuple[List[PairItem], List[PairItem]]:
    rng = random.Random(seed)
    items_copy = items[:]
    rng.shuffle(items_copy)

    if len(items_copy) < 2:
        return items_copy, []

    test_count = max(1, int(round(len(items_copy) * test_ratio)))
    if test_count >= len(items_copy):
        test_count = len(items_copy) - 1

    test_items = items_copy[:test_count]
    train_items = items_copy[test_count:]
    return train_items, test_items

def main() -> int:
    parser = argparse.ArgumentParser(description="Build VR-SBS COCO dataset from image+mask pairs")
    parser.add_argument(
        "--videos-dir",
        default="vr_datasets/rgb_videos",
        help="Directory with RGB videos",
    )
    parser.add_argument(
        "--mattes-dir",
        default="vr_datasets/matte_videos",
        help="Directory with matte videos",
    )
    parser.add_argument(
        "--out-root", default="vr_datasets/out", help="Output dataset root"
    )
    
    parser.add_argument("--supercategory", default="vr_sbs", help="Subfolder name under output root")
    parser.add_argument("--mask-suffix", default="_mask", help="Optional mask suffix (e.g. frame_0001_mask.png)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio in [0,1)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mask-threshold", type=int, default=127, help="Foreground threshold for grayscale masks")
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--category-name", default="person")
    parser.add_argument("--keep-empty", action="store_true", help="Keep images with empty masks as negative samples")
    parser.add_argument("--max-items", type=int, default=0, help="If >0, cap number of pairs before split")
    args = parser.parse_args()

    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")

    images_root = Path(args.images_dir).expanduser().resolve()
    masks_root = Path(args.masks_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    dataset_root = out_root / args.supercategory

    if not images_root.exists():
        raise FileNotFoundError(f"Images dir not found: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks dir not found: {masks_root}")

    pairs, missing = _pair_images_and_masks(images_root, masks_root, args.mask_suffix)
    if args.max_items > 0:
        pairs = pairs[: args.max_items]

    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check naming, --mask-suffix, and folder structure.")

    train_items, test_items = _split_items(pairs, test_ratio=args.test_ratio, seed=args.seed)

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    train_stats = _build_split(
        items=train_items,
        split_dir=train_dir,
        category_id=args.category_id,
        category_name=args.category_name,
        threshold=args.mask_threshold,
        keep_empty=args.keep_empty,
    )
    test_stats = _build_split(
        items=test_items,
        split_dir=test_dir,
        category_id=args.category_id,
        category_name=args.category_name,
        threshold=args.mask_threshold,
        keep_empty=args.keep_empty,
    )

    print("=" * 80)
    print("VR-SBS dataset build complete")
    print(f"Input images dir : {images_root}")
    print(f"Input masks dir  : {masks_root}")
    print(f"Output dataset   : {dataset_root}")
    print(f"Matched pairs    : {len(pairs)}")
    print(f"Missing masks    : {len(missing)}")
    print("-" * 80)
    print(f"Train -> images={train_stats['images']} anns={train_stats['annotations']} json={train_stats['json_path']}")
    print(f"Test  -> images={test_stats['images']} anns={test_stats['annotations']} json={test_stats['json_path']}")
    if missing:
        preview = "\n".join([f"  - {p}" for p in missing[:10]])
        print("-" * 80)
        print("Sample files missing masks (up to 10):")
        print(preview)
    print("=" * 80)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
