import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _validate_split(split_dir: Path, split_name: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return [f"[{split_name}] Missing annotation file: {ann_path}"], warnings, {}

    try:
        coco = _load_json(ann_path)
    except Exception as exc:
        return [f"[{split_name}] Failed to parse JSON: {ann_path} ({exc})"], warnings, {}

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            errors.append(f"[{split_name}] Missing required top-level key: '{key}'")

    if errors:
        return errors, warnings, {}

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    if not isinstance(images, list) or not isinstance(anns, list) or not isinstance(cats, list):
        errors.append(f"[{split_name}] 'images', 'annotations', and 'categories' must be lists")
        return errors, warnings, {}

    image_ids = []
    image_by_id: Dict[int, Dict[str, Any]] = {}
    missing_files = 0
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        width = img.get("width")
        height = img.get("height")

        if img_id is None:
            errors.append(f"[{split_name}] Image entry missing 'id': {img}")
            continue
        if file_name is None:
            errors.append(f"[{split_name}] Image id={img_id} missing 'file_name'")
            continue
        if width is None or height is None:
            warnings.append(f"[{split_name}] Image id={img_id} missing width/height")

        image_ids.append(img_id)
        image_by_id[img_id] = img

        file_path = split_dir / str(file_name)
        if not file_path.exists():
            missing_files += 1

    dup_img_ids = [img_id for img_id, c in Counter(image_ids).items() if c > 1]
    if dup_img_ids:
        errors.append(f"[{split_name}] Duplicate image IDs found (sample): {dup_img_ids[:10]}")

    cat_ids = []
    cat_id_to_name: Dict[int, str] = {}
    for cat in cats:
        cat_id = cat.get("id")
        name = cat.get("name")
        if cat_id is None:
            errors.append(f"[{split_name}] Category entry missing 'id': {cat}")
            continue
        if name is None:
            warnings.append(f"[{split_name}] Category id={cat_id} missing 'name'")
            name = str(cat_id)
        cat_ids.append(cat_id)
        cat_id_to_name[cat_id] = str(name)

    dup_cat_ids = [cat_id for cat_id, c in Counter(cat_ids).items() if c > 1]
    if dup_cat_ids:
        errors.append(f"[{split_name}] Duplicate category IDs found (sample): {dup_cat_ids[:10]}")

    ann_ids = []
    cat_counts: Counter[str] = Counter()
    images_with_ann: Counter[int] = Counter()
    missing_seg = 0
    bad_bbox = 0

    valid_cat_ids = set(cat_id_to_name.keys())
    valid_img_ids = set(image_by_id.keys())

    for ann in anns:
        ann_id = ann.get("id")
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        bbox = ann.get("bbox")
        seg = ann.get("segmentation", None)

        if ann_id is None:
            errors.append(f"[{split_name}] Annotation missing 'id': {ann}")
        else:
            ann_ids.append(ann_id)

        if image_id not in valid_img_ids:
            errors.append(
                f"[{split_name}] Annotation id={ann_id} references unknown image_id={image_id}"
            )
        else:
            images_with_ann[image_id] += 1

        if category_id not in valid_cat_ids:
            errors.append(
                f"[{split_name}] Annotation id={ann_id} references unknown category_id={category_id}"
            )
        else:
            cat_counts[cat_id_to_name[category_id]] += 1

        if not isinstance(bbox, list) or len(bbox) != 4:
            bad_bbox += 1
        else:
            _, _, bw, bh = bbox
            if bw is None or bh is None or bw <= 0 or bh <= 0:
                bad_bbox += 1

        if seg in (None, [], {}):
            missing_seg += 1

    dup_ann_ids = [ann_id for ann_id, c in Counter(ann_ids).items() if c > 1]
    if dup_ann_ids:
        errors.append(f"[{split_name}] Duplicate annotation IDs found (sample): {dup_ann_ids[:10]}")

    if bad_bbox > 0:
        warnings.append(f"[{split_name}] {bad_bbox} annotations have missing/invalid bbox values")

    if len(anns) > 0:
        missing_seg_ratio = missing_seg / len(anns)
        if missing_seg_ratio > 0.10:
            warnings.append(
                f"[{split_name}] {missing_seg}/{len(anns)} annotations missing segmentation "
                f"({missing_seg_ratio:.1%}). Segmentation training may degrade."
            )

    if missing_files > 0:
        errors.append(
            f"[{split_name}] {missing_files}/{len(images)} image files referenced in JSON are missing under {split_dir}"
        )

    num_images_no_ann = sum(1 for img_id in valid_img_ids if images_with_ann[img_id] == 0)

    stats = {
        "images": len(images),
        "annotations": len(anns),
        "categories": len(cats),
        "images_with_no_annotations": num_images_no_ann,
        "top_categories": cat_counts.most_common(10),
    }

    if len(cats) == 0:
        errors.append(f"[{split_name}] No categories found")
    if len(images) == 0:
        errors.append(f"[{split_name}] No images found")
    if len(anns) == 0:
        warnings.append(f"[{split_name}] No annotations found")

    return errors, warnings, stats

def _print_split_report(split_name: str, stats: Dict[str, Any]) -> None:
    if not stats:
        return
    print(f"[{split_name}] images={stats['images']} annotations={stats['annotations']} categories={stats['categories']}")
    print(f"[{split_name}] images with no annotations={stats['images_with_no_annotations']}")
    if stats["top_categories"]:
        top_str = ", ".join([f"{name}:{count}" for name, count in stats["top_categories"]])
        print(f"[{split_name}] top categories: {top_str}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight check for VR-SBS COCO train/test dataset layout")
    parser.add_argument(
        "--dataset-root",
        default="vr_datasets/out/",
        help="Root directory containing <supercategory>/train and /test",
    )
    parser.add_argument("--supercategory", default="vr_sbs", help="Dataset subdirectory under dataset root")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    base = Path(args.dataset_root).expanduser().resolve() / args.supercategory
    train_dir = base / "train"
    test_dir = base / "test"

    print("=" * 80)
    print("VR-SBS COCO preflight")
    print("Dataset:", str(base))
    print("=" * 80)

    all_errors: List[str] = []
    all_warnings: List[str] = []

    train_errors, train_warnings, train_stats = _validate_split(train_dir, "train")
    test_errors, test_warnings, test_stats = _validate_split(test_dir, "test")

    _print_split_report("train", train_stats)
    _print_split_report("test", test_stats)

    all_errors.extend(train_errors)
    all_errors.extend(test_errors)
    all_warnings.extend(train_warnings)
    all_warnings.extend(test_warnings)

    print("-" * 80)
    if all_warnings:
        print("Warnings:")
        for w in all_warnings:
            print("  -", w)

    if all_errors:
        print("Errors:")
        for e in all_errors:
            print("  -", e)

    if all_errors:
        print("\nPreflight FAILED: fix errors before training.")
        return 2

    if args.strict and all_warnings:
        print("\nPreflight FAILED in strict mode due to warnings.")
        return 3

    print("\nPreflight PASSED.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
