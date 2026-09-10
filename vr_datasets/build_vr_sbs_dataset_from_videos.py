import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

@dataclass
class VideoPair:
    rgb_path: Path
    matte_path: Path
    pair_name: str

def _collect_videos(root: Path) -> List[Path]:
    vids = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    vids.sort()
    return vids

def _strip_suffix(stem: str, suffix: str) -> str:
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem

def _pair_videos(
    rgb_dir: Path,
    matte_dir: Path,
    rgb_suffix: str,
    matte_suffix: str,
) -> Tuple[List[VideoPair], List[Path]]:
    rgb_videos = _collect_videos(rgb_dir)
    matte_videos = _collect_videos(matte_dir)

    matte_map: Dict[Tuple[str, str], Path] = {}
    for mv in matte_videos:
        rel_parent = str(mv.relative_to(matte_dir).parent).replace("\\", "/")
        key = (_strip_suffix(mv.stem, matte_suffix), rel_parent)
        matte_map[key] = mv

    pairs: List[VideoPair] = []
    missing: List[Path] = []

    for rv in rgb_videos:
        rel_parent = str(rv.relative_to(rgb_dir).parent).replace("\\", "/")
        key = (_strip_suffix(rv.stem, rgb_suffix), rel_parent)
        mv = matte_map.get(key)
        if mv is None:
            missing.append(rv)
            continue

        pair_name = f"{rel_parent}/{_strip_suffix(rv.stem, rgb_suffix)}".strip("/")
        pair_name = pair_name.replace("/", "__")
        pairs.append(VideoPair(rgb_path=rv, matte_path=mv, pair_name=pair_name))

    return pairs, missing

def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap

def _bbox_and_area_from_mask(mask: np.ndarray) -> Optional[Tuple[List[float], float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    area = float((mask > 0).sum())
    return [float(x_min), float(y_min), float(w), float(h)], area

def _rle_from_mask(mask: np.ndarray) -> Dict[str, object]:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    flat = binary.flatten(order="F")

    counts: List[int] = []
    prev = 0
    run_len = 0
    for pix in flat:
        value = int(pix)
        if value == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = value
    counts.append(run_len)

    return {
        "size": [int(h), int(w)],
        "counts": counts,
    }

def _split_pairs(pairs: List[VideoPair], test_ratio: float, seed: int) -> Tuple[List[VideoPair], List[VideoPair]]:
    if len(pairs) < 2:
        return pairs, []

    rng = random.Random(seed)
    tmp = pairs[:]
    rng.shuffle(tmp)
    n_test = max(1, int(round(len(tmp) * test_ratio)))
    if n_test >= len(tmp):
        n_test = len(tmp) - 1
    return tmp[n_test:], tmp[:n_test]

def _frame_goes_to_test(pair_name: str, sample_slot: int, seed: int, test_ratio: float) -> bool:
    key = f"{pair_name}|{sample_slot}|{seed}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    score = int(h[:8], 16) / 0xFFFFFFFF
    return score < test_ratio

def _process_split(
    split_name: str,
    split_pairs: Sequence[VideoPair],
    split_dir: Path,
    category_id: int,
    category_name: str,
    sample_every: int,
    matte_threshold: int,
    max_frames_per_video: int,
    keep_empty: bool,
    resize_width: int,
    resize_height: int,
    frame_split_enabled: bool,
    frame_split_test_ratio: float,
    frame_split_seed: int,
) -> Dict[str, int]:
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    categories = [{"id": category_id, "name": category_name}]

    img_id = 1
    ann_id = 1
    total_frames_seen = 0
    total_kept = 0

    images_dir = split_dir / "images"
    ann_path = split_dir / "_annotations.coco.json"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if ann_path.exists():
        ann_path.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    for pair in split_pairs:
        rgb_cap = _open_video(pair.rgb_path)
        matte_cap = _open_video(pair.matte_path)

        frame_idx = -1
        kept_from_video = 0
        sampled_slot = -1

        while True:
            ok_rgb, rgb = rgb_cap.read()
            ok_matte, matte = matte_cap.read()

            if not ok_rgb or not ok_matte:
                break

            frame_idx += 1
            total_frames_seen += 1

            if frame_idx % sample_every != 0:
                continue

            sampled_slot += 1

            if frame_split_enabled:
                goes_to_test = _frame_goes_to_test(
                    pair_name=pair.pair_name,
                    sample_slot=sampled_slot,
                    seed=frame_split_seed,
                    test_ratio=frame_split_test_ratio,
                )
                if split_name == "train" and goes_to_test:
                    continue
                if split_name == "test" and not goes_to_test:
                    continue

            if max_frames_per_video > 0 and kept_from_video >= max_frames_per_video:
                break

            if resize_width > 0 and resize_height > 0:
                rgb = cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
                matte = cv2.resize(matte, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)

            if matte.ndim == 3:
                # Use strongest channel to preserve colored matte signals (e.g. dark red masks).
                matte_signal = np.max(matte, axis=2)
            else:
                matte_signal = matte

            mask = (matte_signal > matte_threshold).astype(np.uint8)
            bbox_area = _bbox_and_area_from_mask(mask)
            if bbox_area is None and not keep_empty:
                continue

            h, w = rgb.shape[:2]
            file_name = f"{pair.pair_name}_f{frame_idx:06d}.png"
            rel_file = f"images/{file_name}"
            out_path = images_dir / file_name

            cv2.imwrite(str(out_path), rgb)

            images.append(
                {
                    "id": img_id,
                    "file_name": rel_file,
                    "width": int(w),
                    "height": int(h),
                }
            )

            if bbox_area is not None:
                bbox, area = bbox_area
                seg = _rle_from_mask(mask)

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": seg,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            img_id += 1
            kept_from_video += 1
            total_kept += 1

        rgb_cap.release()
        matte_cap.release()

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    print(
        f"[{split_name}] pairs={len(split_pairs)} frames_seen={total_frames_seen} kept={total_kept} "
        f"images={len(images)} anns={len(annotations)}"
    )

    return {
        "pairs": len(split_pairs),
        "frames_seen": total_frames_seen,
        "kept": total_kept,
        "images": len(images),
        "annotations": len(annotations),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Build VR-SBS COCO dataset directly from RGB+matte video pairs")
    parser.add_argument(
        "--videos-dir",
        default="vr_datasets/rgb_videos",
        help="Directory with RGB videos",
    )
    parser.add_argument("--mattes-dir", default="vr_datasets/matte_videos", help="Directory with matte videos")
    parser.add_argument(
        "--out-root", default="vr_datasets/out", help="Output dataset root"
    )
    parser.add_argument("--supercategory", default="vr_sbs")
    parser.add_argument("--rgb-suffix", default="", help="Suffix to strip from RGB stem before pairing")
    parser.add_argument("--matte-suffix", default="_matte", help="Suffix to strip from matte stem before pairing")
    parser.add_argument("--sample-every", type=int, default=10, help="Keep one frame every N frames")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Split ratio by video pair")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--matte-threshold", type=int, default=127)
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="If >0, cap kept frames per video")
    parser.add_argument("--keep-empty", action="store_true", help="Keep frames with empty matte as negatives")
    parser.add_argument("--resize-width", type=int, default=0, help="Optional output width")
    parser.add_argument("--resize-height", type=int, default=0, help="Optional output height")
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--category-name", default="person")
    parser.add_argument("--disable-single-pair-frame-split", action="store_true",
        help="Disable automatic frame-level train/test split fallback when only one video pair is matched",
    )
    args = parser.parse_args()

    if args.sample_every <= 0:
        raise ValueError("--sample-every must be >= 1")
    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0,1)")
    if (args.resize_width > 0) != (args.resize_height > 0):
        raise ValueError("Set both --resize-width and --resize-height together (or neither)")

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    mattes_dir = Path(args.mattes_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    dataset_root = out_root / args.supercategory

    if not videos_dir.exists():
        raise FileNotFoundError(f"videos-dir not found: {videos_dir}")
    if not mattes_dir.exists():
        raise FileNotFoundError(f"mattes-dir not found: {mattes_dir}")

    pairs, missing = _pair_videos(
        rgb_dir=videos_dir,
        matte_dir=mattes_dir,
        rgb_suffix=args.rgb_suffix,
        matte_suffix=args.matte_suffix,
    )

    if not pairs:
        raise RuntimeError("No RGB/matte video pairs found. Check folder structure and suffix options.")

    train_pairs, test_pairs = _split_pairs(pairs, args.test_ratio, args.seed)

    use_single_pair_frame_split = (
        len(pairs) == 1
        and len(test_pairs) == 0
        and args.test_ratio > 0
        and not args.disable_single_pair_frame_split
    )
    if use_single_pair_frame_split:
        train_pairs = pairs
        test_pairs = pairs

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    print("=" * 80)
    print("VR-SBS video+matte dataset builder")
    print(f"Videos dir     : {videos_dir}")
    print(f"Mattes dir     : {mattes_dir}")
    print(f"Output dataset : {dataset_root}")
    print(f"Matched pairs  : {len(pairs)}")
    print(f"Missing pairs  : {len(missing)}")
    if use_single_pair_frame_split:
        print("Split mode     : single-pair frame-level split enabled")
    print("=" * 80)

    train_stats = _process_split(
        split_name="train",
        split_pairs=train_pairs,
        split_dir=train_dir,
        category_id=args.category_id,
        category_name=args.category_name,
        sample_every=args.sample_every,
        matte_threshold=args.matte_threshold,
        max_frames_per_video=args.max_frames_per_video,
        keep_empty=args.keep_empty,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        frame_split_enabled=use_single_pair_frame_split,
        frame_split_test_ratio=args.test_ratio,
        frame_split_seed=args.seed,
    )
    test_stats = _process_split(
        split_name="test",
        split_pairs=test_pairs,
        split_dir=test_dir,
        category_id=args.category_id,
        category_name=args.category_name,
        sample_every=args.sample_every,
        matte_threshold=args.matte_threshold,
        max_frames_per_video=args.max_frames_per_video,
        keep_empty=args.keep_empty,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        frame_split_enabled=use_single_pair_frame_split,
        frame_split_test_ratio=args.test_ratio,
        frame_split_seed=args.seed,
    )

    if missing:
        print("-" * 80)
        print("Sample RGB videos with no matching matte (up to 10):")
        for p in missing[:10]:
            print(f"  - {p}")

    print("-" * 80)
    print(f"Train summary: {train_stats}")
    print(f"Test  summary: {test_stats}")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
