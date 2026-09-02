import argparse, shutil, gc, os, sys, functools, time, torch, cv2, imageio, numpy as np, tqdm, random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image
from typing import List
import torch.nn.functional as F
from ffmpeg import  norm_video, info, concat_video, extract_segment_frames, mask_overlay, stereo_video, read_frame_from_videos, timestamp, format_timestamp, FISHEYE180_PIPELINE_MODE, packer, run_fisheye180_mode, pack_video
from sammy import sam3_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
_setup_tf32()

class SegmentType(Enum):
    MASK = 'mask'

@dataclass
class SegmentInfo:
    index: int
    start_time: float
    end_time: float
    seg_type: SegmentType
    left_frame_path: str = ''
    right_frame_path: str = ''
    left_mask_path: str = ''
    right_mask_path: str = ''
    video_path: str = ''

ENCODER = 'hevc_nvenc'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

SAM3_MAX = 1008
BATCH_SIZE = 50

import gc, argparse, os, sys, functools, time, torch, cv2, imageio, numpy as np, tqdm, random
from pathlib import Path
from PIL import Image
from typing import Callable, List
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from omegaconf import open_dict
from ffmpeg import stereo_video, read_frame_from_videos

class SegmentType(Enum):
    MASK = 'mask'

@dataclass
class SegmentInfo:
    index: int
    start_time: float
    end_time: float
    seg_type: SegmentType
    left_frame_path: str = ''
    right_frame_path: str = ''
    left_mask_path: str = ''
    right_mask_path: str = ''
    video_path: str = ''

_matanyone_is_first_status = True
_matanyone_tqdm_lines = 1

ENCODER = 'hevc_nvenc'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

SAM3_MAX = 1008
BATCH_SIZE = 50

MATANYONE_V1 = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
MATANYONE_V2 = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"

def _update_status(op_num: int, total_ops: int, label: str, duration: float) -> None:

    global _matanyone_is_first_status

    if not _matanyone_is_first_status:
        sys.stderr.write(f"\033[{1 + _matanyone_tqdm_lines}A")

    sys.stderr.write(f"\r[{op_num}/{total_ops}] {label} ({duration:.1f}s)\033[K\n")

    for i in range(_matanyone_tqdm_lines):
        sys.stderr.write("\r\033[K")

        if i < _matanyone_tqdm_lines - 1:
            sys.stderr.write("\n")

    sys.stderr.flush()
    _matanyone_is_first_status = False

@functools.lru_cache(maxsize=2)
def _load_matanyone_runtime(version: str = 'v2'):

    version = str(version).lower()

    if version == 'v1':
        matanyone_root = Path(__file__).resolve().parent / 'MatAnyone'
        matanyone_root_str = str(matanyone_root)

        if matanyone_root_str not in sys.path:
            sys.path.insert(0, matanyone_root_str)

        from MatAnyone.matanyone.inference.inference_core import InferenceCore
        from MatAnyone.matanyone.utils.get_default_model import get_matanyone_model
        from MatAnyone.matanyone.utils.device import get_default_device

        device = get_default_device()
        pretrain_model_url = MATANYONE_V1
        model_dir = matanyone_root / 'pretrained_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = model_dir / 'matanyone.pth'

        if not ckpt_path.exists():
            sys.stderr.write(" Downloading MatAnyone v1 weights...\n")
            sys.stderr.flush()
            torch.hub.download_url_to_file(pretrain_model_url, str(ckpt_path), progress=False)

        model = get_matanyone_model(str(ckpt_path), device)
        return model, device, InferenceCore, 'v1'

    if version == 'v2':
        matanyone_root = Path(__file__).resolve().parent / 'MatAnyone2'
        matanyone_root_str = str(matanyone_root)

        if matanyone_root_str not in sys.path:
            sys.path.insert(0, matanyone_root_str)

        from MatAnyone2.matanyone2.inference.inference_core import InferenceCore
        from MatAnyone2.matanyone2.utils.get_default_model import get_matanyone2_model
        from MatAnyone2.matanyone2.utils.device import get_default_device

        device = get_default_device()
        pretrain_model_url = MATANYONE_V2
        model_dir = matanyone_root / 'pretrained_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = model_dir / 'matanyone2.pth'

        if not ckpt_path.exists():
            sys.stderr.write(" Downloading MatAnyone2 weights...\n")
            sys.stderr.flush()
            torch.hub.download_url_to_file(pretrain_model_url, str(ckpt_path), progress=False)

        model = get_matanyone2_model(str(ckpt_path), device)

        return model, device, InferenceCore, 'v2'

    raise ValueError(f"Unsupported MatAnyone version: {version}")

def _apply_matanyone_config_overrides(matanyone_model, job: dict, *, verbose: bool = False) -> None:

    cfg = matanyone_model.cfg
    mem_every = job.get('ma2_mem_every')
    max_mem_frames = job.get('ma2_max_mem_frames')
    use_long_term = job.get('ma2_use_long_term')

    if mem_every is None and max_mem_frames is None and use_long_term is None:
        return

    with open_dict(cfg):

        if mem_every is not None:
            cfg.mem_every = int(mem_every)

        if use_long_term is not None:
            cfg.use_long_term = bool(use_long_term)

        if max_mem_frames is not None:
            max_mem_frames = int(max_mem_frames)
            cfg.max_mem_frames = max_mem_frames

            if cfg.long_term.min_mem_frames > max_mem_frames:
                cfg.long_term.min_mem_frames = max_mem_frames

            cfg.long_term.max_mem_frames = max_mem_frames

    if verbose:
        mode = 'on' if cfg.use_long_term else 'off'
        version = str(job.get('matanyone_version', 'v2')).lower()
        model_name = 'MatAnyone v1' if version == 'v1' else 'MatAnyone2'
        sys.stderr.write(
            f" {model_name} cfg override => mem_every={cfg.mem_every}, "
            f"max_mem_frames={cfg.max_mem_frames}, long_term={mode}, "
            f"long_term.max_mem_frames={cfg.long_term.max_mem_frames}\n"
        )
        sys.stderr.flush()

def _apply_temporal_median_filter(phas: np.ndarray, window: int) -> np.ndarray:

    if window <= 1:
        return phas

    try:
        from scipy.ndimage import median_filter

    except Exception as exc:
        raise RuntimeError(
            "Temporal median filtering requires scipy. "
            "Install scipy or set --temporal-median-window 0."
        ) from exc

    return median_filter(phas, size=(window, 1, 1, 1), mode='nearest-exact').astype(np.uint8)

def str_to_list(value):
    return list(map(int, value.split(',')))

def gen_dilate(alpha, min_kernel_size, max_kernel_size):
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
    return dilate.astype(np.float32)

def gen_erosion(alpha, min_kernel_size, max_kernel_size):
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1)*255
    return erode.astype(np.float32)

@torch.inference_mode()

def _matanyone_process_segment(matanyone_model, device, inference_core_cls, job: dict) -> str:

    n_warmup = int(job.get('warmup', 6))
    input_path = job['input_path']
    mask_path = job['mask_path']
    max_size = int(job.get('mask_height', 1008))
    output_path = job['output_path']
    r_erode = int(job.get('erode', 0))
    r_dilate = int(job.get('dilate', 0))
    suffix = job.get('suffix', '')
    temporal_median_window = int(job.get('temporal_median_window', 0))

    _apply_matanyone_config_overrides(matanyone_model, job, verbose=(job.get('op_num', 1) == 1))
    processor = inference_core_cls(matanyone_model, cfg=matanyone_model.cfg)

    frames, fps, length, video_name = read_frame_from_videos(input_path)
    frames = frames.float()

    repeated_frames = frames[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    frames = torch.cat([repeated_frames, frames], dim=0).float()
    length += n_warmup

    os.makedirs(output_path, exist_ok=True)

    if suffix:
        video_name = f'{video_name}_{suffix}'

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)

    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    if mask.shape != (max_size, max_size):
        
        mask = cv2.resize(mask.astype(np.uint8), (max_size, max_size), interpolation=cv2.INTER_AREA).astype(np.float32)

    mask = torch.from_numpy(mask).float().to(device)

    objects = [1]
    phas = []

    for ti in tqdm.tqdm(range(length)):
        image = frames[ti]
        image = (image / 255.).float().to(device)

        if ti == 0:

            output_prob = processor.step(image, mask, objects=objects)
            output_prob = processor.step(image, first_frame_pred=True)

        elif ti <= n_warmup:
            output_prob = processor.step(image, first_frame_pred=True)

        else:
            output_prob = processor.step(image)

        mask = processor.output_prob_to_mask(output_prob)
        pha = mask.unsqueeze(2).cpu().numpy()

        if ti > (n_warmup-1):

            pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)
            phas.append(pha)

    phas_np = np.array(phas, dtype=np.uint8)

    if temporal_median_window > 1 and phas_np.shape[0] >= temporal_median_window:
        phas_np = _apply_temporal_median_filter(phas_np, temporal_median_window)

    output_file = os.path.join(output_path, f'{video_name}_pha.mp4')

    imageio.mimwrite(output_file, phas_np, fps=fps, quality=7)

    del processor, frames, phas, phas_np, mask
    torch.cuda.empty_cache()

    gc.collect()
    return output_file

def matanyone_inference(jobs: list[dict], on_segment_done: Callable[[str], None] = None) -> list[str]:
    global _matanyone_is_first_status

    max_retries = 1
    remaining_jobs = list(jobs)
    completed_paths = []

    if not remaining_jobs:
        return completed_paths

    version = str(remaining_jobs[0].get('matanyone_version', 'v2')).lower()

    for job in remaining_jobs:
        job_version = str(job.get('matanyone_version', version)).lower()

        if job_version != version:
            raise RuntimeError(f"Mixed MatAnyone versions in one batch are not supported: {version} vs {job_version}")

    matanyone_model, device, inference_core_cls, loaded_version = _load_matanyone_runtime(version)

    if loaded_version != version:
        raise RuntimeError(f"Loaded model version mismatch: expected {version}, got {loaded_version}")

    for attempt in range(max_retries):

        batch_completed = []
        _matanyone_is_first_status = True

        try:
            for job in remaining_jobs:

                _update_status(job['op_num'], job['total_ops'], job['label'], job['duration'])

                output_file = _matanyone_process_segment(

                    matanyone_model,
                    device,
                    inference_core_cls,
                    job,
                )

                batch_completed.append(output_file)

                if on_segment_done:
                    on_segment_done(output_file)

            completed_paths.extend(batch_completed)
            sys.stderr.write("\n")

            return completed_paths

        except Exception as exc:

            completed_paths.extend(batch_completed)

            remaining_jobs = remaining_jobs[len(batch_completed):]

            if not remaining_jobs:
                sys.stderr.write("\n")

                return completed_paths

            if attempt < max_retries - 1:
                start_op = len(completed_paths) + 1

                for i, job in enumerate(remaining_jobs):
                    job['op_num'] = start_op + i
                time.sleep(3.0)

                continue

            raise RuntimeError(
                f"MatAnyone inference failed after {max_retries} attempts "
                f"({len(remaining_jobs)} segments remaining): {exc}"
            ) from exc

    return completed_paths

def matanyone(segments: List[SegmentInfo], segments_dir: Path, mask_square: int, args: argparse.Namespace) -> List[SegmentInfo]:
    print(f"MatAnyone model: {args.matanyone_version}")

    matanyout = str(segments_dir / 'matanyone_output')
    os.makedirs(matanyout, exist_ok=True)

    mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
    total_ops = len(mask_segments) * 2

    jobs = []
    for seg in mask_segments:

        if not seg.left_mask_path or not seg.right_mask_path:
            raise RuntimeError(f'Segment [{seg.index}] missing SAM3 masks')

        seg_left_video = str(segments_dir / f'seg{seg.index:02d}_left.mp4')
        seg_right_video = str(segments_dir / f'seg{seg.index:02d}_right.mp4')

        jobs.append({

            'input_path': seg_left_video,
            'mask_path': seg.left_mask_path,
            'output_path': matanyout,
            'max_size': args.mask_height,
            'erode': args.erode,
            'warmup': args.warmup,
            'dilate': args.dilate,
            'matanyone_version': args.matanyone_version,
            'ma2_mem_every': args.ma2_mem_every,
            'ma2_max_mem_frames': args.ma2_max_mem_frames,
            'ma2_use_long_term': args.ma2_use_long_term,
            'temporal_median_window': args.temporal_median_window,
            'op_num': len(jobs) + 1,
            'total_ops': total_ops,
            'label': f'seg{seg.index:02d}_left',
            'duration': seg.end_time - seg.start_time

            })

        jobs.append({

            'input_path': seg_right_video,
            'mask_path': seg.right_mask_path,
            'output_path': matanyout,
            'max_size': args.mask_height,
            'erode': args.erode,
            'warmup': args.warmup,
            'dilate': args.dilate,
            'matanyone_version': args.matanyone_version,
            'ma2_mem_every': args.ma2_mem_every,
            'ma2_max_mem_frames': args.ma2_max_mem_frames,
            'ma2_use_long_term': args.ma2_use_long_term,
            'temporal_median_window': args.temporal_median_window,
            'op_num': len(jobs) + 1,
            'total_ops': total_ops,
            'label': f'seg{seg.index:02d}_right',
            'duration': seg.end_time - seg.start_time

            })

    completed_paths = matanyone_inference(jobs)

    if len(completed_paths) != len(jobs):
        raise RuntimeError(f'Not all jobs completed successfully. Expected {len(jobs)}, got {len(completed_paths)}')

    for seg in mask_segments:

        left_basename = os.path.splitext(os.path.basename(f'seg{seg.index:02d}_left.mp4'))[0]
        right_basename = os.path.splitext(os.path.basename(f'seg{seg.index:02d}_right.mp4'))[0]

        left_pha = os.path.join(matanyout, f'{left_basename}_pha.mp4')
        right_pha = os.path.join(matanyout, f'{right_basename}_pha.mp4')

        if not os.path.exists(left_pha) or not os.path.exists(right_pha):
            raise RuntimeError(f'Could not find generated masks for segment {seg.index}')

        stereo_output = str(segments_dir / f'seg{seg.index:02d}_stereo.mp4')

        seg.video_path = stereo_video(

            left_pha,
            right_pha,
            stereo_output

            )

    return segments

def _input_videos(input_path: str) -> List[Path]:
    path = Path(input_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f'Input path not found: {input_path}')

    if path.is_file():

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f'Unsupported video file: {path}')

        return [path]

    if not path.is_dir():
        raise RuntimeError(f'Input path is not a file or folder: {input_path}')

    videos = sorted(p.resolve() for p in path.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)

    if not videos:
        raise RuntimeError(f'No supported video files found in folder: {input_path}')

    return videos

def process_video(video_path, args: argparse.Namespace, temp_root: Path, batch_mode: bool = False, alpha_output: bool = False) -> str:

    video_path = str(Path(video_path).expanduser().resolve())
    video_name = Path(video_path).stem

    safe_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in video_name)
    temp_dir = temp_root / safe_name

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    frames_dir = temp_dir / 'frames'
    masks_dir = temp_dir / 'masks'
    segments_dir = temp_dir / 'segments'

    for d in [frames_dir, masks_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h, fps, duration, is_vfr  = info(video_path)

    video_args = argparse.Namespace(**vars(args), video=video_path)
    alpha_output = video_args.alpha

    print(f'Specs: {orig_w}x{orig_h}, {fps:.2f}fps, {format_timestamp(duration)}')
    print(f'Mask height: {video_args.mask_height}px')
    print()

    mask_square = video_args.mask_height
    overlay_mask = video_args.overlay_mask

    if overlay_mask is None:

        segments = calculate_segments(

            duration,
            video_args.segment_length

            )

        mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]

        mask_segments = extract_segments(

            video_args,
            segments,
            mask_segments,
            orig_h,
            frames_dir,
            segments_dir

            )

        mask_segments = sam3_masks(

            mask_segments, frames_dir, masks_dir, mask_square,
            video_args.prompt,
            video_args.seed_model,
            video_args.sapiens_threshold,
            video_args.gate_dilate,
            video_args=video_args,

        )

        segments = matanyone(segments, segments_dir, mask_square, video_args)

        output_mask = finalize(

            segments,
            video_name,
            str(video_path),
            video_args=video_args,

        )

        if alpha_output:
            alpha = pack_video(video_path, output_mask)

        else:
            overlay_target = str(Path(video_path).with_name(f"{video_name}_overlay.mp4"))
            overlay_video = mask_overlay(

                video_path,
                output_mask,
                overlay_target,
                background_color=video_args.overlay_color,
                video_args=video_args,

            )

            print(f'Overlay preview: {overlay_video}')
        
        print('=' * 60)
        print(f'Segments: {len(segments)} ({len(mask_segments)} masks)')
        print(f'Output: {output_mask}')
        print()

        with open(temp_dir / 'segments.txt', 'w', encoding='utf-8') as f:
            f.write(f'# {video_name}\n')

            for seg in segments:
                f.write(f'{seg.index},{seg.seg_type.value},{seg.start_time:.3f},{seg.end_time:.3f},{seg.video_path}\n')

        return output_mask

    else:

        overlay_target = str(Path(video_path).with_name(f"{video_name}_overlay.mp4"))

        overlay_video = mask_overlay(

            video_path,
            overlay_mask,
            overlay_target,
            background_color=video_args.overlay_color,
            video_args=video_args,

        )

        print(f'Overlay preview: {overlay_video}')
        print('=' * 60)

        return overlay_video

def calculate_segments(video_duration: float, max_segment_length: float = 5.0) -> List[SegmentInfo]:

    segments: List[SegmentInfo] = []
    chunk_start = 0.0
    index = 0

    while chunk_start < video_duration:
        chunk_end = min(chunk_start + max_segment_length, video_duration)

        if 0 < video_duration - chunk_end < 1.0:
            chunk_end = video_duration

        segments.append(SegmentInfo(index=index, start_time=chunk_start,
                end_time=chunk_end, seg_type=SegmentType.MASK))

        index += 1
        chunk_start = chunk_end

    return segments

def extract_segments(
    args: argparse.Namespace,
    segments: List[SegmentInfo],
    mask_segments: List[SegmentInfo],
    orig_h: int,
    frames_dir: Path,
    segments_dir: Path,
) -> List[SegmentInfo]:

    print(f'Total: {len(segments)} segments')

    for seg in segments:

        dur = seg.end_time - seg.start_time

        print(f'[{seg.index}] {seg.seg_type.value.upper():5} '
            f'{format_timestamp(seg.start_time)} → {format_timestamp(seg.end_time)} ({dur:.1f}s)')
    print()

    for i, seg in enumerate(mask_segments):

        left_frame = str(frames_dir / f'seg{seg.index:02d}_left.png')
        right_frame = str(frames_dir / f'seg{seg.index:02d}_right.png')
        seg_left_video = str(segments_dir / f'seg{seg.index:02d}_left.mp4')
        seg_right_video = str(segments_dir / f'seg{seg.index:02d}_right.mp4')

        left_frame_path, right_frame_path, _, _ = extract_segment_frames(

            stereo_video=args.video,
            start=seg.start_time,
            end=seg.end_time,
            height=orig_h,
            target_height=args.mask_height,
            left_frame_out=left_frame,
            right_frame_out=right_frame,
            left_video_out=seg_left_video,
            right_video_out=seg_right_video,
            progress_prefix=f'[{i + 1}/{len(mask_segments)}]'

            )

        seg.left_frame_path = left_frame_path
        seg.right_frame_path = right_frame_path

    return mask_segments

def finalize(segments: List[SegmentInfo], video_name: str, video_path: str, video_args=None) -> str:

    segment_vid = []

    for seg in sorted(segments, key=lambda s: s.index):

        if seg.video_path and os.path.exists(seg.video_path):
            segment_vid.append(seg.video_path)

        else:
            raise RuntimeError(f'Segment [{seg.index}] missing')

    output_dir = os.path.dirname(video_path) or '.'
    output_mask = os.path.join(output_dir, f'{video_name}_mask.mp4')
    output_mask = concat_video(segment_vid, output_mask)

    return output_mask

def main() -> int:

    start_time = time.time()
    parser = argparse.ArgumentParser(description='VR Video Masking Pipeline')
    parser.add_argument('input_path')
    parser.add_argument('--mask-height', type=int, default=1008)
    parser.add_argument('--segment-length', type=float, default=1)
    parser.add_argument('--erode', type=int, default=4)
    parser.add_argument('--dilate', type=int, default=-4)
    parser.add_argument('--prompt', type=str, default='one woman')
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--seed-model', type=str, default='sam3video', choices=['sam3', 'sam3video', 'sam31video', 'sapiens', 'hybrid'], help='Seed mask mode (sam3, sam3video, sam31video, sapiens, or hybrid)')
    parser.add_argument('--sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask')
    parser.add_argument('--gate-dilate', type=int, default=5, help='Dilate SAM3 gating in hybrid mode')
    parser.add_argument('--matanyone-version', type=str, default='v2', choices=['v1', 'v2'], help='Select MatAnyone runtime version')
    parser.add_argument('--ma2-mem-every', type=int, default=6, help='Override MatAnyone mem_every (works for v1 and v2; e.g. 2 or 3 for faster refresh)')
    parser.add_argument('--ma2-max-mem-frames', type=int, default=2, help='Override MatAnyone memory window in frames (works for v1 and v2)')
    parser.add_argument('--ma2-use-long-term', type=str, default='off', choices=['auto', 'on', 'off'], help='Override MatAnyone long-term memory mode (works for v1 and v2)')
    parser.add_argument('--temporal-median-window', type=int, default=0, help='Temporal median window for alpha cleanup. 0 disables; use odd values >= 3 (e.g. 5)')
    parser.add_argument('--no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding')
    parser.set_defaults(normalize_input=True)
    parser.add_argument('--overlay-output', type=str, default='input_path', help='Write a composited video with the mask over the original source')
    parser.add_argument('--overlay-color', type=str, default='0x00ff00', help='Background color for overlay (use 0x00ff00 for pure green)')
    parser.add_argument('--overlay-mask', type=str, default=None, help='Write a composited video with a provided mask over the original source')
    parser.add_argument('--alpha-packer', type=str, default=None, help='Run alpha packer on its own. ')
    parser.add_argument('--alpha', type=bool, default=False, help='Run alpha packer instead of overlay within pipeline. --alpha <true|false> default is False')
    parser.add_argument('--fisheye180', nargs='?', const=FISHEYE180_PIPELINE_MODE, default=None, help='Convert an SBS equirectangular input video or folder to SBS fisheye180')

    args = parser.parse_args()
    args.matanyone_version = str(args.matanyone_version).lower()

    if not (0.0 <= args.sapiens_threshold <= 1.0):
        raise ValueError('--sapiens-threshold must be between 0.0 and 1.0')
    if args.gate_dilate < 1:
        raise ValueError('--gate-dilate must be >= 1')
    if args.ma2_mem_every is not None and args.ma2_mem_every < 1:
        raise ValueError('--ma2-mem-every must be >= 1')
    if args.ma2_max_mem_frames is not None and args.ma2_max_mem_frames < 2:
        raise ValueError('--ma2-max-mem-frames must be >= 2')

    if args.ma2_use_long_term == 'auto':
        args.ma2_use_long_term = None
    else:
        args.ma2_use_long_term = (args.ma2_use_long_term == 'on')

    if args.temporal_median_window < 0:
        raise ValueError('--temporal-median-window must be >= 0')
    if args.temporal_median_window != 0 and args.temporal_median_window < 3:
        raise ValueError('--temporal-median-window must be 0 or odd >= 3')
    if args.temporal_median_window % 2 == 0 and args.temporal_median_window != 0:
        raise ValueError('--temporal-median-window must be odd (e.g. 3, 5, 7)')

    if args.alpha_packer:
        return packer(args.alpha_packer)

    if args.fisheye180 is not None:
        fisheye_mask = None if args.fisheye180 == FISHEYE180_PIPELINE_MODE else args.fisheye180
        return run_fisheye180_mode(args.input_path, fisheye_mask)

    video_paths = _input_videos(args.input_path)
    temp_root = Path('temp_pipeline')

    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    processed = []
    batch_mode = len(video_paths) > 1

    for index, video_path in enumerate(video_paths, 1):
        video_path = str(video_path)
        print(f'[{index}/{len(video_paths)}] Processing: {video_path}')

        video_args = argparse.Namespace(**vars(args), video=video_path)
        output_mask = process_video(video_path, args, temp_root, batch_mode=batch_mode)
        processed.append((video_path, output_mask))

    for video_path, output_mask in processed:

        print(f'{video_path}')
        print(f'{output_mask}')
    total_end = time.time() - start_time
    print('=' * 60)
    print(f"Total time: {total_end:.2f}s")
    return 0

if __name__ == '__main__':
    sys.exit(main())
