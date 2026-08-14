import argparse, shutil, gc, os, sys, functools, re, subprocess, torch, cv2, time, imageio, numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List
from PIL import Image
from typing import Callable, Tuple
import torch.nn.functional as F

ENCODER = 'hevc_nvenc'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.wmv'}
SAM3_REPO_ID = "sin2piusc/sam3_fta"

SAM3_MAX = 1008
SAM3_BOX_CXCYWH_NORM = (0.5, 0.5, 0.5, 0.5)
SAM3_BOX2_CXCYWH_NORM = (0.5, 0.9, 0.9, 0.18)

SAPIENS_REPO_ID = "facebook/sapiens2-matting-1b" #facebook/sapiens2-seg-1b
SAPIENS_CHECKPOINT = "sapiens2_1b_matting.safetensors"#sapiens2_1b_seg.safetensors
SAPIENS_CONFIG = "sapiens/dense/configs/normal/metasim_render_people/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py"

def encoder_args() -> list[str]:

    return [
        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp',
        '-fps_mode', 'cfr',
        '-r', '60',
        '-c:v', ENCODER,
        '-preset', 'p5',
        '-profile:v', 'main10',
        '-pix_fmt', 'p010le',
        '-g', '30',
        '-b:v', '80M',
        '-maxrate', '80M',
        '-bufsize', '160M',
        '-rc:v', 'cbr',
        '-tag:v', 'hvc1',
        '-map', '0:a?',
        '-aspect', '2:1',
        '-c:a', 'copy',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-metadata:s:v:0', 'stereo_mode=left_right',
        '-movflags', '+faststart+write_colr+use_metadata_tags',
    ]

def _ffmpeg_progress(line: str) -> str:
    parts = []
    for field in ['time=', 'elapsed=', 'speed=']:
        match = re.search(rf'{field}(\S+)', line)
        if match:
            parts.append(f"{field}{match.group(1)}")

    return (' '.join(parts) + '\033[K') if parts else line.strip()

def ffmpeg_progress(
    cmd: list[str],
    progress_prefix: str = "",
    cwd: str | None = None
) -> tuple[int, str]:

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    stderr_lines = []

    for line in process.stderr:
        stderr_lines.append(line)
        if 'frame=' in line:
            print(f"\r{progress_prefix}{_ffmpeg_progress(line)}", end='', flush=True)

    process.wait()
    print()
    return process.returncode, "".join(stderr_lines)

# @functools.lru_cache(maxsize=512)
def info(video_path: str) -> Tuple[int, int, float, float]:

    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    lines = result.stdout.strip().split('\n')
    w, h, fps_str = lines[0].split(',')
    duration = float(lines[1]) if len(lines) > 1 else 0

    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    return int(w), int(h), fps, duration

def frame_count(video_path: str) -> int:

    cmd = [
        'ffprobe', '-v', 'error',
        '-count_frames',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe frame count failed: {result.stderr}")

    raw = (result.stdout or '').strip()
    if not raw or raw == 'N/A':
        raise RuntimeError(f"ffprobe did not return nb_read_frames for {video_path}")
    try:
        count = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid frame count '{raw}' for {video_path}") from exc

    if count <= 0:
        raise RuntimeError(f"Non-positive frame count for {video_path}: {count}")
    return count

def norm_video(source_video, w = None, h = None, progress_prefix: str = "[normalize] ") -> str:

    wi, hi, fps, _ = info(source_video)
    if fps == 60:
        print()
        print(f"[normalize - skipped], FPS = {fps}")
        return source_video
    else:
        source_path = Path(source_video).expanduser().resolve()
        output_video = str(source_path.with_name(f"{source_path.stem}_normed.mp4"))

        enc = encoder_args()
        fps = 60
        if w is not None:
            wi = w
            hi = h

        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', source_video,
            '-filter_complex', f'[0:v]fps={fps},setpts=N/({fps}*TB),scale=w={wi}:h={hi}:flags=lanczos:threads=0',
            *enc,
            output_video,
        ]

        rc, stderr_text = ffmpeg_progress(cmd, progress_prefix=progress_prefix)
        if rc != 0:
            raise RuntimeError(
                "Input normalization failed.\n\nFFmpeg tail:\n"
                + ''.join(stderr_text.splitlines(True)[-40:])
            )
        if not os.path.exists(output_video):
            raise RuntimeError(f"Normalized video not created: {output_video}")

        return output_video

def resize_video(
    source_video: str,
    output_video: str,
    width: int,
    height: int,
    progress_prefix: str = "[resize] ",
) -> str:

    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or '.', exist_ok=True)

    enc = encoder_args()
    cmd = [
        'ffmpeg', '-y', '-hide_banner',
        '-i', source_video,
        '-filter_complex', f'[0:v]fps=60,setpts=N/(60*TB),scale={width}:{height}:flags=lanczos:threads=0',
        *enc,
        output_video,
    ]

    rc, stderr_text = ffmpeg_progress(cmd, progress_prefix=progress_prefix)
    if rc != 0:
        raise RuntimeError(
            "Video resize failed.\n\nFFmpeg tail:\n"
            + ''.join(stderr_text.splitlines(True)[-40:])
        )
    if not os.path.exists(output_video):
        raise RuntimeError(f"Resized video not created: {output_video}")
    return output_video

def concat_video(
    video_list: list[str],
    output_path: str,
    fps: float | None = None,
) -> str:

    BATCH_SIZE = 50
    n = len(video_list)
    enc = encoder_args()

    if n > BATCH_SIZE:
        base_path, ext = os.path.splitext(os.path.abspath(output_path))
        temp_batches = []

        for i in range(0, n, BATCH_SIZE):
            batch_files = video_list[i:i + BATCH_SIZE]
            batch_out = f"{base_path}_batch_{i//BATCH_SIZE}{ext}"
            temp_batches.append(batch_out)
            concat_video(batch_files, batch_out, fps=fps)
            return concat_video(temp_batches, output_path, fps=fps)

        for tb in temp_batches:
            if os.path.exists(tb):
                os.remove(tb)

    abs_vid = [os.path.abspath(v) for v in video_list]
    abs_output = os.path.abspath(output_path)
    common_dir = os.path.commonpath(abs_vid)

    if not os.path.isdir(common_dir):
        common_dir = os.path.dirname(common_dir)

    rel_vid = []
    for video in abs_vid:
        if not os.path.exists(video):
            raise RuntimeError(f"File missing: {video}")
        rel_vid.append(os.path.relpath(video, common_dir).replace('\\', '/'))

    rel_output = os.path.relpath(abs_output, common_dir).replace('\\', '/')
    concat_file = os.path.join(common_dir, "_concat_list.txt")

    with open(concat_file, 'w', encoding='utf-8') as f:
        for rel in rel_vid:
            f.write(f"file '{rel}'\n")

    filter_parts = []
    for i in range(n):
        filter_parts.append(f"[{i}:v]format=yuv420p[v{i}]")

    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")
    filter_complex = ";".join(filter_parts)
    filter_file = os.path.join(common_dir, "_concat_filter.txt")

    cmd_inline = [
        'ffmpeg', '-y',
        *[item for rel in rel_vid for item in ['-i', rel]],
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        *enc,
        rel_output,
    ]

    rc, stderr_text = ffmpeg_progress(cmd_inline, cwd=common_dir)
    if rc != 0 and ('The filename or extension is too long' in stderr_text or 'WinError 206' in stderr_text):
        with open(filter_file, 'w', encoding='utf-8') as f:
            f.write(filter_complex)

        cmd_script = [
            'ffmpeg', '-y',
            *[item for rel in rel_vid for item in ['-i', rel]],
            '-/filter_complex', '_concat_filter.txt',
            '-map', '[outv]',
            *enc,
            rel_output,
        ]
        rc, stderr_text = ffmpeg_progress(cmd_script, cwd=common_dir)

    if rc != 0:
        tail = ''.join(stderr_text.splitlines(True)[-60:])
        raise RuntimeError(f"FFmpeg concatenation failed.\n\nFFmpeg tail:\n{tail}")

    if not os.path.exists(abs_output):
        raise RuntimeError(f"Concat output not created: {output_path}")

    if os.path.exists(filter_file):
        os.remove(filter_file)
    if os.path.exists(concat_file):
        os.remove(concat_file)

    return output_path

def eye_frames(
    video_path: str,
    timestamps: list[float],
    output_dir: str,
    height: int,
) -> list[str]:

    output_paths = []
    eye_size = height
    crop_filter = f"crop={eye_size}:{eye_size}:0:0"

    for ts in timestamps:
        out_path = os.path.join(output_dir, f"frame_{ts:.0f}s.png")

        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-ss', str(ts),
            '-i', video_path,
            '-vf', crop_filter,
            '-frames:v', '1', '-compression_level', '1',
            out_path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.wait()
        if process.returncode == 0 and os.path.exists(out_path):
            output_paths.append(out_path)

    return output_paths

def extract_segment_frames(
    stereo_video: str,
    start: float,
    end: float,
    fps: float,
    height: int,
    target_height: int,
    left_frame_out: str,
    right_frame_out: str,
    left_video_out: str,
    right_video_out: str,
    progress_prefix: str = "",
) -> tuple[str, str, str, str]:

    enc = encoder_args()
    start_frame = round(start * fps)
    end_frame = round(end * fps)
    frames = end_frame - start_frame

    if frames <= 0:
        raise RuntimeError(f"Invalid segment: {start=} {end=} {fps=} -> {frames} frames")

    aligned_start = start_frame / fps
    keyframe_seek = max(0.0, aligned_start - 2.0)
    fine_seek = aligned_start - keyframe_seek
    seg_dur = frames / fps

    orig_eye = height
    target_eye = target_height
 
    frame_left = f"crop={orig_eye}:{orig_eye}:0:0"
    frame_right = f"crop={orig_eye}:{orig_eye}:{orig_eye}:0"

    video_left = f"crop={target_eye}:{target_eye}:0:0"
    video_right = f"crop={target_eye}:{target_eye}:{target_eye}:0"

    scale_w = target_height * 2
    scale_h = target_height

    filter_complex = (
        f"[0:v]trim=start={fine_seek}:duration={seg_dur},setpts=PTS-STARTPTS,split=2[full][toscale];"
        f"[full]split=2[fullL][fullR];"
        f"[fullL]select=eq(n\\,0),{frame_left}[frame_left];"
        f"[fullR]select=eq(n\\,0),{frame_right}[frame_right];"
        f"[toscale]format=nv12,scale={scale_w}:{scale_h}:flags=lanczos,split=2[sL][sR];"
        f"[sL]{video_left}[video_left];"
        f"[sR]{video_right}[video_right]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
    ]

    left_output_args = [
        "-map", "[video_left]", "-frames:v", str(frames),
        left_video_out,
    ]
    right_output_args = [
        "-map", "[video_right]", "-frames:v", str(frames),
        right_video_out,
    ]

    cmd.extend([
        "-ss", str(keyframe_seek),
        "-i", stereo_video,
        "-filter_complex", filter_complex,
        "-map", "[frame_left]", "-frames:v", "1", "-compression_level", "1", left_frame_out,
        "-map", "[frame_right]", "-frames:v", "1", "-compression_level", "1", right_frame_out,
        *left_output_args,
        *right_output_args,
        *enc,
    ])

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
        if "frame=" in line:
            print(f"\r{progress_prefix}{_ffmpeg_progress(line)}", end="", flush=True)

    process.wait()
    if process.returncode != 0:
        tail = "".join(stderr_lines[-60:])
        raise RuntimeError(f"Segment extraction failed.\n\nFFmpeg tail:\n{tail}")

    left_count = frame_count(left_video_out)
    right_count = frame_count(right_video_out)

    print(f"{progress_prefix}timing check: left={left_count} right={right_count} fps={fps:.6f}")
    if left_count != right_count:

        raise RuntimeError(
            "Segment frame-count mismatch detected. "
            f"left={left_count}, right={right_count}, "
            f"start={start:.6f}, end={end:.6f}, fps={fps:.6f}"
        )

    return left_frame_out, right_frame_out, left_video_out, right_video_out

def overlay_path(source_video: str, output_path: str) -> str:
    source_path = Path(source_video).expanduser()
    target_path = Path(output_path).expanduser()
    overlay_stem = source_path.stem

    if target_path.exists() and target_path.is_dir():
        return str(target_path / f"{overlay_stem}_overlay.mp4")

    if target_path.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.wmv'}:
        return str(target_path)

    return str(target_path.with_suffix('.mp4'))

def mask_overlay(
    source_video: str,
    mask_video: str,
    output_path: str,
    background_color: str = '0x00ff00',
) -> str:

    resolved_path = overlay_path(source_video, output_path)
    src_w, src_h, src_fps, src_duration = info(source_video)
    mask_w, mask_h, mask_fps, mask_duration = info(mask_video)

    if src_fps != mask_fps:
        if src_fps < 60:
            source_video = norm_video(source_video)
        if mask_fps < 60:
            mask_video = norm_video(mask_video)

    duration = src_duration
    fps = src_fps

    if (src_w, src_h) != (mask_w, mask_h):
        orig_filter = f"format=rgba,scale={src_w}:{src_h}:flags=lanczos"
        mask_filter = f"format=gray,scale={src_w}:{src_h}:flags=lanczos,lut=a=val/255"
        bg_filter = f"format=rgba,scale={src_w}:{src_h}:flags=lanczos"
    else:
        orig_filter = 'format=rgba'
        mask_filter = 'format=gray,lut=a=val/255'
        bg_filter = 'format=rgba'

    filter_complex = (
        f"[0:v]{orig_filter}[orig];"
        f"[1:v]{mask_filter}[mask_alpha];"
        f"[orig][mask_alpha]alphamerge[alphaed];"
        f"[2:v]{bg_filter}[bg];"
        f"[bg][alphaed]overlay=shortest=1:format=auto[out]"
    )

    os.makedirs(os.path.dirname(os.path.abspath(resolved_path)) or '.', exist_ok=True)

    cmd = [
        'ffmpeg', '-y', '-hide_banner',
        '-i', source_video,
        '-i', mask_video,
        '-f', 'lavfi', '-i', f'color=c={background_color}:s={src_w}x{src_h}:d={duration}:r={fps}',
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-map', '0:a?',

    ]

    cmd.extend(encoder_args())
    cmd.append(resolved_path)

    rc, stderr_text = ffmpeg_progress(cmd)
    if rc != 0:
        raise RuntimeError(f"Mask overlay failed.\n\nFFmpeg tail:\n{''.join(stderr_text.splitlines(True)[-40:])}")

    return resolved_path

def stereo_video(
    left_video: str,
    right_video: str,
    output_path: str,

) -> str:

    enc = encoder_args()

    filter_complex = "[0:v][1:v]hstack=inputs=2[out]"

    cmd = [
        'ffmpeg', '-y',
        '-i', left_video,
        '-i', right_video,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        *enc,
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Stereo stitching failed: {result.stderr}")

    return output_path

def timestamp(ts: str) -> float:

    ts = ts.strip()
    parts = ts.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def sam3_box(
    width: int,
    height: int,
    normalized_box_cxcywh: tuple[float, float, float, float] = SAM3_BOX_CXCYWH_NORM,
) -> list[float]:
    cx, cy, box_w, box_h = normalized_box_cxcywh
    abs_w = box_w * width
    abs_h = box_h * height
    abs_x = (cx - box_w / 2.0) * width
    abs_y = (cy - box_h / 2.0) * height
    return [abs_x, abs_y, abs_w, abs_h]

def sam3_box2(
    width: int,
    height: int,
    normalized_box_cxcywh: tuple[float, float, float, float] = SAM3_BOX2_CXCYWH_NORM,
) -> list[float]:
    cx, cy, box_w, box_h = normalized_box_cxcywh
    abs_w = box_w * width
    abs_h = box_h * height
    abs_x = (cx - box_w / 2.0) * width
    abs_y = (cy - box_h / 2.0) * height
    return [abs_x, abs_y, abs_w, abs_h]

def _sam3_inference(frames_dir: str, output_size: int | None = None, is_intro: bool = False, prompt: str = "one woman", show_plots = False) -> None:

    import matplotlib.pyplot as plt
    from huggingface_hub import snapshot_download
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

    repo_path = snapshot_download(repo_id=SAM3_REPO_ID, local_files_only=False)
    model_path = os.path.join(repo_path, "sam3.pth")
    model = build_sam3_image_model(load_from_HF=False, enable_inst_interactivity=False, enable_segmentation=True, compile=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    processor = Sam3Processor(model, confidence_threshold=0.4, device="cuda" if torch.cuda.is_available() else "cpu")

    folder = Path(frames_dir)
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    if not image_files:
        return

    with torch.inference_mode():
        for frame_path in sorted(image_files):
            output_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            raw = Image.open(frame_path)
            image = raw.convert("RGB")
            raw.close()
            ow, oh = (image.width, image.height)

            if oh > output_size:
                full = image
                image = full.resize((output_size, output_size), Image.Resampling.BILINEAR)
                width, height = image.width, image.height
                full.close()
            else:
                width, height = ow, oh

            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)

            inference_state = processor.set_text_prompt(state=inference_state, prompt="one woman")
            box_input_xywh = [sam3_box(image.width, image.height)]
            box_input_xywh = torch.tensor(box_input_xywh).view(-1, 4)
            box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
            norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
            inference_state = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=True)

            if show_plots:
                plot_results(image, inference_state)
                image_with_box = draw_box_on_image(image, box_input_xywh.flatten().tolist())
                plt.imshow(image_with_box)
                plt.axis("off")  
                plt.show()

            inference_state = processor.set_text_prompt(state=inference_state, prompt="one man")
            box_input_xywh = [sam3_box2(image.width, image.height)]
            box_input_xywh = torch.tensor(box_input_xywh).view(-1, 4)
            box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
            norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
            inference_state = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=False)

            if show_plots:
                plot_results(image, inference_state)
                image_with_box = draw_box_on_image(image, box_input_xywh.flatten().tolist())
                plt.imshow(image_with_box)
                plt.axis("off")  
                plt.show()

            box_input_xywh = [sam3_box(width, height), sam3_box2(width, height)]
            box_input_xywh = torch.tensor(box_input_xywh).view(-1, 4)
            box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh).view(-1,4)
            norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height).tolist()
            box_labels = [True, False]
            for box, label in zip(norm_boxes_cxcywh, box_labels):
                inference_state = processor.add_geometric_prompt(state=inference_state, box=box, label=label)

            if show_plots:
                image_with_box = image.copy()
                for i in range(len(box_input_xywh)):
                    if box_labels[i] == 1:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    image_with_box = draw_box_on_image(image_with_box, box_input_xywh[i], color)
                plt.imshow(image_with_box)
                plt.axis("off") 
                plt.show()

            masks = inference_state["masks"]
            scores = inference_state["scores"]

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            else:
                scores = np.asarray(scores)

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            else:
                masks = np.asarray(masks)

            if len(masks) == 0 or scores.size == 0:
                best_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                print(f"No SAM3 masks/scores for {frame_path.name}; writing empty mask.")
            else:
                best_idx = int(np.argmax(scores))
                best_mask = masks[best_idx]

                if len(best_mask.shape) == 3:
                    best_mask = best_mask[0]

                best_mask = (best_mask * 255).astype(np.uint8)
                print("Confidence:", scores[best_idx])

            del inference_state
            image.close()

            mask_image = Image.fromarray(best_mask)
            mask_image.save(output_path)

            gc.collect()
            torch.cuda.empty_cache()

    del processor, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

def sam3_batch(
    frames_dir: str,
    output_size: int | None = None,
    prompt: str = "woman",
    quiet: bool = False) -> None:
    _sam3_inference(frames_dir, output_size=output_size, is_intro=quiet, prompt=prompt)

def _estimate_alpha(image_bgr: np.ndarray, model) -> np.ndarray:

    h0, w0 = image_bgr.shape[:2]
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs = data["inputs"]

    with torch.no_grad():
        outputs = model(inputs)

    outputs = F.interpolate(
        outputs,
        size=(h0, w0),
        mode="bicubic",
        align_corners=False,
    )
    outputs = outputs.squeeze(0).float().cpu().numpy()
    if outputs.size == 3:
        alpha = outputs[2].clip(0.0, 1.0)
    else:
        alpha = outputs[3].clip(0.0, 1.0)
    return alpha

def _sapiens_inference(frames_dir: str, output_size: int | None = None, threshold: float = 0.5) -> None:
    from huggingface_hub import hf_hub_download
    from sapiens.dense.src.models.core.matting_estimator import MattingEstimator
    from sapiens.dense.src.models.init_model import init_model

    _ = MattingEstimator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = hf_hub_download(repo_id=SAPIENS_REPO_ID, filename=SAPIENS_CHECKPOINT)
    model = init_model(SAPIENS_CONFIG, ckpt, device=device)

    folder = Path(frames_dir)
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    image_files = [f for f in image_files if "_mask" not in f.stem]
    if not image_files:
        return

    with torch.inference_mode():
        for frame_path in sorted(image_files):
            output_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            raw = Image.open(frame_path)
            image = raw.convert("RGB")
            raw.close()

            if output_size and image.height > output_size:
                full = image
                image = full.resize((output_size, output_size), Image.Resampling.BILINEAR)
                full.close()

            image_rgb = np.array(image)
            image_bgr = image_rgb[:, :, ::-1]

            alpha = _estimate_alpha(image_bgr, model)
            mask = (alpha >= threshold).astype(np.uint8) * 255

            Image.fromarray(mask, mode='L').save(output_path)
            image.close()
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

def _sam_sapiens(frames_dir: str, output_size: int | None = None, prompt: str = "one woman", threshold: float = 0.5, gate_dilate: int = 5) -> None:
    sam3_batch(frames_dir, output_size=output_size, prompt=prompt)

    folder = Path(frames_dir)
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
    image_files = [f for f in image_files if "_mask" not in f.stem]
    if not image_files:
        return

    from huggingface_hub import hf_hub_download
    from sapiens.dense.src.models.core.matting_estimator import MattingEstimator
    from sapiens.dense.src.models.init_model import init_model

    _ = MattingEstimator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = hf_hub_download(repo_id=SAPIENS_REPO_ID, filename=SAPIENS_CHECKPOINT)
    model = init_model(SAPIENS_CONFIG, ckpt, device=device)

    k = max(1, int(gate_dilate))
    kernel = np.ones((k, k), np.uint8)

    with torch.inference_mode():
        for frame_path in sorted(image_files):
            sam3_mask_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            if not sam3_mask_path.exists():
                continue

            raw = Image.open(frame_path)
            image = raw.convert("RGB")
            raw.close()

            if output_size and image.height > output_size:
                full = image
                image = full.resize((output_size, output_size), Image.Resampling.BILINEAR)
                full.close()

            image_rgb = np.array(image)
            image_bgr = image_rgb[:, :, ::-1]
            alpha = _estimate_alpha(image_bgr, model)
            sapiens_mask = (alpha >= threshold).astype(np.uint8) * 255

            sam3_mask = np.array(Image.open(sam3_mask_path).convert('L'))
            if sam3_mask.shape != sapiens_mask.shape:
                sam3_mask = cv2.resize(sam3_mask, (sapiens_mask.shape[1], sapiens_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            sam3_gate = (sam3_mask > 127).astype(np.uint8)
            sam3_gate = cv2.dilate(sam3_gate, kernel, iterations=1)
            hybrid_mask = np.where(sam3_gate > 0, sapiens_mask, 0).astype(np.uint8)

            Image.fromarray(hybrid_mask, mode='L').save(sam3_mask_path)
            image.close()
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

def seed_mask_batch(
    frames_dir: str,
    output_size: int | None = None,
    prompt: str = "one woman",
    seed_model: str = "sam3",
    sapiens_threshold: float = 0.5,
    gate_dilate: int = 5) -> None:

    seed_model = (seed_model or "sam3").lower()
    if seed_model == "sam3":
        sam3_batch(frames_dir, output_size=output_size, prompt=prompt)
    elif seed_model == "sapiens":
        _sapiens_inference(frames_dir, output_size=output_size, threshold=sapiens_threshold)
    elif seed_model == "hybrid":
        _sam_sapiens(
            frames_dir,
            output_size=output_size,
            prompt=prompt,
            threshold=sapiens_threshold,
            gate_dilate=gate_dilate)
    else:
        raise ValueError(f"Unsupported seed model: {seed_model}")

_matanyone_is_first_status = True
_matanyone_tqdm_lines = 1

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

@functools.lru_cache(maxsize=1)
def _load_matanyone_runtime():
    matanyone_root = Path(__file__).resolve().parent / 'MatAnyone2'
    matanyone_root_str = str(matanyone_root)
    if matanyone_root_str not in sys.path:
        sys.path.insert(0, matanyone_root_str)

    from MatAnyone2.matanyone2.utils.inference_utils import gen_dilate, gen_erosion, read_frame_from_videos
    from MatAnyone2.matanyone2.inference.inference_core import InferenceCore
    from MatAnyone2.matanyone2.utils.get_default_model import get_matanyone2_model
    from MatAnyone2.matanyone2.utils.device import get_default_device

    device = get_default_device()
    pretrain_model_url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
    model_dir = matanyone_root / 'pretrained_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / 'matanyone2.pth'

    if not ckpt_path.exists():
        sys.stderr.write(" Downloading MatAnyone2 weights...\n")
        sys.stderr.flush()
        torch.hub.download_url_to_file(pretrain_model_url, str(ckpt_path), progress=False)

    model = get_matanyone2_model(str(ckpt_path), device)
    return model, device, InferenceCore, read_frame_from_videos, gen_dilate, gen_erosion

@torch.inference_mode()
def _matanyone_process_segment(matanyone2, device, inference_core_cls, read_frame_from_videos_fn, gen_dilate_fn, gen_erosion_fn, job: dict) -> str:
    import tqdm

    n_warmup = int(job.get('warmup', 6))
    input_path = job['input_path']
    mask_path = job['mask_path']
    output_path = job['output_path']
    r_erode = int(job.get('erode', 0))
    r_dilate = int(job.get('dilate', 0))
    suffix = job.get('suffix', '')

    processor = inference_core_cls(matanyone2, cfg=matanyone2.cfg)
    vframes, _, length, video_name = read_frame_from_videos_fn(input_path)
    vframes = vframes.float()

    repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    length += n_warmup

    os.makedirs(output_path, exist_ok=True)
    if suffix:
        video_name = f'{video_name}_{suffix}'

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    if r_dilate > 0:
        mask = gen_dilate_fn(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion_fn(mask, r_erode, r_erode)
    mask = torch.from_numpy(mask).float().to(device)

    objects = [1]
    phas = []
    for ti in tqdm.tqdm(range(length)):
        image = (vframes[ti] / 255.).float().to(device)

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)
            output_prob = processor.step(image, first_frame_pred=True)
        elif ti <= n_warmup:
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            output_prob = processor.step(image)

        mask = processor.output_prob_to_mask(output_prob)

        if ti > (n_warmup - 1):
            pha = mask.unsqueeze(2).cpu().numpy()
            pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)
            phas.append(pha)

    output_file = os.path.join(output_path, f'{video_name}_pha.mp4')
    imageio.mimwrite(output_file, np.array(phas), fps=60, quality=10)

    del processor, vframes, phas, mask
    torch.cuda.empty_cache()
    gc.collect()
    return output_file

def matanyone_inference(jobs: list[dict], on_segment_done: Callable[[str], None] = None) -> list[str]:
    global _matanyone_is_first_status

    max_retries = 8
    remaining_jobs = list(jobs)
    completed_paths = []

    matanyone2, device, inference_core_cls, read_frame_from_videos_fn, gen_dilate_fn, gen_erosion_fn = _load_matanyone_runtime()

    for attempt in range(max_retries):
        batch_completed = []
        _matanyone_is_first_status = True

        try:
            for job in remaining_jobs:
                _update_status(job['op_num'], job['total_ops'], job['label'], job['duration'])
                output_file = _matanyone_process_segment(
                    matanyone2,
                    device,
                    inference_core_cls,
                    read_frame_from_videos_fn,
                    gen_dilate_fn,
                    gen_erosion_fn,
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

    videos = sorted(
        p.resolve() for p in path.rglob('*')
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)

    if not videos:
        raise RuntimeError(f'No supported video files found in folder: {input_path}')
    return videos

def process_video(video_path, original_video_path, args: argparse.Namespace, temp_root: Path, batch_mode: bool = False) -> str:
    original_video_path = str(Path(original_video_path).expanduser().resolve())
    original_video_name = Path(original_video_path).stem
    safe_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in original_video_name)
    temp_dir = temp_root / safe_name

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    frames_dir = temp_dir / 'frames'
    masks_dir = temp_dir / 'masks'
    segments_dir = temp_dir / 'segments'

    for d in [frames_dir, masks_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    video_path = str(Path(video_path).expanduser().resolve())

    orig_w, orig_h, fps, duration = info(video_path)
    video_args = argparse.Namespace(**vars(args), video=video_path)
    print(f'Specs: {orig_w}x{orig_h}, {fps:.2f}fps, {format_timestamp(duration)}')
    print(f'Mask height: {video_args.mask_height}px')
    print()

    mask_square = video_args.mask_height
    segments = calculate_segments(duration, video_args.segment_length)
    mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
    mask_segments = extract_segments(video_args, segments, mask_segments, fps, orig_h, frames_dir, segments_dir)
    mask_segments = sam3_masks(mask_segments, frames_dir, masks_dir, mask_square,
        video_args.prompt,
        video_args.seed_model,
        video_args.sapiens_threshold,
        video_args.gate_dilate,
    )
    segments = matanyone(segments, segments_dir, mask_square, video_args)

    output_mask = finalize(
        segments,
        original_video_name,
        str(video_path),
        video_path,
        fps=fps,
        target_size=(orig_w, orig_h),
    )

    overlay_target = str(Path(original_video_path).with_name(f"{original_video_name}_overlay.mp4"))
    overlay_video = mask_overlay(
        video_path,
        output_mask,
        overlay_target,
        background_color=video_args.overlay_color,
    )
    print(f'Overlay preview: {overlay_video}')
    print('=' * 60)
    print(f'Segments: {len(segments)} ({len(mask_segments)} masks)')
    print(f'Output: {output_mask}')
    print()

    with open(temp_dir / 'segments.txt', 'w', encoding='utf-8') as f:
        f.write(f'# {original_video_name}\n')
        for seg in segments:
            f.write(f'{seg.index},{seg.seg_type.value},{seg.start_time:.3f},{seg.end_time:.3f},{seg.video_path}\n')

    return output_mask

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
    fps: float,
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
            fps=fps,
            height=orig_h,
            target_height=args.mask_height,
            left_frame_out=left_frame,
            right_frame_out=right_frame,
            left_video_out=seg_left_video,
            right_video_out=seg_right_video,
            progress_prefix=f'[{i + 1}/{len(mask_segments)}] ',
        )
        seg.left_frame_path = left_frame_path
        seg.right_frame_path = right_frame_path

    return mask_segments

def sam3_masks(
    mask_segments: List[SegmentInfo],
    frames_dir: Path,
    masks_dir: Path,
    mask_square: int,
    prompt: str | None,
    seed_model: str,
    sapiens_threshold: float,
    gate_dilate: int,
) -> List[SegmentInfo]:

    print(f"Seed model: {seed_model}")
    seed_mask_batch(
        str(frames_dir),
        output_size=mask_square,
        prompt=prompt,
        seed_model=seed_model,
        sapiens_threshold=sapiens_threshold,
        gate_dilate=gate_dilate,
    )

    for seg in mask_segments:
        if seg.left_frame_path:
            base = os.path.splitext(os.path.basename(seg.left_frame_path))[0]
            mask_src = frames_dir / f'{base}_mask.png'
            if mask_src.exists():
                final_mask_path = str(masks_dir / f'seg{seg.index:02d}_left_mask.png')
                shutil.move(str(mask_src), final_mask_path)
                seg.left_mask_path = final_mask_path

        if seg.right_frame_path:
            base = os.path.splitext(os.path.basename(seg.right_frame_path))[0]
            mask_src = frames_dir / f'{base}_mask.png'
            if mask_src.exists():
                final_mask_path = str(masks_dir / f'seg{seg.index:02d}_right_mask.png')
                shutil.move(str(mask_src), final_mask_path)
                seg.right_mask_path = final_mask_path

    return mask_segments

def matanyone(segments: List[SegmentInfo], segments_dir: Path, mask_square: int, args: argparse.Namespace) -> List[SegmentInfo]:

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
            'max_size': mask_square,
            'erode': args.erode,
            'dilate': args.dilate,
            'op_num': len(jobs) + 1,
            'total_ops': total_ops,
            'label': f'seg{seg.index:02d}_left',
            'duration': seg.end_time - seg.start_time})

        jobs.append({
            'input_path': seg_right_video,
            'mask_path': seg.right_mask_path,
            'output_path': matanyout,
            'max_size': mask_square,
            'erode': args.erode,
            'dilate': args.dilate,
            'op_num': len(jobs) + 1,
            'total_ops': total_ops,
            'label': f'seg{seg.index:02d}_right',
            'duration': seg.end_time - seg.start_time})

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
            stereo_output)

    return segments

def finalize(segments: List[SegmentInfo], video_name: str, video_path: str, fps: float = 60) -> str:

    segment_vid = []
    for seg in sorted(segments, key=lambda s: s.index):
        if seg.video_path and os.path.exists(seg.video_path):
            segment_vid.append(seg.video_path)
        else:
            raise RuntimeError(f'Segment [{seg.index}] missing')

    output_dir = os.path.dirname(video_path) or '.'
    output_mask = os.path.join(output_dir, f'{video_name}_mask.mp4')
    output_mask = concat_video(segment_vid, output_mask, fps=fps)
    return output_mask

def main() -> int:

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Minimal VR Video Masking Pipeline')
    parser.add_argument('input_path')
    parser.add_argument('--mask-height', type=int, default=1008)
    parser.add_argument('--segment-length', type=float, default=10)
    parser.add_argument('--erode', type=int, default=0)
    parser.add_argument('--dilate', type=int, default=0)
    parser.add_argument('--prompt', type=str, default='one woman')
    parser.add_argument('--seed-model', type=str, default='sam3', choices=['sam3', 'sapiens', 'hybrid'], help='Seed mask guy (sam3, sapiens, or hybrid)')
    parser.add_argument('--sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask')
    parser.add_argument('--gate-dilate', type=int, default=5, help='Dilate SAM3 gating in hybrid mode')
    parser.add_argument('--no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding')
    parser.set_defaults(normalize_input=True)
    parser.add_argument('--overlay-output', type=str, default=None, help='Write a composited video with the mask over the original source')
    parser.add_argument('--overlay-color', type=str, default='0x00ff00', help='Background color for the optional overlay preview (use 0x00ff00 for pure green)')
    args = parser.parse_args()

    if not (0.0 <= args.sapiens_threshold <= 1.0):
        raise ValueError('--sapiens-threshold must be between 0.0 and 1.0')
    if args.gate_dilate < 1:
        raise ValueError('--gate-dilate must be >= 1')

    video_paths = _input_videos(args.input_path)
    temp_root = Path('temp_pipeline')
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    processed = []
    batch_mode = len(video_paths) > 1
    for index, video_path in enumerate(video_paths, 1):
        original_video_path = str(video_path)
        print(f'[{index}/{len(video_paths)}] Processing: {original_video_path}')

        if args.normalize_input:
            processing_video_path = norm_video(original_video_path)
        else:
            processing_video_path = original_video_path

        output_mask = process_video(processing_video_path, original_video_path, args, temp_root, batch_mode=batch_mode)
        processed.append((original_video_path, processing_video_path, output_mask))

    for original_video_path, processing_video_path, output_mask in processed:
        print(f'{original_video_path} -> {processing_video_path}')
        print(f'{original_video_path} -> {output_mask}')

    total_end = time.time() - start_time
    print('=' * 60)
    print(f"Total time: {total_end:.2f}s")
    return 0

if __name__ == '__main__':
    sys.exit(main())
