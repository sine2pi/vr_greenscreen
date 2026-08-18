import argparse, shutil, gc, os, sys, functools, re, subprocess, time, torch, cv2, imageio, numpy as np, glob, tqdm, random, av
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image
from typing import Callable, Tuple, List
import torch.nn.functional as F
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from omegaconf import open_dict
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results, load_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
_setup_tf32()

ENCODER = 'hevc_nvenc'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

SAM3_REPO_ID = "sin2piusc/sam3_fta"
SAM3_MAX = 1008
SAM3_BOX_CXCYWH_NORM = (0.5, 0.5, 0.5, 0.5)
SAM3_BOX2_CXCYWH_NORM = (0.5, 0.9, 0.9, 0.18)
BATCH_SIZE = 50 # for concatenation only

SAPIENS_REPO_ID = "facebook/sapiens2-matting-1b"
SAPIENS_CHECKPOINT = "sapiens2_1b_matting.safetensors"
SAPIENS_CONFIG = "assets/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py"

MATANYONE_V1 = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
MATANYONE_V2 = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"

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

def ffmpeg_progress(cmd: list[str], progress_prefix: str = "", cwd: str | None = None) -> tuple[int, str]:

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)

    stderr_lines = []

    for line in process.stderr:
        stderr_lines.append(line)

        if 'frame=' in line:
            print(f"\r{progress_prefix}{_ffmpeg_progress(line)}", end='', flush=True)

    process.wait()
    print()

    return process.returncode, "".join(stderr_lines)

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
    count = int(raw)

    return count

def norm_video(source_video, w = None, h = None, progress_prefix: str = "[normalize] ") -> str:

    wi, hi, fps, _ = info(source_video)
    enc = encoder_args()

    if fps == 60:
        print()
        print(f"[normalize - skipped], FPS = {fps}")
        return source_video

    else:
        source_path = Path(source_video).expanduser().resolve()
        output_video = str(source_path.with_name(f"{source_path.stem}_normed.mp4"))

        fps = 60

        if w is not None:
            wi = w
            hi = h

        cmd = [

            'ffmpeg', '-y', '-hwaccel', 'auto',
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

def resize_video(source_video: str, output_video: str, width: int, height: int, progress_prefix: str = "[resize] ") -> str:

    enc = encoder_args()

    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or '.', exist_ok=True)

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'auto',
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

def concat_video(video_list: list[str], output_path: str, fps: float | None = None) -> str:

    BATCH_SIZE = 50
    n = len(video_list)
    enc = encoder_args()

    if n > BATCH_SIZE:

        base_path, ext = os.path.splitext(os.path.abspath(output_path))
        temp_batches = []

        try:
            for i in range(0, n, BATCH_SIZE):
                batch_files = video_list[i:i + BATCH_SIZE]
                batch_out = f"{base_path}_batch_{i//BATCH_SIZE}{ext}"
                temp_batches.append(batch_out)
                concat_video(batch_files, batch_out, fps=fps)

            return concat_video(temp_batches, output_path, fps=fps)
        finally:
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

        'ffmpeg', '-y', '-hwaccel', 'auto',
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

            'ffmpeg', '-y', '-hwaccel', 'auto',
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

def eye_frames(video_path: str, timestamps: list[float], output_dir: str, height: int) -> list[str]:

    output_paths = []

    eye_size = height
    crop_filter = f"crop={eye_size}:{eye_size}:0:0"

    for ts in timestamps:
        out_path = os.path.join(output_dir, f"frame_{ts:.0f}s.png")

        cmd = [

            'ffmpeg', '-y', '-hwaccel', 'auto',
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
    height: int,
    target_height: int,
    left_frame_out: str,
    right_frame_out: str,
    left_video_out: str,
    right_video_out: str,
    progress_prefix: str = "",
) -> tuple[str, str, str, str]:

    fps = info(stereo_video)[2]

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

def mask_overlay(source_video: str, mask_video: str, output_path: str, background_color: str = '0x00ff00') -> str:

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

        'ffmpeg', '-y', '-hwaccel', 'auto',
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

def stereo_video(left_video: str, right_video: str, output_path: str) -> str:

    enc = encoder_args()

    filter_complex = "[0:v][1:v]hstack=inputs=2[out]"

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'auto',
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

def sam3_box(width: int, height: int, normalized_box_cxcywh: tuple[float, float, float, float] = SAM3_BOX_CXCYWH_NORM) -> list[float]:

    cx, cy, box_w, box_h = normalized_box_cxcywh
    abs_w = box_w * width
    abs_h = box_h * height
    abs_x = (cx - box_w / 2.0) * width
    abs_y = (cy - box_h / 2.0) * height
    box = [abs_x, abs_y, abs_w, abs_h]

    return box

def sam3_box2(width: int, height: int, normalized_box_cxcywh: tuple[float, float, float, float] = SAM3_BOX2_CXCYWH_NORM) -> list[float]:

    cx, cy, box_w, box_h = normalized_box_cxcywh
    abs_w = box_w * width
    abs_h = box_h * height
    abs_x = (cx - box_w / 2.0) * width
    abs_y = (cy - box_h / 2.0) * height
    box = [abs_x, abs_y, abs_w, abs_h]

    return box

def build_sam3_video_predictor(*model_args, checkpoint_path=None, gpus_to_use=None, is_sbs=False, max_num_objects=1, num_obj_for_compile=1, strict_state_dict_loading=False, **model_kwargs):
    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
    return Sam3VideoPredictorMultiGPU(*model_args, checkpoint_path=checkpoint_path, gpus_to_use=gpus_to_use, is_sbs=is_sbs, max_num_objects= max_num_objects, num_obj_for_compile=num_obj_for_compile, strict_state_dict_loading=strict_state_dict_loading, **model_kwargs)

class sam3_video_inference:

    def __init__(self, video_path: str, prompt: str = "one woman", sam31=True):

        self.video_path = video_path
        self.prompt = prompt

        if sam31:

            from sam3.model_builder import build_sam3_multiplex_video_predictor

            self.predictor = build_sam3_multiplex_video_predictor(

                bpe_path=None,
                max_num_objects = 1,
                multiplex_count = 16,
                use_fa3 = False,
                use_rope_real = False,
                compile = False,
                warm_up = False,
                default_output_prob_thresh  = 0.5,
                async_loading_frames  = False,
                num_obj_for_compile=1
                )

        else:

            self.predictor = build_sam3_video_predictor(

                bpe_path=None,
                gpus_to_use = None,
                has_presence_token = False,
                geo_encoder_use_img_cross_attn = False,
                strict_state_dict_loading = False,
                async_loading_frames = True,
                video_loader_type = "cv2",
                apply_temporal_disambiguation = True,
                compile = False,
                is_sbs=None,
                max_num_objects=1,
                num_obj_for_compile=1,
                use_fa3 = False

                )

    def propagate_in_video(self, predictor=None, session_id=None):

        predictor=self.predictor
        outputs_per_frame = {}

        for response in predictor.handle_stream_request(

            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",

            )
        ):
            outputs_per_frame[response["frame_idx"]] = response["outputs"]

        return outputs_per_frame

    def abs_to_rel_coords(self, coords=None, IMG_WIDTH=None, IMG_HEIGHT=None, coord_type="point"):

        if coord_type == "point":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]

        elif coord_type == "box":

            return [

                [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
                for x, y, w, h in coords
            ]

        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    def ivebeenframed(self, video_path=None):

        if video_path is None:
            video_path = self.video_path

        if isinstance(video_path, str) and video_path.endswith(".mp4"):

            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()

                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

        else:
            frames = glob.glob(os.path.join(video_path, "*.jpg"))

            try:
                frames.sort(
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                )

            except ValueError:
                print(
                    f'frame names are not in "<frame_idx>.jpg" format: {frames[:5]=}, '
                    f"falling back to lexicographic sort."
                )
                frames.sort()

        self.sample_img = Image.fromarray(load_frame(frames[0]))

    def track(self, remove=False, refine_object_3=False, refine_object=False):

        predictor, video_path, prompt  = self.predictor, self.video_path, self.prompt
        IMG_WIDTH, IMG_HEIGHT = self.sample_img.size

        response = predictor.handle_request(

            request=dict(
                type="start_session",
                resource_path=video_path,

            )
        )

        session_id = response["session_id"]

        _ = predictor.handle_request(

            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        prompt_text = prompt
        frame_idx = 0

        response = predictor.handle_request(

            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_idx=frame_idx,
                text=prompt_text,

            )
        )

        out = response["outputs"]
        outputs = self.propagate_in_video(predictor, session_id)

        if remove:

            obj_id = 2
            response = predictor.handle_request(

                request=dict(
                    type="remove_object",
                    session_id=session_id,
                    obj_id=obj_id,
                )
            )

            frame_idx = 0
            obj_id = 2
            points_abs = np.array(
                [
                    [740, 450],
                    [760, 630],
                    [840, 640],
                    [760, 550],
                ]
            )

            points_tensor = torch.tensor(self.abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"), dtype=torch.float32)
            points_labels_tensor = torch.tensor(np.array([1, 0, 0, 1]), dtype=torch.int32)

            response = predictor.handle_request(

                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_idx=frame_idx,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    obj_id=obj_id,
                )
            )

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

        if refine_object:

            frame_idx = 0
            obj_id = 2
            points_abs = np.array(

                [
                    [760, 550],
                ]
            )

            points_tensor = torch.tensor(self.abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"), dtype=torch.float32)
            points_labels_tensor = torch.tensor(np.array([1]), dtype=torch.int32)

            response = predictor.handle_request(

                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_idx=frame_idx,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    obj_id=obj_id,
                )
            )

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

            if refine_object_3:

                frame_idx = 0
                obj_id = 3
                points_abs = np.array(

                    [
                        [800, 135],
                        [800, 180],
                    ]
                )

                labels = np.array([1, 0])
                points_tensor = torch.tensor(self.abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"), dtype=torch.float32)
                points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

                response = predictor.handle_request(

                    request=dict(

                        type="add_prompt",
                        session_id=session_id,
                        frame_idx=frame_idx,
                        points=points_tensor,
                        point_labels=points_labels_tensor,
                        obj_id=obj_id,
                    )
                )

                out = response["outputs"]
                outputs = self.propagate_in_video(predictor, session_id)

            else:
                frame_idx = 0
                obj_id = 2
                points_abs = np.array(

                    [
                        [740, 450],
                        [760, 630],
                        [840, 640],
                        [760, 550],
                    ]
                )

                labels = np.array([1, 0, 0, 1])
                points_tensor = torch.tensor(self.abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"), dtype=torch.float32)
                points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

                response = predictor.handle_request(

                    request=dict(

                        type="add_prompt",
                        session_id=session_id,
                        frame_idx=frame_idx,
                        points=points_tensor,
                        point_labels=points_labels_tensor,
                        obj_id=obj_id,
                    )
                )

                out = response["outputs"]
                outputs = self.propagate_in_video(predictor, session_id)

        _ = predictor.handle_request(

            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        predictor.shutdown()

        return outputs

def _extract_sam3_video_mask(outputs, out_h: int, out_w: int) -> np.ndarray | None:

    if not isinstance(outputs, dict):
        return None

    masks = outputs.get("out_binary_masks", None)

    if masks is None:
        return None

    masks = np.asarray(masks)

    if masks.size == 0:
        return None

    if masks.ndim == 2:
        best = masks.astype(np.float32)

    elif masks.ndim == 3:

        if masks.shape[0] == 1:
            best = masks[0].astype(np.float32)

        else:
            probs = outputs.get("out_probs", None)

            if probs is not None:
                probs = np.asarray(probs)

                if probs.size == masks.shape[0]:
                    best = masks[int(np.argmax(probs))].astype(np.float32)

                else:
                    best = masks[0].astype(np.float32)

            else:
                best = masks[0].astype(np.float32)

    else:
        return None

    if best.shape != (out_h, out_w):
        best = cv2.resize(best.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    best = np.clip((best - 0.5) * 3.0 + 0.5, 0.0, 1.0)
    return best

def _sam3_video_inference(frames_dir: str, output_size: int | None = None, prompt: str = "one woman", sam31=False) -> None:

    folder = Path(frames_dir)
    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    if not image_files:
        return

    seq_dir = folder / "_sam3video_seq"

    if seq_dir.exists():
        shutil.rmtree(seq_dir)

    seq_dir.mkdir(parents=True, exist_ok=True)

    frame_shapes: list[tuple[int, int]] = []
    output_paths: list[Path] = []

    for i, frame_path in enumerate(image_files):
        out_path = frame_path.parent / f"{frame_path.stem}_mask.png"
        output_paths.append(out_path)

        raw = Image.open(frame_path)
        image = raw.convert("RGB")
        raw.close()

        if output_size and image.height > output_size:

            full = image
            image = full.resize((output_size, output_size), Image.Resampling.BILINEAR)
            full.close()

        frame_shapes.append((image.height, image.width))
        image.save(seq_dir / f"{i:06d}.jpg", format="JPEG", quality=100)
        image.close()

    tracker = sam3_video_inference(video_path=str(seq_dir), prompt=prompt, sam31=sam31)
    tracker.ivebeenframed()
    outputs_per_frame = tracker.track()

    missing = 0

    for i, out_path in enumerate(output_paths):
        out_h, out_w = frame_shapes[i]
        outputs = outputs_per_frame.get(i)

        soft = _extract_sam3_video_mask(outputs, out_h, out_w)

        if soft is None:
            missing += 1
            soft = np.zeros((out_h, out_w), dtype=np.float32)

        hard = (soft >= 0.5).astype(np.uint8) * 255
        Image.fromarray(hard, mode='L').save(out_path)

    if missing > 0:
        print(f"SAM3 video: missing {missing}/{len(output_paths)} frames; wrote empty masks for those frames")

    if seq_dir.exists():
        shutil.rmtree(seq_dir)

def sam3_video_batch(frames_dir: str, output_size: int | None = None, prompt: str = "one woman", sam31=False) -> None:

    _sam3_video_inference(frames_dir, output_size=output_size, prompt=prompt, sam31=sam31)

def _fill_soft_mask_gaps(
    soft_masks: list[np.ndarray],
    valid_flags: list[bool],
    max_interp_gap: int = 6,
) -> tuple[list[np.ndarray], int]:

    if not soft_masks:
        return soft_masks, 0

    valid_idx = [i for i, ok in enumerate(valid_flags) if ok]

    if not valid_idx:
        return [m.copy() for m in soft_masks], 0

    filled = [m.copy() for m in soft_masks]
    filled_count = 0

    for i in range(len(filled)):

        if valid_flags[i]:
            continue

        prev_i = next((j for j in reversed(valid_idx) if j < i), None)
        next_i = next((j for j in valid_idx if j > i), None)

        if prev_i is not None and next_i is not None:
            gap = next_i - prev_i - 1

            if gap <= max_interp_gap:
                alpha = (i - prev_i) / (next_i - prev_i)
                filled[i] = (1.0 - alpha) * filled[prev_i] + alpha * filled[next_i]

            else:
                filled[i] = filled[prev_i].copy() if (i - prev_i) <= (next_i - i) else filled[next_i].copy()

            filled_count += 1

        elif prev_i is not None:
            filled[i] = filled[prev_i].copy()
            filled_count += 1

        elif next_i is not None:
            filled[i] = filled[next_i].copy()
            filled_count += 1

    return filled, filled_count

def _sam3_inference(frames_dir: str, output_size: int | None = None, prompt: str = "one woman", show_plots = False) -> None:

    folder = Path(frames_dir)
    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    repo_path = snapshot_download(repo_id=SAM3_REPO_ID, local_files_only=False)
    model_path = os.path.join(repo_path, "sam3.pth")

    model = build_sam3_image_model(load_from_HF=False, enable_inst_interactivity=False, enable_segmentation=True, compile=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])

    processor = Sam3Processor(model, confidence_threshold=0.4, device="cuda" if torch.cuda.is_available() else "cpu")

    if not image_files:
        return

    mask_paths: list[Path] = []
    soft_masks: list[np.ndarray] = []
    valid_flags: list[bool] = []

    min_valid_pixels = 512

    with torch.inference_mode():
        for frame_path in image_files:
            output_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            mask_paths.append(output_path)

            raw = Image.open(frame_path)
            image = raw.convert("RGB")
            raw.close()

            ow, oh = image.width, image.height

            if output_size and oh > output_size:
                full = image
                image = full.resize((output_size, output_size), Image.Resampling.BILINEAR)
                width, height = image.width, image.height
                full.close()

            else:
                width, height = ow, oh

            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)

            inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
            box_input_xywh = torch.tensor([sam3_box(image.width, image.height)]).view(-1, 4)

            norm_box_cxcywh = normalize_bbox(box_xywh_to_cxcywh(box_input_xywh), width, height).flatten().tolist()
            inference_state = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=True)

            if show_plots:
                plot_results(image, inference_state)
                plt.imshow(draw_box_on_image(image, box_input_xywh.flatten().tolist()))
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
                best_soft = np.zeros((image.height, image.width), dtype=np.float32)

                print(f"No SAM3 masks/scores for {frame_path.name}; marking as missing for temporal fill")

            else:
                best_idx = int(np.argmax(scores))
                best_soft = masks[best_idx]

                if len(best_soft.shape) == 3:
                    best_soft = best_soft[0]

                best_soft = np.asarray(best_soft, dtype=np.float32)
                best_soft = np.clip((best_soft - 0.5) * 3.0 + 0.5, 0.0, 1.0)

                print("Confidence:", scores[best_idx])

            soft_masks.append(best_soft)
            valid_flags.append(int(np.count_nonzero(best_soft >= 0.5)) >= min_valid_pixels)

            del inference_state
            image.close()

    filled_soft_masks, filled_count = _fill_soft_mask_gaps(soft_masks, valid_flags, max_interp_gap=6)
    if filled_count > 0:
        print(f"Filled {filled_count} missing/weak SAM3 masks using temporal soft-mask interpolation")

    for out_path, soft_mask in zip(mask_paths, filled_soft_masks):
        hard_mask = (soft_mask >= 0.5).astype(np.uint8) * 255
        Image.fromarray(hard_mask, mode='L').save(out_path)

    del processor, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

def sam3_batch(frames_dir: str, output_size: int | None = None, prompt: str = "woman", quiet: bool = False) -> None:
    _sam3_inference(frames_dir, output_size=output_size, prompt=prompt)

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
    folder = Path(frames_dir)

    ckpt = hf_hub_download(repo_id=SAPIENS_REPO_ID, filename=SAPIENS_CHECKPOINT)
    model = init_model(SAPIENS_CONFIG, ckpt, device=device)

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

    elif seed_model == "sam3video":
        sam3_video_batch(frames_dir, output_size=output_size, prompt=prompt, sam31=False)

    elif seed_model == "sam31video":
        sam3_video_batch(frames_dir, output_size=output_size, prompt=prompt, sam31=True)

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

    return median_filter(phas, size=(window, 1, 1, 1), mode='nearest').astype(np.uint8)

def read_frame_from_videos(frame_root):

    if frame_root.endswith(VIDEO_EXTENSIONS):
        video_name = os.path.basename(frame_root)[:-4]
        container = av.open(frame_root)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames_list = []

        for frame in container.decode(stream):
            arr = frame.to_ndarray(format='rgb24')
            frames_list.append(arr)

        container.close()
        frames = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).contiguous()

    length = frames.shape[0]
    return frames, fps, length, video_name

def get_video_paths(input_root):
    video_paths = []

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))

    return sorted(video_paths)

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
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not videos:
        raise RuntimeError(f'No supported video files found in folder: {input_path}')

    return videos

def process_video(video_path, args: argparse.Namespace, temp_root: Path, batch_mode: bool = False) -> str:

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

    orig_w, orig_h, fps, duration = info(video_path)

    video_args = argparse.Namespace(**vars(args), video=video_path)

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

        )

        segments = matanyone(segments, segments_dir, mask_square, video_args)

        output_mask = finalize(

            segments,
            video_name,
            str(video_path),

        )

        overlay_target = str(Path(video_path).with_name(f"{video_name}_overlay.mp4"))

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

    print(f"MatAnyone runtime: {args.matanyone_version}")

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
            'max_size': mask_square,
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

def finalize(segments: List[SegmentInfo], video_name: str, video_path: str) -> str:

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
    parser = argparse.ArgumentParser(description='Minimal VR Video Masking Pipeline')
    parser.add_argument('input_path')
    parser.add_argument('--mask-height', type=int, default=1008)
    parser.add_argument('--segment-length', type=float, default=12)
    parser.add_argument('--erode', type=int, default=4)
    parser.add_argument('--dilate', type=int, default=0)
    parser.add_argument('--prompt', type=str, default='one woman')
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--seed-model', type=str, default='sam3', choices=['sam3', 'sam3video', 'sam31video', 'sapiens', 'hybrid'], help='Seed mask mode (sam3, sam3video, sam31video, sapiens, or hybrid)')
    parser.add_argument('--sapiens-threshold', type=float, default=0.5, help='Threshold for converting Sapiens alpha matte to a binary mask')
    parser.add_argument('--gate-dilate', type=int, default=5, help='Dilate SAM3 gating in hybrid mode')
    parser.add_argument('--matanyone-version', type=str, default='v2', choices=['v1', 'v2'], help='Select MatAnyone runtime version')
    parser.add_argument('--ma2-mem-every', type=int, default=None, help='Override MatAnyone mem_every (works for v1 and v2; e.g. 2 or 3 for faster refresh)')
    parser.add_argument('--ma2-max-mem-frames', type=int, default=None, help='Override MatAnyone memory window in frames (works for v1 and v2)')
    parser.add_argument('--ma2-use-long-term', type=str, default='auto', choices=['auto', 'on', 'off'], help='Override MatAnyone long-term memory mode (works for v1 and v2)')
    parser.add_argument('--temporal-median-window', type=int, default=0, help='Temporal median window for alpha cleanup. 0 disables; use odd values >= 3 (e.g. 5)')
    parser.add_argument('--no-normalize-input', dest='normalize_input', action='store_false', help='Skip upfront input normalization/transcoding')
    parser.set_defaults(normalize_input=True)
    parser.add_argument('--overlay-output', type=str, default='input_path', help='Write a composited video with the mask over the original source')
    parser.add_argument('--overlay-color', type=str, default='0x00ff00', help='Background color for the optional overlay preview (use 0x00ff00 for pure green)')
    parser.add_argument('--overlay-mask', type=str, default=None, help='Write a composited video with a provided mask over the original source')

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

        video_path = norm_video(video_path)
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
