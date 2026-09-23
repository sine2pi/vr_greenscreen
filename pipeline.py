import sys, functools, time, tqdm, random, shutil, gc, os, torch, cv2, numpy as np, glob, matplotlib.pyplot as plt, torch.nn.functional as F, argparse, re, subprocess, json, threading
from PIL import Image, ImageDraw, ImageFilter
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import List
from omegaconf import open_dict
from sam3.model_builder import build_sam3_predictor
from dataclasses import dataclass
from enum import Enum
from sam3.visualization_utils import load_frame, prepare_masks_for_visualization, visualize_formatted_frame_output

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
    sbs_frame_path: str = ''
    sbs_mask_path: str = ''
    video_path: str = ''

_matanyone_is_first_status = True
_matanyone_tqdm_lines = 1

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')
ENCODER = 'hevc_nvenc'
MATANYONE_V1 = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
MATANYONE_V2 = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
FFMPEG_BIN = shutil.which('ffmpeg') or 'ffmpeg'
FFPROBE_BIN = shutil.which('ffprobe') or 'ffprobe'

def have(a):
    if a == bool:
        if a:
            return a is not None
    return a is not None
def aorb(a, b):
    return a if have(a) else b
def aborc(a, b, c):
    return aorb(a, aorb(b, c))
def abcord(a, b, c, d):
    return aorb(a, aborc(b, c, d))

def _save_frames_as_folder(frames, out_dir: Path):
    _ensure_dir(out_dir)
    for i, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), frame_bgr)

def _save_mask(mask_gray: np.ndarray, out_path: Path):
    _ensure_dir(out_path.parent)
    Image.fromarray(mask_gray, mode="L").save(str(out_path))

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def _safe_rel(path: Path, base: Path):
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)

def check_vfr(video_path: str, max_packets_to_read: int = 500) -> bool:

    cmd = [
        FFPROBE_BIN, '-v', 'quiet',
        '-print_format', 'json',
        '-select_streams', 'v:0',
        '-show_entries', 'packet=duration',
        '-read_intervals', f'%+#{max_packets_to_read}',  
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        packets = data.get('packets', [])
        if len(packets) < 10:
            return False  
            
        durations = [p.get('duration') for p in packets if p.get('duration') is not None]
        valid_durations = [int(d) for d in durations if str(d).isdigit()]
        if not valid_durations:
            return False
            
        first_duration = valid_durations[0]
        for d in valid_durations[1:]:
            if d != first_duration:
                return True  
        return False  
        
    except Exception:
        return False  

def normalize_fps(fps: float) -> float:
    rounded = round(fps, 2)
    if abs(rounded - round(rounded)) < 0.05:
        return float(round(rounded))

    return rounded

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

def encoder_args(data=None) -> list[str]:

    fps = data['fps'] if data is not None else 60

    return [

        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp',
        '-fps_mode', 'cfr',
        '-r', str(fps),
        '-c:v', ENCODER,
        '-preset', 'p6',
        '-profile:v', 'main',
        '-pix_fmt', 'yuv420p',
        '-g', '20',
        '-b:v', '70M',
        '-maxrate', '90M',
        '-bufsize', '140M',
        '-rc:v', 'cbr',
        '-tag:v', 'hvc1',
        '-map', '0:a?',
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
            print()
            print(f"\r{progress_prefix}{_ffmpeg_progress(line)}", end='', flush=True)
            print()
    process.wait()
    print()

    return process.returncode, "".join(stderr_lines)

@functools.lru_cache(maxsize=256)
def info(path, ffprobe_bin=FFPROBE_BIN):
    metadata = {}
    cmd_stream = [ffprobe_bin, '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'v:0', path]
    res_stream = subprocess.run(cmd_stream, capture_output=True, text=True)
    data = json.loads(res_stream.stdout)

    if not data.get('streams'):
        return None
        
    stream = data['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    codec = stream.get('codec_name', '') 
    duration = float(stream.get('duration', 0))
    pix_fmt = stream.get('pix_fmt', 'yuv420p')
    fps_str = stream.get('r_frame_rate', '60/1')
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom if denom != 0 else 60.0
    f_tot = stream.get('nb_frames')

    if f_tot:
        frames = int(f_tot)
    else:
        frames = int(duration * fps) if duration > 0 else 0
        
    metadata.update({
        'width': width,
        'height': height,
        'codec': codec,
        'duration': duration,
        'pix_fmt': pix_fmt,
        'fps_str': fps_str,
        'fps': fps,
        'frames': frames,
    })
    return metadata

def norm_video(source_video, w = None, h = None, fps = None, progress_prefix: str = "[normalize] ", video_args = None) -> str:

    data= info(source_video)
    source_path = Path(source_video).expanduser().resolve()
    output_video = str(source_path.with_name(f"{source_path.stem}_normed.mp4"))

    fps = aorb(fps, 60)
    enc = encoder_args(data)

    if w is not None:
        wi = w
        hi = h
    else:
        wi = data['width']
        hi = data['height']

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', source_video,
        '-filter_complex', f'[0:v]fps={fps},setpts=N/({fps}*TB),scale=w={wi}:h={hi}:flags=bilinear:out_range=tv:threads=0',
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

def cfr_video(source_video, video_args = None, progress_prefix: str = "[normalize]") -> str:

    data = info(source_video)
    print(f"-- {source_video} has a Variable Frame Rate - Converting to CFR")
    
    source_path = Path(source_video).expanduser().resolve()
    output_video = str(source_path.with_name(f"{source_path.stem}_CFR.mp4"))

    fps = aorb(data['fps'], 60)
    fps = normalize_fps(fps)

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', source_video,
        '-filter_complex', (
            f'[0:v]fps={fps},setpts=N/({fps}*TB),scale=w={data["width"]}:h={data["height"]}:flags=bilinear:out_range=pc:threads=0[v];'
            f'[0:a]asetpts=N/SR/TB,aresample=async=1:min_comp=0.001:min_hard_comp=0.1:first_pts=0[a]'
        ),
        '-map', '[v]',
        '-map', '[a]',
        '-fps_mode', 'cfr',
        '-r', str(fps),
        '-c:v', 'hevc_nvenc',
        '-preset', 'p6',
        '-profile:v', 'main',
        '-pix_fmt', 'yuv420p',
        '-g', '20',
        '-b:v', '70M',
        '-maxrate', '80M',
        '-bufsize', '120M',  
        '-rc:v', 'cbr',    
        '-tag:v', 'hvc1',
        '-aspect', '2:1',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-metadata:s:v:0', 'stereo_mode=left_right',
        '-movflags', '+faststart+write_colr+use_metadata_tags',
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

def resize_video(source_video, output_video, progress_prefix: str = "[resize] ") -> str:

    data = info(source_video)
    enc = encoder_args(data)
    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or '.', exist_ok=True)

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', output_video,
        '-filter_complex', f'[0:v]fps={data["fps"]},setpts=N/({data["fps"]}*TB),scale={data["width"]}:{data["height"]}:flags=bilinear',
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

    data = info(video_list[0])

    BATCH_SIZE = 50
    n = len(video_list)
    enc = encoder_args(data)

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
        filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={data['fps']},format={data['pix_fmt']}[v{i}]")

    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0,setpts=PTS-STARTPTS[outv]")
    filter_complex = ";".join(filter_parts)
    filter_file = os.path.join(common_dir, "_concat_filter.txt")

    cmd_inline = [

        'ffmpeg', '-y', '-hwaccel', 'cuda',
        *[item for rel in rel_vid for item in ['-i', rel]],
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        rel_output,
    ]

    rc, stderr_text = ffmpeg_progress(cmd_inline, cwd=common_dir)

    if rc != 0 and ('The filename or extension is too long' in stderr_text or 'WinError 206' in stderr_text):

        with open(filter_file, 'w', encoding='utf-8') as f:
            f.write(filter_complex)

        cmd_script = [

            'ffmpeg', '-y', '-hwaccel', 'cuda',
            *[item for rel in rel_vid for item in ['-i', rel]],
            '-/filter_complex', '_concat_filter.txt',
            '-map', '[outv]',
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
    data = info(video_path)
    enc = encoder_args(data)
    output_paths = []
    eye_size = height
    crop_filter = f"crop={eye_size}:{eye_size}:0:0"
    for ts in timestamps:
        out_path = os.path.join(output_dir, f"frame_{ts:.0f}s.png")

        cmd = [

            'ffmpeg', '-y', '-hwaccel', 'cuda',
            '-ss', str(ts),
            '-i', video_path,
            '-vf', crop_filter,
            '-frames:v', '1', '-compression_level', '1',
            *enc,
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

    data = info(stereo_video)
    enc = encoder_args(data)

    start_frame = round(start * data['fps'])
    end_frame = round(end * data['fps'])
    frames = end_frame - start_frame

    if frames <= 0:
        raise RuntimeError(f"Invalid segment: {start=} {end=} {data['fps']=} -> {frames} frames")

    aligned_start = start_frame / data['fps']
    keyframe_seek = max(0.0, aligned_start - 2.0)
    fine_seek = aligned_start - keyframe_seek
    seg_dur = frames / data['fps']

    orig_eye = target_height
    target_eye = target_height

    frame_left = f"crop={orig_eye}:{orig_eye}:0:0"
    frame_right = f"crop={orig_eye}:{orig_eye}:{orig_eye}:0"

    video_left = f"crop={target_eye}:{target_eye}:0:0"
    video_right = f"crop={target_eye}:{target_eye}:{target_eye}:0"

    scale_w = target_height * 2
    scale_h = target_height

    filter_complex = (

        f"[0:v]trim=start={fine_seek}:duration={seg_dur},setpts=PTS-STARTPTS,fps={data['fps']},scale={scale_w}:{scale_h}:flags=bilinear,split=2[full][toscale];"
        f"[full]split=2[fullL][fullR];"
        f"[fullL]select=eq(n\\,0),{frame_left}[frame_left];"
        f"[fullR]select=eq(n\\,0),{frame_right}[frame_right];"
        f"[toscale]format=nv12,scale={scale_w}:{scale_h}:flags=bilinear,split=2[sL][sR];"
        f"[sL]{video_left}[video_left];"
        f"[sR]{video_right}[video_right]"
    )

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
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

    process.wait()

    if process.returncode != 0:
        tail = "".join(stderr_lines[-60:])
        raise RuntimeError(f"Segment extraction failed.\n\nFFmpeg tail:\n{tail}")

    return left_frame_out, right_frame_out, left_video_out, right_video_out

def extract_segment_sbs(
    stereo_video: str,
    start: float,
    end: float,
    target_height: int,
    sbs_frame_out: str,
    sbs_video_out: str,
    progress_prefix: str = "",
) -> tuple[str, str]:

    data = info(stereo_video)
    _ = data['width'], data['height'], data['duration'], data['pix_fmt'], data['frames']
    enc = encoder_args(data)
    start_frame = round(start * data['fps'])
    end_frame = round(end * data['fps'])
    frames = end_frame - start_frame

    if frames <= 0:
        raise RuntimeError(f"Invalid segment: {start=} {end=} {data['fps']=} -> {frames} frames")

    aligned_start = start_frame / data['fps']
    keyframe_seek = max(0.0, aligned_start - 2.0)
    fine_seek = aligned_start - keyframe_seek
    seg_dur = frames / data['fps']
    scale_w = target_height * 2
    scale_h = target_height

    filter_complex = (
        f"[0:v]trim=start={fine_seek}:duration={seg_dur},setpts=PTS-STARTPTS,"
        f"fps={data['fps']},format=nv12,scale={scale_w}:{scale_h}:flags=bilinear[out]"
    )

    cmd_video = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-hide_banner',
        '-ss', str(keyframe_seek),
        '-i', stereo_video,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-frames:v', str(frames),
        *enc,
        sbs_video_out,
    ]
    rc, stderr_text = ffmpeg_progress(cmd_video, progress_prefix=progress_prefix)
    if rc != 0:
        tail = "".join(stderr_text.splitlines(True)[-60:])
        raise RuntimeError(f"SBS segment extraction failed.\n\nFFmpeg tail:\n{tail}")

    cmd_frame = [
        '-hwaccel', 'cuda',
        '-i', sbs_video_out,
        '-vf', 'select=eq(n\\,0)',
        '-frames:v', '1',
        '-compression_level', '1',
        sbs_frame_out,
    ]
    result = subprocess.run(cmd_frame, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SBS first-frame extraction failed: {result.stderr}")

    return sbs_frame_out, sbs_video_out

def overlay_path(source_video: str, output_path: str) -> str:
    source_path = Path(source_video).expanduser()
    target_path = Path(output_path).expanduser()
    overlay_stem = source_path.stem
    if target_path.exists() and target_path.is_dir():
        return str(target_path / f"{overlay_stem}_overlay.mp4")
    if target_path.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.wmv'}:
        return str(target_path)
    return str(target_path.with_suffix('.mp4'))

def get_bg(mask_path: str, size: int, background_color: str = '0x00ff00'):
    mask_path = Path(mask_path).expanduser()
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={background_color}:s={size}x{size}:d=1,format=gray", "-frames:v","1",
           "-vf", f"format=rgba,scale={size}:{size}:flags=lanczos", str(mask_path)]
    subprocess.run(cmd, capture_output=True, text=True)
    return str(mask_path)

def mask_overlay(source_video: str, mask_video: str, output_path: str, background_color: str = '0x00ff00', video_args: argparse.Namespace = None) -> str:

    resolved_path = overlay_path(source_video, output_path)
    os.makedirs(os.path.dirname(os.path.abspath(resolved_path)) or '.', exist_ok=True)

    data = info(source_video)
    enc = encoder_args(data)

    os.makedirs(os.path.dirname(os.path.abspath(resolved_path)) or '.', exist_ok=True)

    orig_filter = f"setpts=PTS-STARTPTS,fps={data['fps']},format=rgba"
    mask_filter = f"setpts=PTS-STARTPTS,fps={data['fps']},format=gray,scale={data['width']}:{data['height']},lut=a=val/255"
    bg_filter = f"setpts=PTS-STARTPTS,fps={data['fps']},format=rgba"

    filter_complex = (
        f"[0:v]{orig_filter}[orig];"
        f"[1:v]{mask_filter}[mask_alpha];"
        f"[orig][mask_alpha]alphamerge[alphaed];"
        f"[2:v]{bg_filter}[bg];"
        f"[bg][alphaed]overlay=format=auto[out]"
    )

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', source_video,
        '-i', mask_video,
        '-f', 'lavfi', '-i', f'color=c={background_color}:s={data["width"]}x{data["height"]}:d={data["duration"]}:r={data["fps"]}',
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-t', str(data["duration"]),
        *enc,
        resolved_path,
    ]

    rc, stderr_text = ffmpeg_progress(cmd)
    if rc != 0:
        raise RuntimeError(f"Mask overlay failed.\n\nFFmpeg tail:\n{''.join(stderr_text.splitlines(True)[-40:])}")
    return resolved_path

def stereo_video(left_video: str, right_video: str, output_path: str) -> str:

    data = info(left_video)
    enc = encoder_args(data)
    filter_complex = "[0:v][1:v]hstack=inputs=2[out]"

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'cuda',
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

def get_video_paths(input_root):
    video_paths = []

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))

    return sorted(video_paths)

def _ceil_to(n: int, base: int) -> int:
    return ((n + base - 1) // base) * base

def get_circle_mask(size: int) -> str:

    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.gettempdir())
    mask_path = tmp_dir / f"circle_mask_{size}.png"

    try:

        scale = 4
        size_hr = size * scale
        print(f"High-resolution mask size: {size_hr}")
        circle_img = Image.new("L", (size_hr, size_hr), 0)

        draw = ImageDraw.Draw(circle_img)
        draw.ellipse([0, 0, size_hr - 1, size_hr - 1], fill=255)

        print(f" circle_img size before resize: {circle_img.size}")

        circle_img = circle_img.resize((size, size), Image.Resampling.LANCZOS)
        print(f"Resized mask size: {size}")
        print(f" circle_img size before blur: {circle_img.size}")
        circle_img = circle_img.filter(ImageFilter.GaussianBlur(radius=1))
        circle_img.save(str(mask_path))

    except ImportError:
        cmd = [
            "ffmpeg","-y",
            "-f","lavfi",
            "-i", f"color=c=white:s={size}x{size}:d=1,format=gray",
            "-vf", "geq=lum='if(lte((X-W/2)*(X-W/2)+(Y-H/2)*(Y-H/2),(min(W,H)/2)*(min(W,H)/2)),255,0)'"
            "-frames:v", "1",
            str(mask_path),
        ]

        subprocess.run(cmd, capture_output=True, text=True)

    return str(mask_path)

def _input_pairs(input_path: str) -> list[tuple[Path, Path]]:
    path = Path(str(input_path)).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video file: {path}")

        mask_path = path.with_name(f"{path.stem}_mask{path.suffix}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {path}: expected {mask_path}")
        return [(path, mask_path)]

    if not path.is_dir():
        raise RuntimeError(f"Input path is not a file or folder: {input_path}")

    pairs: list[tuple[Path, Path]] = []
    for candidate in sorted(path.rglob('*')):
        if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if candidate.stem.endswith('_mask'):
            continue
        mask_path = candidate.with_name(f"{candidate.stem}_mask{candidate.suffix}")
        if mask_path.exists():
            pairs.append((candidate.resolve(), mask_path.resolve()))
    if not pairs:
        raise RuntimeError(f"No original/mask video pairs found in folder: {input_path}")
    return pairs

def pack_video(
    video_path: str,
    mask_path: str,
    output_path: str | None = None,
    sync_frames = None,
    progress_prefix: str = "[ALPHA] "

) -> str:

    if not output_path:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_alpha{ext}"

    print(f"{'='*60}")
    print(f"Alpha packing: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    actual_mask = mask_path
    synced_tmp = None

    if sync_frames is not None:
        fps = info(mask_path)[2]
        print(f"Syncing mask by {sync_frames} frame(s)...")
        synced_tmp = sync_mask_to_video(mask_path, fps=fps, frame_offset=sync_frames)
        actual_mask = synced_tmp
    
    data = info(video_path)
    mask_data = info(actual_mask)
    print(f"Video: {data['width']}x{data['height']}")
    print(f"Mask: {mask_data['width']}x{mask_data['height']}")

    out_h = _ceil_to(data['height'], 32)
    enc = encoder_args(data)

    if data['width'] == 2 * data['height']:
        out_w = 2 * out_h
    else:
        out_w = _ceil_to(data['width'], 32)

    if (out_w, out_h) != (data['width'], data['height']):
        print(f"NVENC-aligned output: {out_w}x{out_h}")

    overlay_size = int(out_h * 0.4)
    print(f"Initial overlay size: {overlay_size}")
    overlay_size = (overlay_size // 4) * 4
    half_overlay = overlay_size // 2
    print(f"Half overlay size: {half_overlay}, overlay size: {overlay_size}, out_w: {out_w}, out_h: {out_h}")

    if data['height'] <= 2400:
        erosion_threshold = 32768
        contrast = 2.0
        gamma = 1.2

    else:
        erosion_threshold = 65535
        contrast = 2.5
        gamma = 1.4

    sigma = 1.8

    erosion_filter = f"erosion=threshold0={erosion_threshold}:coordinates=255,"
    print(f"Mask Gen Params: gblur={sigma:.1f}, erosion={erosion_threshold}, contrast={contrast}, gamma={gamma}")
    print(f"Adjusted overlay size: {overlay_size}")
    circle_mask = get_circle_mask(overlay_size)

    filter_parts: list[str] = [

        f"[0:v]scale=w={out_w}:h={out_h}:flags=bilinear[vid]",
        "[1:v]split=2[mask1][mask2]",
        "[2:v]format=gray,split=2[circle_l][circle_r]",

        (
            f"[mask1]crop=ih:ih:0:0,"
            f"scale={overlay_size}:{overlay_size}:flags=area,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[left_scaled]"
        ),
        "[left_scaled][circle_l]alphamerge,format=rgba[left_circle]",

        (
            f"[mask2]crop=ih:ih:iw-ih:0,"
            f"scale={overlay_size}:{overlay_size}:flags=area,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[right_scaled]"
        ),
        "[right_scaled][circle_r]alphamerge,format=rgba[right_circle]",
        "[left_circle]split=2[left_for_top][left_for_bottom]",
        f"[left_for_top]crop={overlay_size}:{half_overlay}:0:0,format=rgba[left_top]",
        f"[left_for_bottom]crop={overlay_size}:{half_overlay}:0:{half_overlay},format=rgba[left_bottom]",
        "[right_circle]split=4[r1][r2][r3][r4]",
        f"[r1]crop={half_overlay}:{half_overlay}:0:0,format=rgba[right_tl]",
        f"[r2]crop={half_overlay}:{half_overlay}:{half_overlay}:0,format=rgba[right_tr]",
        f"[r3]crop={half_overlay}:{half_overlay}:0:{half_overlay},format=rgba[right_bl]",
        f"[r4]crop={half_overlay}:{half_overlay}:{half_overlay}:{half_overlay},format=rgba[right_br]",
        "[vid][left_top]overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h[v1]",
        "[v1][left_bottom]overlay=x=(main_w-overlay_w)/2:y=0[v2]",
        "[v2][right_tl]overlay=x=main_w-overlay_w:y=main_h-overlay_h[v3]",
        "[v3][right_tr]overlay=x=0:y=main_h-overlay_h[v4]",
        "[v4][right_bl]overlay=x=main_w-overlay_w:y=0[v5]",
        "[v5][right_br]overlay=x=0:y=0[out]",
    ]

    filter_complex = ";".join(filter_parts)

    cmd: list[str] = [

        'ffmpeg', '-y', '-hwaccel', 'cuda',
        "-filter_threads", "0",
        "-threads", "0",
        "-i", video_path,
        "-i", mask_path,
        "-loop", "1",
        "-i", circle_mask,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-shortest",
        *enc,
        output_path,
    ]

    rc, stderr_text = ffmpeg_progress(cmd, progress_prefix=progress_prefix)

    if rc != 0:
        raise RuntimeError(
            "Alpha failed.\n\nFFmpeg tail:\n"
            + ''.join(stderr_text.splitlines(True)[-40:])
        )

    if not os.path.exists(output_path):
        raise RuntimeError(f"Alpha failed: {output_path}")
    if synced_tmp and os.path.exists(synced_tmp):
        os.remove(synced_tmp)
    print(f"Alpha packed: {output_path}")
    return output_path

def packer(input_path, video_args=None) -> int:

    input_pairs = _input_pairs(input_path)
    processed = []
    
    for (video_path, mask_path) in input_pairs:
        if video_args.fisheye180:
            video_path = fisheye180(str(video_path), flag=False)
            mask_path = fisheye180(str(mask_path), flag=True)
        packed_path = pack_video(str(video_path), str(mask_path))
        processed.append((str(video_path), str(mask_path), packed_path))
        print()

    if not processed:
        print(" No files were packed successfully")
        return 1

    for video_path, mask_path, packed_path in processed:
        print(f"{video_path} <- {mask_path} -> {packed_path}")

    return 0

def _decomp_paths(packed_video: Path) -> tuple[Path, Path]:
    stem = packed_video.stem

    if stem.endswith('_alpha'):
        base_stem = stem[:-6]
    else:
        base_stem = f'{stem}_decomposed'

    video_output = packed_video.with_name(f'{base_stem}{packed_video.suffix}')
    mask_output = packed_video.with_name(f'{base_stem}_mask{packed_video.suffix}')
    return video_output, mask_output

def alpha_decomp(
    packed_video: str,
    video_output: str | None = None,
    mask_output: str | None = None,
    cleanup_mask_path: str | None = None,
    progress_prefix: str = "[DECOMPOSE] ",
) -> tuple[str, str]:

    packed_path = Path(packed_video).expanduser().resolve()
    if not packed_path.exists():
        raise FileNotFoundError(f'Packed video not found: {packed_video}')
    if packed_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError(f'Unsupported video file: {packed_video}')

    data = info(str(packed_path))
    enc = encoder_args(data)
    eye_size = data['height']
    overlay_size = int(data['height'] * 0.4)
    overlay_size = (overlay_size // 4) * 4
    half_overlay = overlay_size // 2

    if overlay_size <= 0 or half_overlay <= 0:
        raise RuntimeError(f'Packed video is too small to decode alpha payload: {data["width"]}x{data["height"]}')
    if data['width'] < overlay_size or data['height'] < overlay_size:
        raise RuntimeError(f'Packed video dimensions are invalid for alpha payload decode: {data["width"]}x{data["height"]}')

    default_video_out, default_mask_out = _decomp_paths(packed_path)
    video_out = Path(video_output).expanduser().resolve() if video_output else default_video_out
    mask_out = Path(mask_output).expanduser().resolve() if mask_output else default_mask_out

    print(f"{'='*60}")
    print(f"Decomposing alpha-packed video: {packed_path.name}")
    print(f"{'='*60}")

    video_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    if cleanup_mask_path is not None:
        cleanup_mask = Path(cleanup_mask_path).expanduser().resolve()
        if not cleanup_mask.exists():
            raise FileNotFoundError(f'Decompose cleanup mask not found: {cleanup_mask_path}')
        if cleanup_mask.suffix.lower() != '.png':
            raise RuntimeError(f'Decompose cleanup mask must be a PNG image: {cleanup_mask_path}')

        clean_filter = (
            "[1:v]format=rgba[mask_src];"
            "[mask_src][0:v]scale2ref[mask][video];"
            "[video][mask]overlay=0:0:format=auto[out]"
        )
        copy_cmd = [
            'ffmpeg', '-y', '-hwaccel', 'cuda',
            '-i', str(packed_path),
            '-i', str(cleanup_mask),
            '-filter_complex', clean_filter,
            '-map', '[out]',
            *enc,
            str(video_out),
        ]
    else:
        copy_cmd = [
            'ffmpeg', '-y', '-hwaccel', 'cuda',
            '-i', str(packed_path),
            '-map', '0',
            '-c', 'copy',
            str(video_out),
        ]

    copy_rc, copy_stderr = ffmpeg_progress(copy_cmd, progress_prefix=f"{progress_prefix}[VIDEO] ")
    if copy_rc != 0:
        raise RuntimeError(
            "Video extraction failed.\n\nFFmpeg tail:\n"
            + ''.join(copy_stderr.splitlines(True)[-40:])
        )

    center_x = (data['width'] - overlay_size) // 2
    right_eye_x = data['width'] - eye_size

    circle_gate = "geq=lum='if(lte((X-W/2)*(X-W/2)+(Y-H/2)*(Y-H/2),(min(W,H)/2)*(min(W,H)/2)),min(lum(X,Y)*4.5,255),0)':cb=128:cr=128"

    filter_parts = [
        f"[0:v]crop={overlay_size}:{half_overlay}:{center_x}:{data['height'] - half_overlay}[left_top]",
        f"[0:v]crop={overlay_size}:{half_overlay}:{center_x}:0[left_bottom]",
        "[left_top][left_bottom]vstack=inputs=2[left_circle_raw]",
        f"[left_circle_raw]format=gray,{circle_gate},scale={eye_size}:{eye_size}:flags=bilinear[left_eye]",
        f"[0:v]crop={half_overlay}:{half_overlay}:{data['width'] - half_overlay}:{data['height'] - half_overlay}[r1]",
        f"[0:v]crop={half_overlay}:{half_overlay}:0:{data['height'] - half_overlay}[r2]",
        f"[0:v]crop={half_overlay}:{half_overlay}:{data['width'] - half_overlay}:0[r3]",
        f"[0:v]crop={half_overlay}:{half_overlay}:0:0[r4]",
        "[r1][r2]hstack=inputs=2[right_top]",
        "[r3][r4]hstack=inputs=2[right_bottom]",
        "[right_top][right_bottom]vstack=inputs=2[right_circle_raw]",
        f"[right_circle_raw]format=gray,{circle_gate},scale={eye_size}:{eye_size}:flags=bilinear[right_eye]",
        "[0:v]format=gray,geq=lum='0'[mask_bg]",
        "[mask_bg][left_eye]overlay=0:0[mask_left]",
        f"[mask_left][right_eye]overlay={right_eye_x}:0[mask_comp]",
        "[mask_comp]scale=in_range=tv:out_range=tv,format=gray[out]",
    ]

    mask_cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', str(packed_path),
        '-filter_complex', ';'.join(filter_parts),
        '-map', '[out]',
        '-r', str(data['fps']),
        '-c:v', ENCODER,
        '-preset', 'p7',
        '-pix_fmt', 'yuv420p',
        '-an',
        str(mask_out),
    ]

    mask_rc, mask_stderr = ffmpeg_progress(mask_cmd, progress_prefix=f"{progress_prefix}[MASK] ")
    if mask_rc != 0:
        raise RuntimeError(
            "Mask extraction failed.\n\nFFmpeg tail:\n"
            + ''.join(mask_stderr.splitlines(True)[-40:])
        )

    print(f"Decomposed video: {video_out}")
    print(f"Decomposed mask: {mask_out}")
    return str(video_out), str(mask_out)

def decompose_alpha(input_path: str, cleanup_mask_path: str | None = None) -> int:
    packed_videos = _input_videos(input_path)
    outputs: list[tuple[str, str, str]] = []

    for index, packed_video in enumerate(packed_videos, 1):
        print(f"[{index}/{len(packed_videos)}] Decompose: {packed_video.name}")
        video_out, mask_out = alpha_decomp(
            str(packed_video),
            cleanup_mask_path=cleanup_mask_path,
        )
        outputs.append((str(packed_video), video_out, mask_out))
        print()

    print('=' * 60)
    print('Alpha decomposition complete')
    print('=' * 60)
    for packed_video, video_out, mask_out in outputs:
        print(f"{packed_video} -> {video_out}")
        print(f"{packed_video} -> {mask_out}")

    return 0

def sync_mask_to_video(mask_path: str, fps: float, frame_offset: int = 0) -> str:

    data = info(mask_path)
    enc = encoder_args(data)

    frame_duration = abs(frame_offset) / fps
    if frame_offset > 0:
        vf = f"trim=start={frame_duration},setpts=PTS-STARTPTS"
    elif frame_offset < 0:
        vf = f"tpad=start_duration={frame_duration}:color=black"
    else:
        vf = "null"

    base, ext = os.path.splitext(mask_path)
    synced_path = f"{base}_synced{ext}"

    cmd = [

        'ffmpeg', '-y',
        '-i', mask_path,
        '-vf', vf,
        *enc,
        synced_path,
    ]

    ffmpeg_progress(cmd)
    return synced_path

def fisheye180(input_video: str, flag=False) -> str:
    data = info(input_video)
    enc = encoder_args(data)
    print('Starting FISHEYE180 conversion...')
    input_video = str(Path(input_video).expanduser().resolve())
    filename, ext = os.path.splitext(input_video)
    output_video = f'{filename}_FISHEYE180{ext}' if not flag else f'{filename}_FISHEYE180_mask{ext}'
    
    target_w = data['width']
    target_h = data['height']
    fps = data['fps']
    eye_w = target_w // 2

    if eye_w <= 0 or target_h <= 0:
        raise RuntimeError(f'Invalid input dimensions for fisheye conversion: {target_w}x{target_h}')

    filter_parts = [
        f'[0:v]fps={fps},setpts=N/({fps}*TB),split=2[left_src][right_src]',
        f'[left_src]crop=iw/2:ih:0:0,v360=input=hequirect:output=fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180:w={eye_w}:h={target_h}[left]',
        f'[right_src]crop=iw/2:ih:iw/2:0,v360=input=hequirect:output=fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180:w={eye_w}:h={target_h}[right]',
        f'[left][right]hstack,scale=w={target_w}:h={target_h}:flags=bilinear[stacked]',
    ]
    mask_png = 'assets/black_mask.png'
    if mask_png is not None:
        mask_png = str(Path(mask_png).expanduser().resolve())
        if not os.path.exists(mask_png):
            raise FileNotFoundError(f'Fisheye mask not found: {mask_png}')
        if Path(mask_png).suffix.lower() != '.png':
            raise RuntimeError(f'Fisheye mask must be a PNG image: {mask_png}')

        filter_parts.extend([
            '[1:v]format=rgba[mask_src]',
            '[mask_src][stacked]scale2ref[mask][stacked_ref]',
            '[stacked_ref][mask]overlay=0:0:format=auto[out]',
        ])
    else:
        filter_parts.append('[stacked]copy[out]')

    filter_complex = ';'.join(filter_parts)

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', input_video,
    ]

    if mask_png is not None:
        cmd.extend(['-i', mask_png])

    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-map', '0:a?',
        *enc,
        output_video,
    ])

    rc, _ = ffmpeg_progress(cmd)

    if rc != 0:
        raise RuntimeError(f'FFmpeg failed with exit code {rc}')

    print(f'FISHEYE180 output: {output_video}')

    return output_video

def run_fisheye180_mode(input_path: str, flag: bool = False) -> int:

    video_paths = _input_videos(input_path)
    outputs: list[str] = []

    for index, video_path in enumerate(video_paths, 1):
        print(f'[{index}/{len(video_paths)}] FISHEYE180: {video_path}')
        output_path = fisheye180(str(video_path), flag=flag)
        outputs.append(output_path)
        print()

    print('=' * 60)
    print('FISHEYE180 conversion complete')
    print('=' * 60)
    for output_path in outputs:
        print(output_path)

    return 0

def video_frames(frame_root, max_size):

    if frame_root.endswith(VIDEO_EXTENSIONS):
        video_name = os.path.basename(frame_root)[:-4]

        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate',
            '-of', 'json',
            frame_root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream = json.loads(result.stdout)['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])

        frame_rate = stream.get('avg_frame_rate')
        if not frame_rate or frame_rate == '0/0':
            frame_rate = stream.get('r_frame_rate')
        num, _, den = frame_rate.partition('/')
        den = den or '1'
        fps = float(num) / float(den) if float(den) != 0 else float(num)

        command = [
            "ffmpeg",
            "-v", "error",
            "-i", frame_root,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]

        command.extend([
        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp', '-c:v', 'hevc_nvenc', '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1', '-g', '20', '-b:v', '50M', '-c:a', 'aac', '-b:a', '256k', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags', '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709'])

        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {process.stderr.decode(errors='ignore')}")

        frame_size = width * height * 3
        raw = process.stdout
        num_frames = len(raw) // frame_size
        arr = np.frombuffer(raw[:num_frames * frame_size], dtype=np.uint8)
        arr = arr.reshape(num_frames, height, width, 3)

        frames = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).contiguous()
        frames = frames.float()

        if max_size is not None:
            if frames.shape != (max_size, max_size):
                frames = torch.nn.functional.interpolate(
                    frames,
                    size=(max_size, max_size),
                    mode="area",
                )

    length = frames.shape[0]
    return frames, fps, length, video_name

class TorchCodecVideoLoader:

    def __init__(self, video_path, image_size=None, offload_video_to_cpu=False, img_mean=None, img_std=None, gpu_device=None, norm=False):
        from torchcodec import _core as core

        self.image_size = image_size
        self.norm = norm
        self.out_device = torch.device("cpu") if offload_video_to_cpu else (gpu_device or torch.device("cuda"))
        decode_device = (gpu_device or torch.device("cuda")) if torch.cuda.is_available() else torch.device("cpu")
        
        if norm:
            self.img_mean = torch.tensor(img_mean, dtype=torch.float16, device=self.out_device).view(3, 1, 1)
            self.img_std = torch.tensor(img_std, dtype=torch.float16, device=self.out_device).view(3, 1, 1)

        self.decoder = core.create_from_file(video_path, "exact")
        core.scan_all_streams_to_update_metadata(self.decoder)
        core.add_video_stream(
            self.decoder, dimension_order="NCHW", device=str(decode_device),
            num_threads=1 if decode_device.type == "cuda" else 4
        )

        meta = core.get_container_metadata(self.decoder)
        stream = meta.streams[meta.best_video_stream_index]
        self.num_frames = stream.num_frames_from_content
        self.video_height = stream.height
        self.video_width = stream.width

        self.images = [None] * self.num_frames
        self.exception = None

        self.thread = threading.Thread(target=self._background_decode, daemon=True)
        self.thread.start()

    @torch.inference_mode()
    def _background_decode(self):
        from torchcodec import _core as core
        try:
            pbar = tqdm(desc=f"frame loading (TorchCodec) ]", total=self.num_frames)
            for i in range(self.num_frames):
                frame_data, *_ = core.get_frame_at_index(self.decoder, frame_index=i)
                frame = frame_data.float()

                if self.image_size is not None:
                    frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False).squeeze(0)
                frame = frame.half() / 255.0
                if frame.device != self.out_device:
                    frame = frame.to(self.out_device, non_blocking=True)
                if self.norm:
                    frame = (frame - self.img_mean) / self.img_std

                self.images[i] = frame
                pbar.update(1)
            pbar.close()
        except Exception as e:
            self.exception = e

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if idx < 0: idx += self.num_frames
        if idx < 0 or idx >= self.num_frames: raise IndexError("Frame index out of bounds")

        max_retries = 1200
        for _ in range(max_retries):
            if self.exception: raise RuntimeError("Background decoding failed") from self.exception
            if self.images[idx] is not None:
                return self.images[idx]
            time.sleep(0.01)
        raise RuntimeError(f"Timeout waiting for frame {idx} to decode.")

    def get_all_frames(self, start=0, max_frames=None):
        end = min(start + max_frames, self.num_frames) if max_frames else self.num_frames
        return torch.stack([self[i] for i in range(start, end)])

def download_ckpt_from_hf(version="sam3", force_download=False, local_files_only=False, token=None):
    from huggingface_hub import hf_hub_download

    if version == "sam3.1":
        repo_id = "sin2piusc/sam31sin"
        ckpt_name = "sam3.1_multiplex.pt"
    elif version == "sam3lite":
        repo_id = "vil-uob/sam3-litetext-l"
        ckpt_name = "model.safetensors"
    elif version == "sam3image":
        repo_id = "sin2piusc/sam3_fta"
        ckpt_name = "sam3.pth"
    elif version == "sam3m":
        repo_id = "feyninc/multimatte"
        ckpt_name = "model.safetensors"
    elif version == "local":
        checkpoint_path = r"sam3/sam3.pt"
        return checkpoint_path
    else:
        repo_id = "facebook/sam3"
        ckpt_name = "sam3.pt"
  
    return hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_name,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
 
class sam3_video_inference:
    def __init__(self, video_path, video_args):

        self.video_path = video_path
        self.video_args = video_args
        bpe_path = 'assets/bpe_simple_vocab_16e6.txt.gz'
        checkpoint_path = download_ckpt_from_hf(version=video_args.model, force_download=False, local_files_only=False)

        self.predictor = build_sam3_predictor(
            checkpoint_path = checkpoint_path,
            bpe_path = bpe_path,
            version = video_args.model,
            compile = False,
            warm_up = True,
            max_num_objects = 1,
            multiplex_count = 16,
            use_fa3 = False,
            use_rope_real = False,
            async_loading_frames = False,
            num_obj_for_compile=1,
            apply_temporal_disambiguation=True,
            device = device,
            video_loader_type="cv2",
            load_from_HF=False,
            default_output_prob_thresh=0.1, 
            strict_state_dict_loading=False, 
            session_expiration_sec=1200, 
            eval_mode=True, 
       
        )

    def propagate_in_video(self, predictor=None, session_id=None, max_frame_num_to_track=None):

        print()
        print(f"Sam3 inference. ... ♩ ♪ ♫ ♬")
        print(f"Prompt: {self.video_args.prompt}")
        print(f"Add box: {self.video_args.add_box}")
        print(f"Sub box: {self.video_args.sub_box}")
        print()

        predictor=self.predictor
        outputs = {}

        for response in predictor.handle_stream_request(

                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="forward",
                    output_prob_thresh = 0.1,
                    frame_idx=0,

                )):

            outputs[response["frame_idx"]] = response["outputs"]

        return outputs

    def abs_to_rel_coords(self, coords=None, IMG_WIDTH=None, IMG_HEIGHT=None, coord_type="box"):
        if coord_type == "point":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
        elif coord_type == "box":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT] for x, y, w, h in coords]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    def track(self, video_path = None, remove = False, sub_box = False, add_point = 0, warp=False):

        predictor, video_path, prompt, show_plots, add_box = self.predictor, self.video_path, self.video_args.prompt, self.video_args.show_plots, self.video_args.add_box

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
                frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            except ValueError:
                print(f'frame names are not in "<frame_idx>.jpg" format: {frames[:5]=}, '
                    f"falling back to lexicographic sort.")
                frames.sort()

        image = Image.fromarray(load_frame(frames[0]))
        W, H = image.size
        print(f'Image size: {W}x{H}')

        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
                offload_video_to_cpu=True,

            )
        )

        session_id = response["session_id"]
        is_success = predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        print(f'is_success: {is_success["is_success"]}')
        predictor.model.hotstart_delay = 0

        boxes = np.array([[0.2, 0.2, 0.6, 0.6], [0.7, 0.0, 0.3, 0.99], [0.0, 0.0, 0.3, 0.99], [0.1, 0.8, 0.7, 0.1]]) if add_box else None
        labels = np.array([1,0,0,0]) if add_box else None

        prompt_text = prompt if prompt is not None else None
        frame_idx = 0

        response = predictor.handle_request(
            request=dict(
                type = "add_prompt",
                session_id = session_id,
                frame_idx = frame_idx,
                text = prompt_text,
                bounding_boxes = boxes,
                bounding_box_labels = labels,
            )
        )

        frame_idx = response["frame_idx"]
        outputs = self.propagate_in_video(predictor, session_id)

        if show_plots:

            out = response["outputs"]
            outputs_per_frame = prepare_masks_for_visualization({frame_idx: out})
            vis_frame_stride = 2
            plt.close("all")
            for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
                visualize_formatted_frame_output(
                    frame_idx,
                    frames,
                    outputs_list=[outputs_per_frame],
                    titles=["SAM 3.1 Dense Tracking outputs"],
                    figsize=(6, 6))

        _ = predictor.handle_request(

            request=dict(
                type="close_session",
                session_id=session_id,
                run_gc_collect=True,
                clear_cache_threshold=100
            )
        )

        predictor.shutdown()
        return outputs

def sam3_video(frames_dir, mask_segments, video_args) -> None:

    output_size = video_args.mask_height
    folder = Path(frames_dir)
    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    if not image_files:
        return

    seq_dir = folder / "_sam3video_seq"
    video_frames_dir = folder / "_sam3video_frames"
    print(output_size)

    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    if video_frames_dir.exists():
        shutil.rmtree(video_frames_dir)

    video_frames_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)
    frame_shapes: list[tuple[int, int]] = []
    output_paths: list[Path] = []

    soft_masks = []
    valid_flags = []
    min_valid_pixels = int(output_size * 0.8)

    for i, frame_path in enumerate(image_files):
        out_path = frame_path.parent / f"{frame_path.stem}_mask.png"
        output_paths.append(out_path)
        raw = Image.open(frame_path)
        image = raw.convert("RGB")
        raw.close()

        frame_shapes.append((output_size, output_size))
        image.save(seq_dir / f"{i:06d}.jpg", format="JPEG", quality=100)
        image.close()

    tracker = sam3_video_inference(video_path=str(seq_dir), video_args=video_args)
    inference_state = tracker.track()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, out_path in enumerate(output_paths):
            out_h, out_w = frame_shapes[i]
            outputs = inference_state.get(i)
            masks = (outputs or {}).get("out_binary_masks", None)
            scores = (outputs or {}).get("out_probs", None)

            if masks is None:
                return None

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            else:
                scores = np.asarray(scores)

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
           
            else:
                masks = np.asarray(masks)

            if len(masks) == 0 or scores.size == 0:
                best_soft = np.zeros((out_h, out_w), dtype=np.float32)
                print(f"No SAM3 masks/scores for {out_path.name}; marking as missing for temporal fill")
            
            else:
                best_idx = int(np.argmax(scores))
                best_soft = masks[best_idx]
                if len(best_soft.shape) == 3:
                    best_soft = best_soft[0]
                best_soft = np.asarray(best_soft, dtype=np.float32)
                print("Confidence:", scores[best_idx])

            soft_masks.append(best_soft)
            valid_flags.append(np.count_nonzero(best_soft >= 0.5) >= min_valid_pixels)

    filled_masks, filled_count = fill_soft(soft_masks, valid_flags, max_interp_gap=6)
    if filled_count > 0:
        print(f"Filled {filled_count} missing/weak SAM3 masks using temporal soft-mask interpolation")
    
    for out_path, soft_mask in zip(output_paths, filled_masks):
        mask = Image.fromarray((np.clip((soft_mask - 0.5) * 10.0 + 0.5, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        
        mask.save(out_path)

    if seq_dir.exists():
        shutil.rmtree(seq_dir)

    del tracker, inference_state
    gc.collect()
    torch.cuda.empty_cache()

def fill_soft(
    soft_masks,
    valid_flags,
    max_interp_gap = 6,

):
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

def sam3_masks(
            video_args: argparse.Namespace,
            frames_dir: Path, 
            masks_dir: Path, 
            mask_segments: List[SegmentInfo], 
):

    sam3_video(str(frames_dir), mask_segments, video_args)

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

        if seg.sbs_frame_path:
            base = os.path.splitext(os.path.basename(seg.sbs_frame_path))[0]
            mask_src = frames_dir / f'{base}_mask.png'

            if mask_src.exists():
                final_mask_path = str(masks_dir / f'seg{seg.index:02d}_sbs_mask.png')
                shutil.move(str(mask_src), final_mask_path)
                seg.sbs_mask_path = final_mask_path

    return mask_segments

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

def _config_overrides(matanyone_model, job: dict, *, verbose: bool = False) -> None:

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
            f"long_term.max_mem_frames={cfg.long_term.max_mem_frames}\n")
        sys.stderr.flush()

def _elliptical_kernel(kernel_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    center = (kernel_size - 1) / 2.0
    radius = kernel_size / 2.0
    y, x = torch.meshgrid(coordinates, coordinates, indexing='ij')
    return ((x - center).square() + (y - center).square() <= radius * radius).to(dtype)

def _binary_mask(alpha: torch.Tensor) -> torch.Tensor:
    return (alpha > 127).to(alpha.dtype)

def gen_dilate(alpha: torch.Tensor, min_kernel_size: int, max_kernel_size: int) -> torch.Tensor:
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = _elliptical_kernel(kernel_size, device=alpha.device, dtype=alpha.dtype)
    binary = _binary_mask(alpha)
    dilated = F.conv2d(
        binary.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=kernel_size // 2)
    return (dilated[0, 0, :alpha.shape[-2], :alpha.shape[-1]] > 0).to(alpha.dtype) * 255

def gen_erosion(alpha: torch.Tensor, min_kernel_size: int, max_kernel_size: int) -> torch.Tensor:
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    if kernel_size <= 1:
        return _binary_mask(alpha)
    kernel = _elliptical_kernel(kernel_size, device=alpha.device, dtype=alpha.dtype)
    binary = _binary_mask(alpha)
    padded_foreground = F.pad(
        binary.unsqueeze(0).unsqueeze(0),
        (kernel_size // 2,) * 4,
        value=0)
    eroded = F.conv2d(padded_foreground, kernel.unsqueeze(0).unsqueeze(0))
    return (eroded[0, 0, :alpha.shape[-2], :alpha.shape[-1]] == kernel.sum()).to(alpha.dtype) * 255

def _matanyone_process_segment(matanyone_model, device, inference_core, job, video_args, data, save_image=False) -> str:
    n_warmup = video_args.warmup
    input_path = job['input_path']
    mask_path = job['mask_path']
    max_size = video_args.mask_height
    output_path = job['output_path']
    r_erode = video_args.erode
    r_dilate = video_args.dilate
    suffix = job.get('suffix', '')
    output_path = output_path if output_path is not None else TemporaryDirectory().name

    _config_overrides(matanyone_model, job, verbose=(job.get('op_num', 1) == 1))
    processor = inference_core(matanyone_model, cfg=matanyone_model.cfg)
    frames, fps, length, video_name = video_frames(input_path, max_size)
    
    frames = frames.float()
    repeated_frames = frames[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    frames = torch.cat([repeated_frames, frames], dim=0).float()
    length += n_warmup

    os.makedirs(output_path, exist_ok=True)
    if suffix:
        video_name = f"{video_name}_{suffix}"

    if save_image:
        os.makedirs(f"{output_path}/{video_name}", exist_ok=True)
        os.makedirs(f"{output_path}/{video_name}/pha", exist_ok=True)
        os.makedirs(f"{output_path}/{video_name}/fgr", exist_ok=True)

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float().to(device)

    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    if mask.shape != (max_size, max_size):
        print(f"Mask shape before interpolation: {mask.shape}")
        if max_size > 0:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(max_size, max_size), mode="nearest-exact"
            )[0, 0]
        print(f"Mask shape after interpolation: {mask.shape}")

    objects = [1]
    phas = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ti in tqdm.tqdm(range(length)):
            image = frames[ti]
            image = (image / 255.).float().to(device)

            if ti == 0:
                output_prob = processor.step(image, mask, objects=objects)
                output_prob = processor.step(image, first_frame_pred=True, force_permanent=False)

            elif ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True, force_permanent=False)

            else:
                output_prob = processor.step(image)

            mask = processor.output_prob_to_mask(output_prob, matting=True)
            pha = mask.unsqueeze(2).cpu().numpy()

            if ti > (n_warmup-1):
                pha = torch.round(mask * 255).to(torch.uint8)
                pha = torch.clamp(pha, 0, 255).cpu()
                phas.append(pha)

    output_file = os.path.join(output_path, f'{video_name}_pha.mp4')
    first_frame = phas[0]
    print(f"first_frame.shape: {first_frame.shape}")
 
    if first_frame.ndim == 3:
        first_frame = first_frame.squeeze(0)
    height, width = first_frame.shape

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', "-pix_fmt", "gray", '-r', str(fps),
        '-i', '-']

    if input_path:
        cmd.extend(['-i', input_path, '-map', '0:v', '-map', '1:a?'])

    cmd.extend([
        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp', '-c:v', 'hevc_nvenc', '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1', '-g', '20', '-b:v', '50M', '-c:a', 'aac', '-b:a', '256k', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags', '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709', output_file])

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        for pha in phas:
            process.stdin.write(pha.numpy().tobytes())
    finally:
        process.stdin.close()
        process.wait()

    return output_file

def matanyone_inference(jobs: list[dict], on_segment_done, video_args, data: dict) -> list[str]:
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

    matanyone_model, device, inference_core, loaded_version = _load_matanyone_runtime(version)
    if loaded_version != version:
        raise RuntimeError(f"Loaded model version mismatch: expected {version}, got {loaded_version}")

    for attempt in range(max_retries):
        batch_completed = []
        _matanyone_is_first_status = True

        try:
            for job in remaining_jobs:
                _update_status(job['op_num'], job['total_ops'], job['label'], job['duration'])
                output_file = _matanyone_process_segment(matanyone_model, device, inference_core, job, video_args, data)
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

def matanyone(
            video_args: argparse.Namespace,
            segments_dir: Path, 
            mask_segments: List[SegmentInfo], 
            segments: List[SegmentInfo],
            data: dict,
):
    print()
    print(f"MatAnyone inference. ... ♩ ♪ ♫ ♬")
    print(f"MatAnyone model: {video_args.matanyone_version}")

    matanyout = str(segments_dir / 'matanyone_output')
    os.makedirs(matanyout, exist_ok=True)
    mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
    total_ops = len(mask_segments) * 2

    jobs = []
    for seg in mask_segments:
        if not seg.left_mask_path or not seg.right_mask_path:
            sbs_video = str(segments_dir / f'seg{seg.index:02d}_sbs.mp4') 
            jobs.append({
                'input_path': sbs_video,
                'mask_path': seg.sbs_mask_path,
                'output_path': matanyout,
                'matanyone_version': video_args.matanyone_version,
                'ma2_mem_every': video_args.ma2_mem_every,
                'ma2_max_mem_frames': video_args.ma2_max_mem_frames,
                'ma2_use_long_term': video_args.ma2_use_long_term,
                'op_num': len(jobs) + 1,
                'total_ops': total_ops,
                'label': f'seg{seg.index:02d}_sbs',
                'duration': seg.end_time - seg.start_time,
                'data': data})
            
        else:
            seg_left_video = str(segments_dir / f'seg{seg.index:02d}_left.mp4')
            seg_right_video = str(segments_dir / f'seg{seg.index:02d}_right.mp4')
            jobs.append({
                'input_path': seg_left_video,
                'mask_path': seg.left_mask_path,
                'output_path': matanyout,
                'matanyone_version': video_args.matanyone_version,
                'ma2_mem_every': video_args.ma2_mem_every,
                'ma2_max_mem_frames': video_args.ma2_max_mem_frames,
                'ma2_use_long_term': video_args.ma2_use_long_term,
                'op_num': len(jobs) + 1,
                'total_ops': total_ops,
                'label': f'seg{seg.index:02d}_left',
                'duration': seg.end_time - seg.start_time,
                'data': data})
            jobs.append({
                'input_path': seg_right_video,
                'mask_path': seg.right_mask_path,
                'output_path': matanyout,
                'matanyone_version': video_args.matanyone_version,
                'ma2_mem_every': video_args.ma2_mem_every,
                'ma2_max_mem_frames': video_args.ma2_max_mem_frames,
                'ma2_use_long_term': video_args.ma2_use_long_term,
                'op_num': len(jobs) + 1,
                'total_ops': total_ops,
                'label': f'seg{seg.index:02d}_right',
                'duration': seg.end_time - seg.start_time,
                'data': data})

    completed_paths = matanyone_inference(jobs, None, video_args, data)

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

def process_video(video_path, args: argparse.Namespace, temp_root: Path) -> str:
    
    video_path = str(Path(video_path).expanduser().resolve())
    video_name = Path(video_path).stem
    data = info(video_path)
    print(f"Specs: {data['width']}x{data['height']} @ {data['fps']}fps, duration: {data['duration']}, format: {data['pix_fmt']}, frames: {data['frames']}")
    print()
    
    safe_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in video_name)
    temp_dir = temp_root / safe_name

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    frames_dir = temp_dir / 'frames'
    masks_dir = temp_dir / 'masks'
    segments_dir = temp_dir / 'segments'

    for d in [frames_dir, masks_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    video_args = argparse.Namespace(**vars(args), video=video_path)
    alpha_output = video_args.alpha
    overlay_mask = video_args.overlay_mask

    if overlay_mask is None:
        segments = calculate_segments(
            data['duration'],
            video_args.segment_length,
            debug=video_args.debug,
            )

        mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]

        mask_segments = extract_segments(
            video_args,
            frames_dir, 
            segments_dir, 
            mask_segments, 
            segments,
            data,
        )

        mask_segments = sam3_masks(
            video_args,
            frames_dir, 
            masks_dir, 
            mask_segments, 
        )

        segments = matanyone(
            video_args,
            segments_dir, 
            mask_segments, 
            segments,
            data,
            )

        output_mask = finalize(
            segments,
            video_name,
            str(video_path),
        )

        if alpha_output:
            alpha_video = packer(
                video_path, 
                video_args
                )
      
        else:
            output_path = str(Path(video_path).with_name(f"{video_name}_overlay.mp4"))
            overlay_video = mask_overlay(
                video_path,
                output_mask,
                output_path,
                background_color=video_args.overlay_color,
                video_args=video_args,
            )

            overlay_data = info(output_path)
            print(f"Overlay video info: {overlay_data['width']}x{overlay_data['height']} @ {overlay_data['fps']}fps, duration: {overlay_data['duration']}, format: {overlay_data['pix_fmt']}, frames: {overlay_data['frames']}")
        print('=' * 60)
        print(f'Segments: {len(segments)} ({len(mask_segments)} masks) - Output: {output_mask}')
        print()

        with open(temp_dir / 'segments.txt', 'w', encoding='utf-8') as f:
            f.write(f'# {video_name}\n')
            for seg in segments:
                f.write(f'{seg.index},{seg.seg_type.value},{seg.start_time:.3f},{seg.end_time:.3f},{seg.video_path}\n')

        print(f"info() cache: {info.cache_info()}")
        return output_mask

    else:
        overlay_target = str(Path(video_path).with_name(f"{video_name}_overlay.mp4"))
        overlay_video = mask_overlay(
            video_path,
            overlay_mask,
            overlay_target,
            background_color=video_args.overlay_color,
            video_args=video_args,
            data=data,
            )

        print(f'Overlay preview: {overlay_video}')
        print('=' * 60)
        return aorb(overlay_video, alpha_video)

def calculate_segments(video_duration: float, max_segment_length: float = 5.0, debug = None) -> List[SegmentInfo]:

    segments: List[SegmentInfo] = []
    chunk_start = 0.0
    index = 0
    while chunk_start < (video_duration if debug is None else debug):
        chunk_end = min(chunk_start + max_segment_length, video_duration)
        if 0 < video_duration - chunk_end < 0.1:
            chunk_end = video_duration
        segments.append(SegmentInfo(index=index, start_time=chunk_start,
                end_time=chunk_end, seg_type=SegmentType.MASK))
        index += 1
        chunk_start = chunk_end
    return segments

def extract_segments(
            video_args: argparse.Namespace,
            frames_dir: Path, 
            segments_dir: Path, 
            mask_segments: List[SegmentInfo], 
            segments: List[SegmentInfo],
            data: dict,

) -> List[SegmentInfo]:

    print(f'Total: {len(segments)} segments')
    for seg in segments:
        dur = seg.end_time - seg.start_time
        print(f'[{seg.index}] {seg.seg_type.value.upper():5} '
            f'{format_timestamp(seg.start_time)} → {format_timestamp(seg.end_time)} ({dur:.1f}s)')
    print()

    for i, seg in enumerate(mask_segments) if video_args.debug is None else enumerate(mask_segments[:video_args.debug]):
        
        if video_args.sbs:
            sbs_frame = str(frames_dir / f'seg{seg.index:02d}_sbs.png')
            sbs_video = str(segments_dir / f'seg{seg.index:02d}_sbs.mp4')
            sbs_frame_path, _ = extract_segment_sbs(
                        stereo_video=video_args.video,
                        start=seg.start_time,
                        end=seg.end_time,
                        target_height=video_args.mask_height,
                        sbs_frame_out = sbs_frame,
                        sbs_video_out = sbs_video,
                        progress_prefix=f'[{i + 1}/{len(mask_segments)}]')
            seg.sbs_frame_path = sbs_frame_path

        else:
            left_frame = str(frames_dir / f'seg{seg.index:02d}_left.png')
            right_frame = str(frames_dir / f'seg{seg.index:02d}_right.png')
            seg_left_video = str(segments_dir / f'seg{seg.index:02d}_left.mp4')
            seg_right_video = str(segments_dir / f'seg{seg.index:02d}_right.mp4')

            left_frame_path, right_frame_path, _, _ = extract_segment_frames(
                stereo_video=video_args.video,
                start=seg.start_time,
                end=seg.end_time,
                height=data['height'],
                target_height=video_args.mask_height,
                left_frame_out=left_frame,
                right_frame_out=right_frame,
                left_video_out=seg_left_video,
                right_video_out=seg_right_video,
                progress_prefix=f'[{i + 1}/{len(mask_segments)}]')
            seg.left_frame_path = left_frame_path
            seg.right_frame_path = right_frame_path

    return mask_segments

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
    parser = argparse.ArgumentParser(description="VR Video Masking and things and stuff")
    parser.add_argument("--model", type=str, default="sam3.1")
    parser.add_argument("input_path", type=str, default="videos")
    parser.add_argument("--mask-height", type=int, default=1280)
    parser.add_argument("--segment-length", type=float, default=2)
    parser.add_argument("--erode", type=int, default=6)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="agirl")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--add-box", type=bool, default=False)
    parser.add_argument("--sub-box", type=bool, default=False)
    parser.add_argument("--sbs", type=bool, default=False)
    parser.add_argument('--matanyone-version', type=str, default='v2', choices=['v1', 'v2'], help='Select MatAnyone runtime version')
    parser.add_argument('--ma2-mem-every', type=int, default=2, help='Override MatAnyone mem_every (works for v1 and v2; e.g. 2 or 3 for faster refresh)')
    parser.add_argument('--ma2-max-mem-frames', type=int, default=2, help='Override MatAnyone memory window in frames (works for v1 and v2)')
    parser.add_argument('--ma2-use-long-term', type=str, default='off', choices=['auto', 'on', 'off'], help='Override MatAnyone long-term memory ')
    parser.add_argument('--overlay-output', type=str, default='input_path', help='Write a composited video with the mask over the original source')
    parser.add_argument('--overlay-color', type=str, default='0x00ff00', help='Background color for overlay (use 0x00ff00 for pure green)')
    parser.add_argument('--overlay-mask', type=str, default=None, help='Write a composited video with a provided mask over the original source')
    parser.add_argument('--alpha-packer', type=bool, default=False, help='Run alpha packer. Provide folder with video and mask (_mask.<ext>) for  input_path')
    parser.add_argument('--decompose-alpha', '--decompose_alpha', dest='decompose_alpha', action='store_true', help='Reverse of alpha packer')
    parser.add_argument('--decompose-clean-mask', type=str, default='assets/black_mask.png', help='PNG overlay used to clean alpha payload regions')
    parser.add_argument('--alpha', type=bool, default=False, help='Run alpha packer instead of overlay. --alpha <true|false>')
    parser.add_argument('--show-plots', type=bool, default=False, help='Sam3 mask plots will be displayed if True.')
    parser.add_argument('--fisheye180', type=bool, default=False, help='Convert video or folder to SBS fisheye180. Works with alphapacker')
    parser.add_argument('--debug', type=int, default=None, help='Debug mode: process only the first N segments')
    args = parser.parse_args()
    args.matanyone_version = str(args.matanyone_version).lower()

    if args.ma2_mem_every is not None and args.ma2_mem_every < 1:
        raise ValueError('--ma2-mem-every must be >= 1')
    if args.ma2_max_mem_frames is not None and args.ma2_max_mem_frames < 2:
        raise ValueError('--ma2-max-mem-frames must be >= 2')
    if args.ma2_use_long_term == 'auto':
        args.ma2_use_long_term = None
    else:
        args.ma2_use_long_term = (args.ma2_use_long_term == 'on')

    if args.alpha_packer:
        return packer(args.input_path, fisheye=args.fisheye180)
    if args.decompose_alpha:
        cleanup_mask = args.decompose_clean_mask
        if cleanup_mask is not None and str(cleanup_mask).strip().lower() in {'none', 'off', 'false', '0'}:
            cleanup_mask = None
        return decompose_alpha(args.input_path, cleanup_mask_path=cleanup_mask)

    video_paths = _input_videos(args.input_path)
    temp_root = Path('temp_pipeline')
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    processed = []
    batch_mode = len(video_paths) > 1
    for index, video_path in enumerate(video_paths, 1):
        video_path = str(video_path)
        video_args = argparse.Namespace(**vars(args), video=video_path)
        is_vfr = check_vfr(video_path)
        if is_vfr:
            video_path = cfr_video(video_path, video_args) 
        output_mask = process_video(video_path, args, temp_root)
        processed.append((video_path, output_mask))
    for video_path, output_mask in processed:
        print(f'{video_path}')
        print(f'{output_mask}')
    total_end = time.time() - start_time
    print('=' * 60)
    print(f"Total time: {total_end:.2f}s")
    print(f"info() cache: {info.cache_info()}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
