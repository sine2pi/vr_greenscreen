import sys, functools, time, tqdm, random, shutil, gc, os, torch, cv2, numpy as np, glob, matplotlib.pyplot as plt, torch.nn.functional as F, argparse, re, subprocess, json, threading
from PIL import Image, ImageDraw, ImageFilter
import tempfile
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
MASK_EXT = '.mkv'
MATANYONE_V1 = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
MATANYONE_V2 = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
FFMPEG_BIN = shutil.which('ffmpeg') or 'ffmpeg'
FFPROBE_BIN = shutil.which('ffprobe') or 'ffprobe'

class StageTimer:
    """Prints elapsed time since the previous mark; no-op unless enabled (tied to --debug)."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.last = time.perf_counter()

    def mark(self, label: str):
        now = time.perf_counter()
        if self.enabled:
            print(f'[timing] {label}: {now - self.last:.2f}s')
        self.last = now

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

class VideoMetadataManager:
    _instance = None
    data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VideoMetadataManager, cls).__new__(cls)
        return cls._instance

    def set_metadata(self, ffprobe_dict):
        self.data = ffprobe_dict

    def get_metadata(self):
        return self.data
        
    def clear(self):
        self.data = {}

def load_frame(frame):
    if isinstance(frame, np.ndarray):
        img = frame
    elif isinstance(frame, Image.Image):
        img = np.array(frame)
    elif isinstance(frame, (str, os.PathLike)) and os.path.isfile(frame):
        with Image.open(frame) as im:
            img = np.array(im.convert("RGB"))
    else:
        raise ValueError(f"Invalid video frame type: {type(frame)=}")
    return img

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
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)

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

def lossless_args(pix_fmt: str = 'gray') -> list[str]:
    return ['-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1', '-g', '1',
            '-slices', '16', '-slicecrc', '1', '-pix_fmt', pix_fmt, '-an']

def encoder_args(data=None, audio: bool = True) -> list[str]:
    fps = data['fps'] if data is not None else 60

    codec_args = [
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
    ]

    audio_args = ['-map', '0:a?', '-c:a', 'copy'] if audio else ['-an']

    return [

        '-fps_mode', 'cfr',
        '-r', str(fps),
        *codec_args,
        *audio_args,
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-metadata:s:v:0', 'stereo_mode=left_right',
        '-movflags', '+faststart+write_colr+use_metadata_tags',
    ]

def ffmpeg_progress(cmd: list[str], progress_prefix: str = "", cwd: str | None = None) -> tuple[int, str]:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    stderr_lines = []
    for line in process.stderr:
        stderr_lines.append(line)
    process.wait()

    return process.returncode, "".join(stderr_lines)

def final_encode(input_args: list[str], filter_complex: str, output_path: str, data: dict, duration=None,
                 progress_prefix: str = "", what: str = "Encode", extra_args: list[str] | None = None) -> str:
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    cmd = [FFMPEG_BIN, '-y', '-hwaccel', 'auto', *(extra_args or []), *input_args,
           '-filter_complex', filter_complex, '-map', '[out]',
           *(['-t', str(duration)] if duration else []),
           *encoder_args(data), output_path]

    rc, stderr_text = ffmpeg_progress(cmd, progress_prefix=progress_prefix)
    if rc != 0:
        raise RuntimeError(f"{what} failed.\n\nFFmpeg tail:\n{''.join(stderr_text.splitlines(True)[-40:])}")
    if not os.path.exists(output_path):
        raise RuntimeError(f"{what} failed: {output_path} was not created")
    return output_path

@functools.lru_cache(maxsize=256)
def info(path, ffprobe_bin=FFPROBE_BIN):
    metadata = {}
    cmd_stream = [ffprobe_bin, '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', '-select_streams', 'v:0', path]
    res_stream = subprocess.run(cmd_stream, capture_output=True, text=True)
    data = json.loads(res_stream.stdout)

    if not data.get('streams'):
        return None
        
    stream = data['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    codec = stream.get('codec_name', '') 
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    duration = _num(stream.get('duration')) or _num(data.get('format', {}).get('duration'))
    pix_fmt = stream.get('pix_fmt', 'yuv420p')
    fps_str = stream.get('r_frame_rate', '60/1')
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom if denom != 0 else 60.0
    f_tot = stream.get('nb_frames')

    if f_tot:
        frames = int(f_tot)
    else:
        frames = round(duration * fps) if duration > 0 else 0
        
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

        'ffmpeg', '-y', '-hwaccel', 'auto',
        '-i', source_video,
        '-filter_complex', f'[0:v]fps={fps},setpts=N/({fps}*TB),scale=w={wi}:h={hi}:flags=bilinear:out_range=pc:threads=0',
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

def cfr_video(source_video) -> str:

    data = info(source_video)
    print(f"-- {source_video} has a Variable Frame Rate - Converting to CFR")
    
    source_path = Path(source_video).expanduser().resolve()
    output_video = str(source_path.with_name(f"{source_path.stem}_CFR.mp4"))

    fps = aorb(data['fps'], 60)
    fps = normalize_fps(fps)

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'auto',
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

    rc, stderr_text = ffmpeg_progress(cmd, progress_prefix = "[cfr]")

    if rc != 0:
        raise RuntimeError(
            "CFR conversion failed.\n\nFFmpeg tail:\n"
            + ''.join(stderr_text.splitlines(True)[-40:])
        )

    if not os.path.exists(output_video):
        raise RuntimeError(f"CFR video not created: {output_video}")

    return output_video

def resize_video(source_video, output_video, progress_prefix: str = "[resize] ") -> str:

    data = info(source_video)
    enc = encoder_args(data)
    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or '.', exist_ok=True)

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'auto',
        '-i', source_video,
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

def eye_frames(video_path: str, timestamps: list[float], output_dir: str, height: int) -> list[str]:
    data = info(video_path)
    enc = encoder_args(data)
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
            *enc,
            out_path
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.wait()

        if process.returncode == 0 and os.path.exists(out_path):
            output_paths.append(out_path)

    return output_paths

def list_keyframes(path: str, ffprobe_bin=FFPROBE_BIN) -> list[float]:
   
    cmd = [ffprobe_bin, '-v', 'quiet', '-select_streams', 'v:0',
           '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0', path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    times = []
    for line in res.stdout.splitlines():
        parts = line.strip().split(',')
        if len(parts) >= 2 and 'K' in parts[1]:
            try:
                times.append(float(parts[0]))
            except ValueError:
                pass
    return sorted(set(times))

def _segment_chain(data: dict, start: float, end: float, w: int, h: int, keyframe_start: bool = False):
    fps = data['fps']
    frames = round(end * fps) - round(start * fps)
    if frames <= 0:
        raise RuntimeError(f"Invalid segment: {start=} {end=} {fps=} -> {frames} frames")

    aligned_start = round(start * fps) / fps
    if keyframe_start:
        seek, fine = max(0.0, aligned_start - 0.5 / fps), 0.0
    else:
        seek = max(0.0, aligned_start - 2.0)
        fine = aligned_start - seek

    vf = (f"trim=start={fine}:duration={frames / fps},setpts=PTS-STARTPTS,fps={fps},"
          f"scale={w}:{h}:flags=bilinear:in_color_matrix=bt709,format=rgb24")
    return seek, vf, frames

def decode_segment(video_path: str, start: float, end: float, w: int, h: int, keyframe_start: bool = False) -> torch.Tensor:
    data = info(video_path)
    seek, vf, frames = _segment_chain(data, start, end, w, h, keyframe_start)
    cmd = [FFMPEG_BIN, '-v', 'error', '-hwaccel', 'auto', '-ss', str(seek), '-i', str(video_path),
           '-vf', vf, '-frames:v', str(frames), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']

    buf = torch.empty((frames, h, w, 3), dtype=torch.uint8)
    view = memoryview(buf.numpy()).cast('B')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got = 0
    chunk = 1 << 20
    while got < len(view):
        n = proc.stdout.readinto(view[got:got + chunk])
        if not n:
            break
        got += n
    proc.stdout.close()
    err = proc.stderr.read().decode(errors='ignore')
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Segment decode failed.\n\nFFmpeg tail:\n{err[-2000:]}")

    n_frames = got // (w * h * 3)
    if n_frames == 0:
        raise RuntimeError(f"Segment decode produced no frames: {video_path} {start}-{end}")
    return buf[:n_frames].permute(0, 3, 1, 2)

def extract_first_frame(video_path: str, start: float, end: float, w: int, h: int, outputs: list[tuple[str, str]], keyframe_start: bool = False) -> None:
    data = info(video_path)
    seek, vf, _ = _segment_chain(data, start, end, w, h, keyframe_start)
    vf = vf.replace(',format=rgb24', '')
    parts = ';'.join(f"[f{i}]{crop + ',' if crop else ''}null[o{i}]" for i, (_, crop) in enumerate(outputs))
    split = f"[0:v]{vf},select=eq(n\\,0),split={len(outputs)}" + ''.join(f"[f{i}]" for i in range(len(outputs)))
    cmd = [FFMPEG_BIN, '-y', '-v', 'error', '-hwaccel', 'auto', '-ss', str(seek), '-i', str(video_path),
           '-filter_complex', f"{split};{parts}"]
    for i, (path, _) in enumerate(outputs):
        cmd += ['-map', f'[o{i}]', '-frames:v', '1', '-compression_level', '1', path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"First-frame extraction failed: {result.stderr[-2000:]}")

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

def mask_overlay(source_video: str, mask_video: str, output_path: str, background_color: str = '0x00ff00', video_args: argparse.Namespace = None, data: dict = None) -> str:

    resolved_path = overlay_path(source_video, output_path)
    os.makedirs(os.path.dirname(os.path.abspath(resolved_path)) or '.', exist_ok=True)

    if data is None:
        data = info(source_video)

    duration = data["duration"] if video_args.debug is None else video_args.debug

    os.makedirs(os.path.dirname(os.path.abspath(resolved_path)) or '.', exist_ok=True)

    orig_filter = f"setpts=PTS-STARTPTS,fps={data['fps']},format=yuva420p"
    mask_filter = f"setpts=PTS-STARTPTS,fps={data['fps']},format=gray,scale={data['width']}:{data['height']}:flags=bilinear+accurate_rnd"
    bg_filter = f"setpts=PTS-STARTPTS,fps={data['fps']},format=yuv420p"

    filter_complex = (
        f"[0:v]{orig_filter}[orig];"
        f"[1:v]{mask_filter}[mask_alpha];"
        f"[orig][mask_alpha]alphamerge[alphaed];"
        f"[2:v]{bg_filter}[bg];"
        f"[bg][alphaed]overlay=format=yuv420[out]"
    )

    final_encode(
        ['-i', source_video, '-i', mask_video,
         '-f', 'lavfi', '-i', f'color=c={background_color}:s={data["width"]}x{data["height"]}:d={duration}:r={data["fps"]}'],
        filter_complex, resolved_path, data, duration=duration, what="Mask overlay")
    return resolved_path

def get_video_paths(input_root):
    video_paths = []

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))

    return sorted(video_paths)

def _ceil_to(n: int, base: int) -> int:
    return ((n + base - 1) // base) * base

@functools.lru_cache(maxsize=8)
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

def _mask_for(video: Path) -> Path | None:
    for candidate in (video.with_name(f"{video.stem}_mask{MASK_EXT}"), video.with_name(f"{video.stem}_mask{video.suffix}")):
        if candidate.exists():
            return candidate
    return None

def _input_pairs(input_path: str) -> list[tuple[Path, Path]]:
    path = Path(str(input_path)).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video file: {path}")

        mask_path = _mask_for(path)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {path}: expected {path.stem}_mask{MASK_EXT}")
        return [(path, mask_path)]

    if not path.is_dir():
        raise RuntimeError(f"Input path is not a file or folder: {input_path}")

    pairs: list[tuple[Path, Path]] = []
    for candidate in sorted(path.rglob('*')):
        if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if candidate.stem.endswith('_mask'):
            continue
        mask_path = _mask_for(candidate)
        if mask_path is not None:
            pairs.append((candidate.resolve(), mask_path.resolve()))
    if not pairs:
        raise RuntimeError(f"No original/mask video pairs found in folder: {input_path}")
    return pairs

def fisheye_chain(src: str, dst: str, w: int, h: int, fps: float, png: str, tail: str = '') -> list[str]:
    eye_w = w // 2
    v360 = f'v360=input=hequirect:output=fisheye:iv_fov=180:ih_fov=180:v_fov=180:h_fov=180:w={eye_w}:h={h}'
    return [
        f'{src}fps={fps},setpts=N/({fps}*TB),split=2[{dst}_ls][{dst}_rs]',
        f'[{dst}_ls]crop=iw/2:ih:0:0,{v360}[{dst}_l]',
        f'[{dst}_rs]crop=iw/2:ih:iw/2:0,{v360}[{dst}_r]',
        f'[{dst}_l][{dst}_r]hstack,scale=w={w}:h={h}:flags=bilinear[{dst}_st]',
        f'{png}[{dst}_st]scale2ref[{dst}_m][{dst}_ref]',
        f'[{dst}_ref][{dst}_m]overlay=0:0:format=auto{tail}[{dst}]',
    ]

def pack_video(
    video_path: str,
    mask_path: str,
    output_path: str | None = None,
    sync_frames = None,
    progress_prefix: str = "[ALPHA] ",
    erode: int = 1,
    blur: float = 1.8,
    contrast: float | None = None,
    gamma: float | None = None,
    ref_size: int = 1024,
    tmix: int = 1,
    video_args = None,
    fisheye: bool = False,

) -> str:

    if tmix < 1 or tmix % 2 == 0:
        raise ValueError(f"alpha tmix must be an odd number >= 1, got {tmix}")

    if not output_path:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}{'_FISHEYE180' if fisheye else ''}_alpha{ext}"

    actual_mask = mask_path
    synced_tmp = None

    if sync_frames is not None:
        fps = info(mask_path)['fps']
        synced_tmp = sync_mask_to_video(mask_path, fps=fps, frame_offset=sync_frames)
        actual_mask = synced_tmp
    
    data = info(video_path)

    out_h = _ceil_to(data['height'], 32)

    if data['width'] == 2 * data['height']:
        out_w = 2 * out_h
    else:
        out_w = _ceil_to(data['width'], 32)

    if (out_w, out_h) != (data['width'], data['height']):
        print(f"NVENC-aligned output: {out_w}x{out_h}")

    overlay_size = int(out_h * 0.4)
    overlay_size = (overlay_size // 4) * 4
    half_overlay = overlay_size // 2
    print(f"Half overlay size: {half_overlay}, overlay size: {overlay_size}, out_w: {out_w}, out_h: {out_h}")

    if data['height'] <= 2400:
        default_contrast, default_gamma = 2.0, 1.2
    else:
        default_contrast, default_gamma = 2.5, 1.4

    contrast = default_contrast if contrast is None else contrast
    gamma = default_gamma if gamma is None else gamma

    size_scale = overlay_size / ref_size if ref_size > 0 else 1.0
    sigma = blur * size_scale
    erode_iters = max(1, round(erode * size_scale)) if erode > 0 else 0

    shaping = "erosion," * erode_iters
    if sigma > 0:
        shaping += f"gblur=sigma={sigma:.3f},"
    if contrast != 1.0 or gamma != 1.0:
        shaping += f"eq=contrast={contrast}:gamma={gamma},"

    if fisheye:
        black_png = str(Path('assets/black_mask.png').expanduser().resolve())
        if not os.path.exists(black_png):
            raise FileNotFoundError(f'Fisheye mask not found: {black_png}')
        mdata = info(actual_mask)
        fisheye_parts = [
            "[3:v]format=rgba,split=2[bm_v][bm_m]",
            *fisheye_chain("[0:v]", "fv", data['width'], data['height'], data['fps'], "[bm_v]"),
            *fisheye_chain("[1:v]", "fm", mdata['width'], mdata['height'], mdata['fps'], "[bm_m]", tail=",format=gray"),
        ]
        vid_src = "[fv]"
        mask_src = "[fm]"
        extra_inputs = ["-i", black_png]
    else:
        fisheye_parts, extra_inputs = [], []
        vid_src, mask_src = "[0:v]", "[1:v]"

    if tmix > 1:
        mask_prep = f"{mask_src}tmix=frames={tmix},trim=start_frame={(tmix - 1) // 2},setpts=PTS-STARTPTS,split=2[mask1][mask2]"
    else:
        mask_prep = f"{mask_src}split=2[mask1][mask2]"

    print(f"Mask Gen Params: erode={erode_iters}, gblur={sigma:.2f}, contrast={contrast}, gamma={gamma}, tmix={tmix} (payload scale x{size_scale:.2f})")
    print(f"Adjusted overlay size: {overlay_size}")
    circle_mask = get_circle_mask(overlay_size)

    filter_parts: list[str] = [

        *fisheye_parts,
        f"{vid_src}scale=w={out_w}:h={out_h}:flags=bilinear[vid]",
        mask_prep,
        "[2:v]format=gray,split=2[circle_l][circle_r]",

        (
            f"[mask1]crop=ih:ih:0:0,"
            f"scale={overlay_size}:{overlay_size}:flags=area,"
            f"{shaping}"
            "format=gbrp[left_scaled]"
        ),
        "[left_scaled][circle_l]alphamerge,format=rgba[left_circle]",

        (
            f"[mask2]crop=ih:ih:iw-ih:0,"
            f"scale={overlay_size}:{overlay_size}:flags=area,"
            f"{shaping}"
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
    duration = data["duration"] if video_args.debug is None else video_args.debug
    final_encode(
        ["-i", video_path, "-i", actual_mask, "-loop", "1", "-i", circle_mask, *extra_inputs],
        filter_complex, output_path, data, duration=duration,
        progress_prefix=progress_prefix, what="Alpha pack",
        extra_args=["-filter_threads", "0", "-threads", "0"],
    )
    if synced_tmp and os.path.exists(synced_tmp):
        os.remove(synced_tmp)
    print(f"Alpha packed: {output_path}")
    return output_path

def _alpha_pack_params(video_args) -> dict:
    return {
        'erode': getattr(video_args, 'alpha_erode', 1),
        'blur': getattr(video_args, 'alpha_blur', 1.8),
        'contrast': getattr(video_args, 'alpha_contrast', None),
        'gamma': getattr(video_args, 'alpha_gamma', None),
        'ref_size': getattr(video_args, 'alpha_ref_size', 1024),
        'tmix': getattr(video_args, 'alpha_tmix', 1),
    }

def packer(input_path, video_args=None, mask_path=None) -> int:

    input_pairs = [(Path(input_path), Path(mask_path))] if mask_path else _input_pairs(input_path)
    processed = []
    
    for (video_path, mask_path) in input_pairs:
        packed_path = pack_video(str(video_path), str(mask_path), video_args=video_args,
                                 fisheye=bool(video_args.fisheye180), **_alpha_pack_params(video_args))
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
    mask_output = packed_video.with_name(f'{base_stem}_mask{MASK_EXT}')
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
        final_encode(['-i', str(packed_path), '-i', str(cleanup_mask)], clean_filter, str(video_out), data,
                     progress_prefix=f"{progress_prefix}[VIDEO] ", what="Video cleanup")
    else:
        copy_cmd = [
            'ffmpeg', '-y', '-hwaccel', 'auto',
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
        "[mask_bg][left_eye]overlay=0:0[mask_l]",
        f"[mask_l][right_eye]overlay={right_eye_x}:0[mask_comp]",
        "[mask_comp]scale=in_range=tv:out_range=pc,format=gray[out]",
    ]

    mask_cmd = [
        'ffmpeg', '-y', '-hwaccel', 'auto',
        '-i', str(packed_path),
        '-filter_complex', ';'.join(filter_parts),
        '-map', '[out]',
        '-r', str(data['fps']),
        *lossless_args('gray'),
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

    frame_duration = abs(frame_offset) / fps
    if frame_offset > 0:
        vf = f"trim=start={frame_duration},setpts=PTS-STARTPTS"
    elif frame_offset < 0:
        vf = f"tpad=start_duration={frame_duration}:color=black"
    else:
        vf = "null"

    base, _ = os.path.splitext(mask_path)
    synced_path = f"{base}_synced{MASK_EXT}"

    cmd = [

        'ffmpeg', '-y',
        '-i', mask_path,
        '-vf', f"{vf},format=gray",
        *lossless_args('gray'),
        synced_path,
    ]

    ffmpeg_progress(cmd)
    return synced_path

def fisheye180(input_video: str, flag=False) -> str:
    data = info(input_video)
    enc = lossless_args('gray') if flag else encoder_args(data)

    input_video = str(Path(input_video).expanduser().resolve())
    filename, ext = os.path.splitext(input_video)
    output_video = f'{filename}_FISHEYE180{ext}' if not flag else f'{filename}_FISHEYE180_mask{MASK_EXT}'
    
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
    out_label = '[out]'
    if flag:
        filter_complex += ';[out]format=gray[outg]'
        out_label = '[outg]'

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'auto',
        '-i', input_video,
    ]

    if mask_png is not None:
        cmd.extend(['-i', mask_png])

    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', out_label,
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

class sam3_video_inference:
    @staticmethod
    def download_ckpt(version="sam3", force_download=False, local_files_only=False, token=None):
        from huggingface_hub import hf_hub_download

        if version == "sam3.1":
            repo_id, ckpt_name = "sin2piusc/sam31sin", "sam3.1_multiplex.pt"
        elif version == "sam3lite":
            repo_id, ckpt_name = "vil-uob/sam3-litetext-l", "model.safetensors"
        elif version == "sam3image":
            repo_id, ckpt_name = "sin2piusc/sam3_fta", "sam3.pth"
        elif version == "sam3m":
            repo_id, ckpt_name = "feyninc/multimatte", "model.safetensors"
        elif version == "local":
            return r"sam3/sam3.pt"
        else:
            repo_id, ckpt_name = "facebook/sam3", "sam3.pt"

        return hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_name,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )

    def __init__(self, video_path, video_args, frames=None):

        self.video_path = video_path
        self.frames = frames
        self.video_args = video_args

        bpe_path = 'assets/bpe_simple_vocab_16e6.txt.gz'
        checkpoint_path = self.download_ckpt(version=video_args.model, force_download=False, local_files_only=False)

        self.predictor = build_sam3_predictor(
            checkpoint_path = checkpoint_path,
            bpe_path = bpe_path,
            version = video_args.model,
            compile = False,
            warm_up = False,
            max_num_objects = 1,
            multiplex_count = 16,
            use_fa3 = False,
            use_rope_real = True,
            async_loading_frames = False,
            num_obj_for_compile=1,
            apply_temporal_disambiguation=True,
            device = device,
            video_loader_type="cv2",
            load_from_HF=False,
            default_output_prob_thresh=0.2, 
            strict_state_dict_loading=False, 
            session_expiration_sec=1200, 
            eval_mode=True, 
       
        )

        self.predictor.model.hotstart_delay = 0
        self.predictor.model.suppress_unmatched_only_within_hotstart = False
        self.predictor.model.suppress_det_close_to_boundary=True
        self.predictor.model.suppress_overlapping_based_on_recent_occlusion_threshold=0.7
        self.predictor.model.allow_unoccluded_to_suppress = False
        self.predictor.model.decrease_trk_keep_alive_for_empty_masklets=True
        self.predictor.model.max_trk_keep_alive = 600
        self.predictor.model.new_det_thresh=0.0
        self.predictor.model.fill_hole_area=16
        self.predictor.model.sprinkle_removal_area=8
        self.predictor.model.is_multiplex = True
        self.predictor.model.masklet_confirmation_enable = True
        self.predictor.model.masklet_confirmation_consecutive_det_thresh=3

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
                    max_frame_num_to_track=max_frame_num_to_track,

                )):

            self.clear_ignored_masks(response["outputs"])
            outputs[response["frame_idx"]] = response["outputs"]

        return outputs

    def _ignore_fracs(self):
        fracs = {}
        
        for side in ("bottom", "left", "right"):
            frac = float(getattr(self.video_args, f"ignore_{side}", 0.0) or 0.0)
            
            if not 0.0 <= frac < 1.0:
                raise ValueError(f"--ignore-{side} must be in [0, 1), got {frac}")
            fracs[side] = frac
        
        if fracs["left"] + fracs["right"] >= 1.0:
            raise ValueError("--ignore-left and --ignore-right must sum to less than 1")
        return fracs

    @staticmethod
    def _blank_region(arr, fracs, value, static=None):

        if isinstance(arr, Image.Image):
            arr = np.array(arr)

        h, w = arr.shape[-2], arr.shape[-1]
        if fracs["bottom"] > 0:
            arr[..., int(round(h * (1.0 - fracs["bottom"]))):, :] = value
        
        if fracs["left"] > 0:
            arr[..., :, :int(round(w * fracs["left"]))] = value
        
        if fracs["right"] > 0:
            arr[..., :, int(round(w * (1.0 - fracs["right"]))):] = value
        
        if static is not None:
            if torch.is_tensor(static):
                static = static.cpu().numpy()

            if static.shape != (h, w):
                static = cv2.resize(static.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            static = static.astype(bool)
            
            if torch.is_tensor(arr):
                static = torch.from_numpy(static).to(arr.device)
            arr[..., static] = value
        return arr

    @staticmethod
    def _has_ignore(fracs, static):
        return static is not None or any(fracs.values())

    @torch.inference_mode()
    def _compute_static_mask(self, tensors):

        thresh = float(getattr(self.video_args, "ignore_static", 0.0) or 0.0)
        if thresh <= 0:
            return None
        num_frames = len(tensors)

        if num_frames < 1:
            print(f"--ignore-static skipped: needs at least 1 frame, got {num_frames}")
            return None

        std = float(np.mean(getattr(self.predictor.model, "image_std", (0.5, 0.5, 0.5))))
        scan_device = "cuda" if torch.cuda.is_available() else "cpu"
        lo = hi = None

        for i in range(0, num_frames, max(1, num_frames // 32)):
            frame = tensors[i].to(scan_device).float()
            lo = frame.clone() if lo is None else torch.minimum(lo, frame)
            hi = frame.clone() if hi is None else torch.maximum(hi, frame)
        static = ((hi - lo).amax(0) * std < thresh).cpu().numpy()

        _, labels = cv2.connectedComponents(static.astype(np.uint8), connectivity=4)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        static = np.isin(labels, border[border != 0])
        margin = int(getattr(self.video_args, "static_margin", 0) or 0)
        
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            static = cv2.erode(static.astype(np.uint8), kernel).astype(bool)

        frac = float(static.mean())
        print(f"--ignore-static: {frac * 100:.1f}% of the frame is static across {num_frames} frames")
        if frac > 0.9:
            print("--ignore-static skipped: nearly the whole frame is static (subject never moves?)")
            return None
        return static

    @torch.inference_mode()
    def blank_ignored(self, session_id):
        fracs = self._ignore_fracs()
        state = self.predictor._get_session(session_id)["state"]
        tensors = state["input_batch"].img_batch.tensors
        self.video_args.static_mask = self._compute_static_mask(tensors)
        static = self.video_args.static_mask

        if self._has_ignore(fracs, static):
            self._blank_region(tensors, fracs, 0, static)
        self._set_token_drop(tensors.shape[-2:], fracs, static)

    def _set_token_drop(self, hw, fracs, static):

        model = self.predictor.model
        mods = [m for m in model.modules() if hasattr(m, "set_ignore_pixels")]
        vits = [m for m in mods if type(m).__name__ == "ViT"]
        encs = [m for m in mods if "Encoder" in type(m).__name__]
        decs = [m for m in mods if "Decoder" in type(m).__name__]
        active = self._has_ignore(fracs, static)
        want_enc = active and getattr(self.video_args, "drop_tokens_enc", False)
        want_dec = active and getattr(self.video_args, "drop_tokens", False)
        want_vit = active and getattr(self.video_args, "drop_tokens_vit", False)
        ignore = None

        if want_enc or want_dec or want_vit:
            ignore = np.zeros(tuple(hw), dtype=bool)
            self._blank_region(ignore, fracs, True, static)
        for m in encs:
            m.set_ignore_pixels(ignore if want_enc else None)
        for m in decs:
            m.set_ignore_pixels(ignore if want_dec else None)
        for m in vits:
            m.set_ignore_pixels(ignore if want_vit else None)
        if ignore is not None:
            print(f"token drop: {ignore.mean() * 100:.1f}% pixels ignored (vit={want_vit}, encoder={want_enc}, decoder={want_dec})")

    def clear_ignored_masks(self, out):
   
        fracs = self._ignore_fracs()
        static = getattr(self.video_args, "static_mask", None)
        masks = out.get("out_binary_masks") if isinstance(out, dict) else None
        if not self._has_ignore(fracs, static) or masks is None or len(masks) == 0:
            return
        
        self._blank_region(masks, fracs, False, static)

    def abs_to_rel_coords(self, coords=None, IMG_WIDTH=None, IMG_HEIGHT=None, coord_type="box"):
        
        if coord_type == "point":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
        elif coord_type == "box":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT] for x, y, w, h in coords]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    def track(self, video_path = None, boxes = None, labels = None, refine_object_0=False, refine_object_1=False, refine_object_2=False, refine_object_3=False, shutdown=True, frames=None, chunk_size=16):
        predictor, video_path, prompt, show_plots, add_box, sub_box = self.predictor, self.video_path, self.video_args.prompt, self.video_args.show_plots, self.video_args.add_box, self.video_args.sub_box
        
        if video_path is None:
            video_path = self.video_path

        if frames is not None:
            frames = frames

        else:
            frames = glob.glob(os.path.join(video_path, "*.png"))
            try:
                frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            except ValueError:
                print(f'frame names are not in "<frame_idx>.png" format: {frames[:5]=}, '
                    f"falling back to lexicographic sort.")
                frames.sort()
  
        H, W = load_frame(frames[0]).shape[:2]

        start_frame_idx = 0
        num_frames = len(frames)
        masks = torch.zeros((num_frames, H, W), dtype=torch.float32)
        scores = torch.zeros(num_frames, dtype=torch.float32)
        propagation_direction = "forward"
        close_after_propagation = True
        kill_model = shutdown

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
                run_gc_collect=True,
            )
        )

        print(f'is_success: {is_success["is_success"]}')

        self.blank_ignored(session_id)
   
        if add_box and not sub_box:
            boxes = np.array([[W  * 0.2, H * 0.2, W  * 0.45, H * 0.45]])
            labels = np.array([1])
            boxes = torch.tensor(
                self.abs_to_rel_coords(boxes, W, H, coord_type="box"),
                dtype=torch.float32,
            )
            labels = torch.tensor(labels, dtype=torch.int32)

        if sub_box and not add_box:
            boxes = np.array([[0, H * 0.9, W, H * 0.1]])
            labels = np.array([0])
            boxes = torch.tensor(
                self.abs_to_rel_coords(boxes, W, H, coord_type="box"),
                dtype=torch.float32,
            )
            labels = torch.tensor(labels, dtype=torch.int32)

        if add_box and sub_box:
            boxes = np.array([
                [W  * 0.1, H * 0.1, W  * 0.8, H * 0.8], 
                [0, H * 0.8, W, H * 0.2],
                ])
            
            labels = np.array([1, 0])
            boxes = torch.tensor(
                self.abs_to_rel_coords(boxes, W, H, coord_type="box"),
                dtype=torch.float32,
            )
            labels = torch.tensor(labels, dtype=torch.int32)
         
        prompt_text = prompt if prompt is not None else None
        frame_idx = 0
        obj_ids = 0

        predictor.model.hotstart_delay = 0
        response = predictor.handle_request(
            request=dict(
                type = "add_prompt",
                session_id = session_id,
                frame_idx = frame_idx,
                text = prompt_text,
                bounding_boxes = boxes,
                bounding_box_labels = labels,
                rel_coordinates=True,
                obj_id = obj_ids,           
                output_prob_thresh=0.1,
                    )
        )

        if refine_object_0:
            frame_idx = 0
            obj_id = 0

            center = np.array([[W // 2, H // 2]])
            center_bottom = np.array([[W // 2, H]])

            points_abs = np.array(
                [
                    [W // 2, H // 2],
                    [W // 2, H - H // 10],
                ]
            )
         
            labels = np.array([1, 0])
                
            points_tensor = torch.tensor(
                self.abs_to_rel_coords(points_abs, W, H, coord_type="point"),
                dtype=torch.float32,
            )
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

        if refine_object_1:
            frame_idx = 0
            obj_id = 1
            points_abs = np.array(
                [
                    [740, 450],
                    [760, 630],
                    [840, 640],
                    [760, 550],
                ]
            )

            labels = np.array([1, 0, 0, 1])
    
            points_tensor = torch.tensor(
                self.abs_to_rel_coords(points_abs, W, H, coord_type="point"),
                dtype=torch.float32,
            )
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

        plot_outputs = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=None
                )
                ):

            frame_idx = response.get("frame_idx", 0)
            merged, score = self._merge_outputs(response.get("outputs") or {})
            if show_plots:
                plot_outputs[frame_idx] = response.get("outputs")

            if merged is not None and 0 <= frame_idx < num_frames:
                masks[frame_idx] = merged
                scores[frame_idx] = score

        if show_plots:
            outputs_per_frame = prepare_masks_for_visualization(plot_outputs)
            vis_frame_stride = 1
            plt.close("all")
            for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
                visualize_formatted_frame_output(
                    frame_idx,
                    frames,
                    outputs_list=[outputs_per_frame],
                    titles=["SAM 3.1 Dense Tracking outputs"],
                    figsize=(6, 6))

        if close_after_propagation:
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                    run_gc_collect=True,
                    ))

        if kill_model:
            predictor.shutdown()

        del predictor

        return scores, masks

    @staticmethod
    def _merge_outputs(outputs):
        if "out_mask_logits" in outputs and outputs["out_mask_logits"].shape[0] > 0:
            merged = torch.sigmoid(torch.as_tensor(outputs["out_mask_logits"]).float()).amax(0)
        elif "out_binary_masks" in outputs and outputs["out_binary_masks"].shape[0] > 0:
            merged = torch.as_tensor(outputs["out_binary_masks"]).float().amax(0)
        else:
            return None, 0.0

        probs = outputs.get("out_probs")
        if probs is not None and len(probs) > 0:
            score = torch.as_tensor(probs).float().amax()
        else:
            score = merged.amax()
        return merged.cpu(), float(score)

    @staticmethod
    def fill_soft(soft_masks, valid_flags, max_interp_gap=6):
        filled = [m.clone() for m in soft_masks]
        valid_idx = [i for i, ok in enumerate(valid_flags) if ok]
        if not valid_idx:
            return filled, 0

        filled_count = 0
        for i in range(len(filled)):
            if valid_flags[i]:
                continue

            prev_i = next((j for j in reversed(valid_idx) if j < i), None)
            next_i = next((j for j in valid_idx if j > i), None)

            if prev_i is not None and next_i is not None:
                if next_i - prev_i - 1 <= max_interp_gap:
                    alpha = (i - prev_i) / (next_i - prev_i)
                    filled[i] = (1.0 - alpha) * filled[prev_i] + alpha * filled[next_i]
                else:
                    src = prev_i if (i - prev_i) <= (next_i - i) else next_i
                    filled[i] = filled[src].clone()
            else:
                filled[i] = filled[prev_i if prev_i is not None else next_i].clone()
            filled_count += 1

        return filled, filled_count

    @classmethod
    def run(cls, frames_dir, video_args):
        sam3_height = video_args.sam3_height
        min_valid_pixels = int(sam3_height * 0.8)
        chunk_size = max(1, int(getattr(video_args, "sam3_chunk", 0) or 150)) if video_args.debug is None else video_args.debug

        folder = Path(frames_dir)
        image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
        image_files = [f for f in image_files if "_mask" not in f.stem]
        if not image_files:
            return

        with Image.open(image_files[0]) as first:
            size = (round(sam3_height * first.width / first.height), sam3_height)

        seq_dir = folder / "_sam3video_seq"
        if seq_dir.exists():
            shutil.rmtree(seq_dir)
        seq_dir.mkdir(parents=True)

        output_paths = [f.with_name(f"{f.stem}_mask.png") for f in image_files]

        for i, frame_path in enumerate(image_files):
            chunk_dir = seq_dir / f"c{i // chunk_size:05d}"
            chunk_dir.mkdir(exist_ok=True)
            with Image.open(frame_path) as raw:
                img = raw.convert("RGB")
                if img.size != size:
                    img = img.resize(size, Image.Resampling.BILINEAR)
                img.save(chunk_dir / f"{i % chunk_size:06d}.png", compress_level=1)

        tracker = cls(video_path=str(seq_dir), video_args=video_args)
        soft_masks = []
        valid_flags = []

        for c in range((len(output_paths) + chunk_size - 1) // chunk_size):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tracker.video_path = str(seq_dir / f"c{c:05d}")
                scores, masks = tracker.track(shutdown=False, chunk_size=chunk_size)

            for j in range(min(chunk_size, len(output_paths) - c * chunk_size)):
                soft = masks[j]
                if scores[j] <= 0:
                    print(f"No SAM3 mask for {output_paths[c * chunk_size + j].name}; marking as missing for temporal fill")
                else:
                    print("Confidence:", float(scores[j]))
                soft_masks.append(soft)
                valid_flags.append(int((soft >= 0.5).sum()) >= min_valid_pixels)

        filled_masks, filled_count = cls.fill_soft(soft_masks, valid_flags, max_interp_gap=6)
        if filled_count > 0:
            print(f"Filled {filled_count} missing/weak SAM3 masks using temporal soft-mask interpolation")

        for out_path, soft_mask in zip(output_paths, filled_masks):
            alpha = ((soft_mask - 0.5) * 10.0 + 0.5).clamp(0.0, 1.0).mul(255).to(torch.uint8).numpy()
            Image.fromarray(alpha, mode="L").save(out_path)

        shutil.rmtree(seq_dir, ignore_errors=True)
        tracker.predictor.shutdown()
        del tracker, soft_masks, valid_flags, filled_masks
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def seed_masks(cls, video_args, frames_dir, masks_dir, mask_segments):
        cls.run(str(frames_dir), video_args)

        for seg in mask_segments:
            for frame_path, tag in ((seg.left_frame_path, 'l'), (seg.right_frame_path, 'r'), (seg.sbs_frame_path, 'sbs')):
                if not frame_path:
                    continue
                base = os.path.splitext(os.path.basename(frame_path))[0]
                mask_src = frames_dir / f'{base}_mask.png'

                if mask_src.exists():
                    final_mask_path = str(masks_dir / f'seg{seg.index:02d}_{tag}_mask.png')
                    shutil.move(str(mask_src), final_mask_path)
                    setattr(seg, {'l': 'left', 'r': 'right', 'sbs': 'sbs'}[tag] + '_mask_path', final_mask_path)

        return mask_segments

    @classmethod
    def seed_masks_sbs(cls, video_args, sbs_dir, masks_dir, mask_segments):
        sbs_dir.mkdir(parents=True, exist_ok=True)
        candidates = [s for s in mask_segments if s.left_frame_path and s.right_frame_path]

        for seg in candidates:
            with Image.open(seg.left_frame_path) as left, Image.open(seg.right_frame_path) as right:
                sbs = Image.new('RGB', (left.width + right.width, max(left.height, right.height)))
                sbs.paste(left.convert('RGB'), (0, 0))
                sbs.paste(right.convert('RGB'), (left.width, 0))
                sbs.save(sbs_dir / f'seg{seg.index:02d}_sbs.png', compress_level=1)

        cls.run(str(sbs_dir), video_args)

        failed = []
        for seg in candidates:
            mask_src = sbs_dir / f'seg{seg.index:02d}_sbs_mask.png'
            if not mask_src.exists():
                failed.append(seg)
                continue

            mask = torch.from_numpy(np.array(Image.open(mask_src).convert('L')))
            height, width = mask.shape
            halves = {'left': mask[:, :width // 2], 'right': mask[:, width // 2:]}
            areas = {name: int((half > 127).sum()) for name, half in halves.items()}

            if min(areas.values()) == 0 or min(areas.values()) < 0.5 * max(areas.values()):
                print(f"seg{seg.index:02d}: SBS mask halves disagree (areas L={areas['left']} R={areas['right']}), using per-eye SAM3")
                failed.append(seg)
                continue

            for name, half in halves.items():
                final_mask_path = str(masks_dir / f'seg{seg.index:02d}_{name}_mask.png')
                Image.fromarray(half.numpy()).resize((height, height), Image.Resampling.BILINEAR).save(final_mask_path)
                setattr(seg, f'{name}_mask_path', final_mask_path)

        return failed

    @staticmethod
    def seed_report(mask_segments):
        print("Stereo seed masks (left vs right):")
        for seg in mask_segments:
            if not (seg.left_mask_path and seg.right_mask_path):
                continue
            left = torch.from_numpy(np.array(Image.open(seg.left_mask_path).convert('L'))) > 127
            right = torch.from_numpy(np.array(Image.open(seg.right_mask_path).convert('L'))) > 127
            la, ra = int(left.sum()), int(right.sum())
            ratio = min(la, ra) / max(la, ra, 1)
            rows = [m.any(dim=1).nonzero().flatten() for m in (left, right)]
            dy = max(abs(int(rows[0][0]) - int(rows[1][0])), abs(int(rows[0][-1]) - int(rows[1][-1]))) / left.shape[0] if all(len(r) for r in rows) else 1.0
            print(f"  seg{seg.index:02d}: area ratio {ratio:.3f}, vertical extent diff {dy * 100:.1f}% of height")

def stereo_matte_report(left_pha: torch.Tensor, right_pha: torch.Tensor, label: str, size: int = 250, ratio_warn: float = 0.85, dy_warn: float = 0.05) -> None:
    def load(t):
        small = [F.interpolate(t[i:i + 16, None].float(), size=(size, size), mode='area')[:, 0] for i in range(0, len(t), 16)]
        return (torch.cat(small) > 127).numpy() if small else np.zeros((0, size, size), bool)

    left, right = load(left_pha), load(right_pha)
    n = min(len(left), len(right))
    if n == 0:
        print(f"{label}: stereo matte check skipped (could not read mattes)")
        return

    ratios, dys = [], []
    for i in range(n):
        la, ra = int(left[i].sum()), int(right[i].sum())
        ratios.append(min(la, ra) / max(la, ra, 1))
        rows = [np.where(m.any(axis=1))[0] for m in (left[i], right[i])]
        dys.append(max(abs(int(rows[0][0]) - int(rows[1][0])), abs(int(rows[0][-1]) - int(rows[1][-1]))) / size if all(len(r) for r in rows) else 1.0)

    worst = int(np.argmin(ratios))
    flagged = sum((r < ratio_warn) or (d > dy_warn) for r, d in zip(ratios, dys))
    status = "WARN" if flagged else "ok"
    print(f"{label}: stereo matte check {status} - worst area ratio {ratios[worst]:.3f} at frame {worst}, "
          f"max vertical extent diff {max(dys) * 100:.1f}%, frames flagged (ratio < {ratio_warn} or extent diff > {dy_warn * 100:.0f}%): {flagged}/{n}")

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

def _config_overrides(matanyone_model, video_args, *, verbose: bool = False) -> None:

    cfg = matanyone_model.cfg
    mem_every = video_args.ma2_mem_every
    max_mem_frames = video_args.ma2_max_mem_frames
    use_long_term = video_args.ma2_use_long_term

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
        version = str(video_args.matanyone_version).lower()
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

def _matanyone_crop_box(h, w, fracs, static, mult=16):
    ignore = np.zeros((h, w), dtype=bool)
    sam3_video_inference._blank_region(ignore, fracs, True, static)
    keep = ~ignore
    if not keep.any():
        return None
    rows = np.flatnonzero(keep.any(1))
    cols = np.flatnonzero(keep.any(0))

    def span(lo, hi, dim):
        n = min(dim, -(-(hi - lo + 1) // mult) * mult)
        s = min(max(lo - (n - (hi - lo + 1)) // 2, 0), dim - n)
        return s, s + n

    r0, r1 = span(int(rows[0]), int(rows[-1]), h)
    c0, c1 = span(int(cols[0]), int(cols[-1]), w)
    return None if (r1 - r0) * (c1 - c0) >= h * w else (r0, r1, c0, c1)

def _matanyone_process_segment(matanyone_model, device, inference_core, frames, mask_path, video_args, verbose=False) -> torch.Tensor:
    n_warmup = video_args.warmup
    max_size = video_args.matanyone_height
    r_erode = video_args.erode
    r_dilate = video_args.dilate

    _config_overrides(matanyone_model, video_args, verbose=verbose)
    processor = inference_core(matanyone_model, cfg=matanyone_model.cfg)

    repeated_frames = frames[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    frames = torch.cat([repeated_frames, frames], dim=0)
    length = frames.shape[0]

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float().to(device)

    if r_dilate != 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode != 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    if mask.shape != tuple(frames.shape[-2:]):
        if video_args.debug is not None:
            print(f"Mask shape before interpolation: {mask.shape}")
 
        if max_size > 0:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=tuple(frames.shape[-2:]), mode="nearest-exact"
            )[0, 0]
        if video_args.debug is not None:
            print(f"Mask shape after interpolation: {mask.shape}")

    objects = [1]
    phas = []

    ignore_fracs = {s: float(getattr(video_args, f"ignore_{s}", 0.0) or 0.0) for s in ("bottom", "left", "right")}
    static_mask = getattr(video_args, "static_mask", None)
    full_hw = tuple(frames.shape[-2:])
    crop = None

    if getattr(video_args, "blank_matanyone", False) and (static_mask is not None or any(ignore_fracs.values())):
        ign = np.zeros(full_hw, dtype=bool)
        sam3_video_inference._blank_region(ign, ignore_fracs, True, static_mask)
        frames[..., torch.from_numpy(ign)] = 128
        print(f"--blank-matanyone: {ign.mean() * 100:.1f}% of each frame set to gray (same size, no compute saved)")

    if mask.shape == full_hw:
        crop = _matanyone_crop_box(full_hw[0], full_hw[1], ignore_fracs, static_mask)

    if crop is not None:
        r0, r1, c0, c1 = crop
        frames = frames[..., r0:r1, c0:c1]
        mask = mask[r0:r1, c0:c1]

        if video_args.debug is not None:
            print(f"--crop-matanyone: matting {r1 - r0}x{c1 - c0} of {full_hw[0]}x{full_hw[1]} ({(r1 - r0) * (c1 - c0) / (full_hw[0] * full_hw[1]) * 100:.0f}% of area)")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ti in tqdm.tqdm(range(length)):
            image = frames[ti].to(device).float() / 255.

            if ti == 0:
                output_prob = processor.step(image, mask, objects=objects)
                output_prob = processor.step(image, first_frame_pred=True, force_permanent=False)

            elif ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True, force_permanent=False)

            else:
                output_prob = processor.step(image)

            mask = processor.output_prob_to_mask(output_prob, matting=True)

            if ti > (n_warmup-1):
                pha = torch.round(mask * 255).to(torch.uint8)
                pha = torch.clamp(pha, 0, 255).cpu()

                if crop is not None:
                    full = torch.zeros(*pha.shape[:-2], *full_hw, dtype=pha.dtype)
                    full[..., crop[0]:crop[1], crop[2]:crop[3]] = pha
                    pha = full

                if static_mask is not None or any(ignore_fracs.values()):
                    sam3_video_inference._blank_region(pha, ignore_fracs, 0, static_mask)
                phas.append(pha.reshape(full_hw))

    return torch.stack(phas)

class MaskWriter:
    def __init__(self, path: str, fps_str: str):
        self.path, self.fps_str, self.proc, self.shape = path, fps_str, None, None

    def write(self, pha: torch.Tensor) -> None:
        if self.proc is None:
            self.shape = tuple(pha.shape[-2:])
            h, w = self.shape
            cmd = [FFMPEG_BIN, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{w}x{h}',
                   '-framerate', self.fps_str, '-i', '-', *lossless_args('gray'), self.path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if tuple(pha.shape[-2:]) != self.shape:
            raise RuntimeError(f"Mask size changed mid-video: {tuple(pha.shape[-2:])} vs {self.shape}")
        self.proc.stdin.write(pha.contiguous().numpy().tobytes())

    def close(self) -> str:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"Mask encode failed: {self.path}")
        return self.path

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

    global _matanyone_is_first_status

    version = str(video_args.matanyone_version).lower()
    matanyone_model, device, inference_core, loaded_version = _load_matanyone_runtime(version)
    if loaded_version != version:
        raise RuntimeError(f"Loaded model version mismatch: expected {version}, got {loaded_version}")

    mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
    total = len(mask_segments)
    first = True
    _matanyone_is_first_status = True

    th = video_args.matanyone_height
    keyframe_start = bool(getattr(video_args, 'keyframe_segments', False))
    output_mask = str(segments_dir / f'{Path(video_args.video_path).stem}_mask{MASK_EXT}')
    writer = MaskWriter(output_mask, data['fps_str'])

    def matte(seg, eye, frames, mask_path, op):
        nonlocal first
        _update_status(op, total * 2, f'seg{seg.index:02d}_{eye}', seg.end_time - seg.start_time)
        pha = _matanyone_process_segment(
            matanyone_model, device, inference_core,
            frames, mask_path, video_args, verbose=first)
        first = False
        return pha

    try:
        for n, seg in enumerate(mask_segments if video_args.debug is None else mask_segments[:video_args.debug]):
            if seg.left_mask_path and seg.right_mask_path:
                t0 = time.perf_counter()
                frames = decode_segment(video_args.video_path, seg.start_time, seg.end_time, th * 2, th, keyframe_start)
                print(f"seg{seg.index:02d}: decoded {len(frames)} frames in {time.perf_counter() - t0:.1f}s")
                left_pha = matte(seg, 'left', frames[..., :th], seg.left_mask_path, n * 2 + 1)
                right_pha = matte(seg, 'right', frames[..., th:], seg.right_mask_path, n * 2 + 2)
                del frames

                pha = torch.cat([left_pha, right_pha], dim=-1)
            else:
                frames = decode_segment(video_args.video_path, seg.start_time, seg.end_time, th * 2, th, keyframe_start)
                pha = matte(seg, 'sbs', frames, seg.sbs_mask_path, n * 2 + 1)
                del frames

            t0 = time.perf_counter()
            for frame in pha:
                writer.write(frame)
            print(f"seg{seg.index:02d}: wrote {len(pha)} mask frames in {time.perf_counter() - t0:.1f}s")
            seg.video_path = output_mask

            gc.collect()
            torch.cuda.empty_cache()
    finally:
        writer.close()

    sys.stderr.write("\n")
    return segments, output_mask

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

def deliver_mask(master_mask: str, source_video: str, data: dict) -> str:
    out = str(Path(source_video).with_name(f'{Path(source_video).stem}_mask.mp4'))
    chain = (f"[0:v]scale={data['width']}:{data['height']}:flags=bicubic+accurate_rnd:in_range=pc:out_range=tv,"
             f"format=yuv420p[out]")
    return final_encode(['-i', master_mask], chain, out, data, what="Mask delivery")

def process_video(video_path, args: argparse.Namespace, temp_root: Path) -> str:
    
    video_path = str(Path(video_path).expanduser().resolve())
    video_name = Path(video_path).stem
    timer = StageTimer(args.debug is not None)
    data = info(video_path)
    timer.mark('ffprobe info()')
    print(f"Specs: {data['width']}x{data['height']} @ {data['fps']}fps, duration: {data['duration']}, format: {data['pix_fmt']}")
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

    video_args = argparse.Namespace(**vars(args), video_path=video_path)
    alpha_output = video_args.alpha
    overlay_mask = video_args.overlay_mask
    run_all = video_args.all

    if video_args.overlay_mask is not None:
        overlay_target = str(Path(video_path).with_name(f"{video_name}_overlay.mp4"))
        overlay_video = mask_overlay(
            video_path,
            overlay_mask,
            overlay_target,
            background_color=video_args.overlay_color,
            video_args=video_args,
            data=data,
            )

        print(f'Overlay: {overlay_video}')
        print('=' * 60)
        return overlay_video

    segments = calculate_segments(
        data['duration'],
        video_args.segment_length,
        debug=video_args.debug,
        frames=int(data['frames']),
        fps=data['fps'],
        segment_frames=video_args.segment_frames or 0,
        keyframes=list_keyframes(video_args.video_path) if video_args.keyframe_segments else None,
        )

    mask_segments = [s for s in segments if s.seg_type == SegmentType.MASK]
    timer.mark('calculate_segments')

    mask_segments = extract_segments(
        video_args,
        frames_dir, 
        segments_dir, 
        mask_segments, 
        segments,
        data,
    )

    timer.mark('extract_segments (ffmpeg)')

    if video_args.sam3_sbs:
        failed = sam3_video_inference.seed_masks_sbs(video_args, temp_dir / 'sbs_frames', masks_dir, mask_segments)
        if failed:
            sam3_video_inference.seed_masks(video_args, frames_dir, masks_dir, failed)
    else:
        mask_segments = sam3_video_inference.seed_masks(
            video_args,
            frames_dir, 
            masks_dir, 
            mask_segments, 
        )

    timer.mark('sam3_masks')
   
    segments, output_mask = matanyone(
        video_args,
        segments_dir, 
        mask_segments, 
        segments,
        data,
        )
    
    timer.mark('matanyone')

    master_mask = output_mask
    if getattr(video_args, 'save_mask', False):
        output_mask = deliver_mask(master_mask, video_path, data)
        timer.mark('mask delivery')

    if run_all:
        video_args.fisheye180 = True
        packer(
            video_path, 
            video_args,
            mask_path=master_mask,
            )
        
        mask_overlay(
            video_path,
            master_mask,
            output_path = str(Path(video_path).with_name(f"{video_name}_overlay.mp4")),
            background_color=video_args.overlay_color,
            video_args=video_args,
            )

    if alpha_output:
        packer(
            video_path, 
            video_args,
            mask_path=master_mask,
            )
        
    else:
        mask_overlay(
            video_path,
            master_mask,
            output_path = str(Path(video_path).with_name(f"{video_name}_overlay.mp4")),
            background_color=video_args.overlay_color,
            video_args=video_args,
            )

    if video_args.debug:
        sam3_video_inference.seed_report(mask_segments)

    timer.mark('packer/overlay')
    print('=' * 60)
    print(f'Segments: {len(segments)} ({len(mask_segments)} masks) - Output: {output_mask}')
    print()

    with open(temp_dir / 'segments.txt', 'w', encoding='utf-8') as f:
        f.write(f'# {video_name}\n')
        for seg in segments:
            f.write(f'{seg.index},{seg.seg_type.value},{seg.start_time:.3f},{seg.end_time:.3f},{seg.video_path}\n')

    print(f"info() cache: {info.cache_info()}")
    return output_mask

def calculate_segments(video_duration: float, max_segment_length: float = 5.0, debug = None, frames: int = 0, fps: float = 0.0, segment_frames: int = 0, keyframes: list[float] | None = None) -> List[SegmentInfo]:
    max_segment_length = debug if debug is not None else max_segment_length
    video_duration = debug if debug is not None else video_duration
    frames = int(round(debug * fps)) if debug is not None else frames
    if keyframes and fps:
        limit = int(round(video_duration * fps)) if debug is not None else (frames or int(round(video_duration * fps)))
        starts = sorted({0} | {round(k * fps) for k in keyframes if round(k * fps) < limit})
        bounds = starts + [limit]
        segments = []
        for s, e in zip(bounds, bounds[1:]):
            if e <= s:
                continue
            segments.append(SegmentInfo(index=len(segments), start_time=s / fps,
                    end_time=e / fps, seg_type=SegmentType.MASK))
        return segments
    if segment_frames and frames and fps:

        seg_frames = max(1, int(segment_frames))
        limit =  frames
        segments = []
        start = 0
        while start < limit:
            end = min(start + seg_frames, limit)
            if 0 < limit - end < 2:
                end = limit
            segments.append(SegmentInfo(index=len(segments), start_time=start / fps,
                    end_time=end / fps, seg_type=SegmentType.MASK))
            start = end
        return segments

    segments: List[SegmentInfo] = []
    chunk_start = 0.0
    index = 0
    while chunk_start < (video_duration):
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

    for seg in segments:
        dur = seg.end_time - seg.start_time
    print(f'Total: {len(segments)} segments')

    th = video_args.matanyone_height
    keyframe_start = bool(getattr(video_args, 'keyframe_segments', False))

    for i, seg in enumerate(mask_segments) if video_args.debug is None else enumerate(mask_segments[:video_args.debug]):
        seg_t0 = time.perf_counter()

        if video_args.sbs:
            seg.sbs_frame_path = str(frames_dir / f'seg{seg.index:02d}_sbs.png')
            extract_first_frame(video_args.video_path, seg.start_time, seg.end_time, th * 2, th,
                                [(seg.sbs_frame_path, '')], keyframe_start)
        else:
            seg.left_frame_path = str(frames_dir / f'seg{seg.index:02d}_l.png')
            seg.right_frame_path = str(frames_dir / f'seg{seg.index:02d}_r.png')
            extract_first_frame(video_args.video_path, seg.start_time, seg.end_time, th * 2, th,
                                [(seg.left_frame_path, f'crop={th}:{th}:0:0'),
                                 (seg.right_frame_path, f'crop={th}:{th}:{th}:0')], keyframe_start)

        print(f'[timing] segment {seg.index} extract: {time.perf_counter() - seg_t0:.2f}s')
    return mask_segments

def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="VR Video Masking and things and stuff")
    parser.add_argument("--model", type=str, default="sam3.1")
    parser.add_argument("input_path", type=str, default="videos")
    parser.add_argument("--matanyone-height", type=int, default=1008)
    parser.add_argument("--sam3-height", type=int, default=1008)
    parser.add_argument("--sam3-chunk", type=int, default=600, help="Max frames per SAM3 session (frees GPU memory between chunks)")
    parser.add_argument("--segment-length", type=float, default=1)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="agirl")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--add-box", type=bool, default=False)
    parser.add_argument("--sub-box", type=bool, default=False)
    parser.add_argument("--ignore-bottom", type=float, default=0.2, metavar='FRAC', help='Fraction (0-1) of each frame, measured from the bottom, that SAM3 should ignore (blanked in the model input and cleared from its masks)')
    parser.add_argument("--ignore-left", type=float, default=0.15, metavar='FRAC', help='Fraction (0-1) of each frame, measured from the left, that SAM3 should ignore')
    parser.add_argument("--ignore-right", type=float, default=0.15, metavar='FRAC', help='Fraction (0-1) of each frame, measured from the right, that SAM3 should ignore')
    parser.add_argument("--ignore-static", type=float, default=0.0, metavar='THRESH', help='Ignore border-connected pixels whose value never changes by more than THRESH (0-1 pixel units, e.g. 0.04) across the SAM3 seed frames; 0 disables')
    parser.add_argument("--static-margin", type=int, default=0, metavar='PX', help='Shrink the static region by PX pixels (at SAM3 resolution) to keep it away from the subject')
    parser.add_argument("--blank-matanyone", action='store_true', help='Fill the ignored area of MatAnyone2 input frames with flat gray, keeping the full frame size (no speedup, no geometry change)')
    parser.add_argument("--crop-matanyone", action='store_true', help='Run MatAnyone2 only on the bounding box of the non-ignored area (multiple of 16); alpha is pasted back into a full-size zero frame. Needs --ignore-* or --ignore-static')
    parser.add_argument("--drop-tokens-enc", action='store_true', help='Experimental: also skip fully ignored patches in the SAM3 fusion encoder (can change boxes)')
    parser.add_argument("--drop-tokens-vit", action='store_true', help='Run the SAM3 ViT backbone only on a crop of the patch grid covering the non-ignored area (rounded up to whole 24-token windows); output grid is restored by edge replication. Inference-only, no retraining')
    parser.add_argument("--drop-tokens", action='store_true', help='Remove fully ignored (edge/static) patches from the SAM3 decoder image cross-attention instead of only blanking them; inference-only, no retraining')
    parser.add_argument("--sbs", type=bool, default=False)
    parser.add_argument('--sam3-sbs', action='store_true', help='Seed both eyes from one SAM3 pass over side-by-side frames (falls back to per-eye SAM3 if the halves disagree)')
    parser.add_argument('--matanyone-version', type=str, default='v2', choices=['v1', 'v2'], help='Select MatAnyone runtime version')
    parser.add_argument('--ma2-mem-every', type=int, default=2, help='Override MatAnyone mem_every')
    parser.add_argument('--ma2-max-mem-frames', type=int, default=2, help='Override MatAnyone memory window in frames (works for v1 and v2)')
    parser.add_argument('--ma2-use-long-term', type=str, default='off', choices=['auto', 'on', 'off'], help='Override MatAnyone long-term memory ')
    parser.add_argument('--overlay-color', type=str, default='0x00ff00', help='Background color for overlay (use 0x00ff00 for pure green)')
    parser.add_argument('--overlay-mask', type=str, default=None, help='Write a composited video with a provided mask over the original source')
    parser.add_argument('--alpha-packer', type=bool, default=False, help='Run alpha packer. Provide folder with video and mask (_mask.<ext>) for  input_path')
    parser.add_argument('--decompose-alpha', '--decompose_alpha', dest='decompose_alpha', action='store_true', help='Reverse of alpha packer')
    parser.add_argument('--decompose-clean-mask', type=str, default='assets/black_mask.png', help='PNG overlay used to clean alpha payload regions')
    parser.add_argument('--save-mask', type=bool, default=False, help='Also write the matte as a source-resolution HEVC mp4 next to the source (<name>_mask.mp4). --save-mask <true|false>')
    parser.add_argument('--alpha', type=bool, default=False, help='Run alpha packer instead of overlay. --alpha <true|false>')
    parser.add_argument('--all', type=bool, default=False, help='Run alpha packer and overlay. --all <true|false>')
    parser.add_argument('--alpha-erode', type=int, default=1, help='Alpha pack matte choke (3x3 erosion passes at --alpha-ref-size payload size; 0 = off)')
    parser.add_argument('--alpha-blur', type=float, default=1.8, help='Alpha pack matte feather (gblur sigma at --alpha-ref-size payload size; 0 = off)')
    parser.add_argument('--alpha-contrast', type=float, default=None, help='Alpha pack matte contrast (default: 2.0 up to 2400px high, else 2.5; 1.0 = off)')
    parser.add_argument('--alpha-gamma', type=float, default=None, help='Alpha pack matte gamma (default: 1.2 up to 2400px high, else 1.4; 1.0 = off)')
    parser.add_argument('--alpha-ref-size', type=int, default=1024, help='Payload size in px that --alpha-erode/--alpha-blur refer to; they scale with the real payload size (0 = no scaling)')
    parser.add_argument('--alpha-tmix', type=int, default=1, help='Odd number of frames to average the matte over before packing (1 = off)')
    parser.add_argument('--show-plots', type=bool, default=False, help='Sam3 mask plots will be displayed if True.')
    parser.add_argument('--fisheye180', type=bool, default=False, help='Convert video or folder to SBS fisheye180. Works with alphapacker')
    parser.add_argument('--keyframe-segments', action='store_true', help='Split at the video\'s own keyframes: one segment per keyframe-to-keyframe span (lengths vary; overrides --segment-frames/--segment-length) and seek without pre-roll decoding')
    parser.add_argument('--segment-frames', type=int, default=None, metavar='N', help='Split into segments of exactly N frames each (overrides --segment-length; a final remainder shorter than N is kept as its own segment)')
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
        return packer(args.input_path, args)
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
        video_args = argparse.Namespace(**vars(args), video_path=video_path)
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
