import re, subprocess, os, torch, av, numpy as np, argparse
from pathlib import Path
from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFilter

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

SAM3_MAX = 1008
BATCH_SIZE = 50

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

def encoder_args() -> list[str]:

    return [

        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp',
        '-fps_mode', 'cfr',
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
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate:format=duration',
        '-of', 'csv=p=0',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    lines = result.stdout.strip().split('\n')

    w, h, fps_str,fps_avg = lines[0].split(',')

    cmd2 = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    duration = float(subprocess.check_output(cmd2).decode().strip())

#     vfr_cmd = [
#         'ffmpeg', '-i', video_path, 
#         '-vf', 'vfrdet', 
#         '-f', 'null', '-'
#     ]
#     vfr_result = subprocess.run(vfr_cmd, capture_output=True, text=True)
    
#     is_vfr = False
#     vfr_match = re.search(r'VFR:(\d+\.\d+)', vfr_result.stderr)
#     if vfr_match:
#         vfr_score = float(vfr_match.group(1))
  
#         is_vfr = vfr_score > 0.0

    is_vfr = False
    if fps_str != fps_avg:
        is_vfr = True

    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)

    else:
        fps = float(fps_str)

    return int(w), int(h), fps, duration, is_vfr

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

def norm_video(source_video, w = None, h = None, fps = None, progress_prefix: str = "[normalize] ", video_args = None) -> str:

    wi, hi, _, duration, is_vfr = info(source_video)
    enc = encoder_args()

    if video_args.normalize_input:
        print(f" normalize_input = {video_args.normalize_input}")
        return source_video

    else:
        source_path = Path(source_video).expanduser().resolve()
        output_video = str(source_path.with_name(f"{source_path.stem}_normed.mp4"))

        fps = aorb(fps, 60)

        if w is not None:
            wi = w
            hi = h

        cmd = [

            'ffmpeg', '-y', '-hwaccel', 'auto',
            '-i', source_video,
            '-filter_complex', f'[0:v]fps={fps},setpts=N/({fps}*TB),scale=w=iw:h=ih:flags=lanczos:threads=0',
            '-r', str(fps),
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
    fps = info(source_video)[2]

    os.makedirs(os.path.dirname(os.path.abspath(output_video)) or '.', exist_ok=True)

    cmd = [

        'ffmpeg', '-y', '-hwaccel', 'auto',
        '-i', source_video,
        '-filter_complex', f'[0:v]fps={fps},setpts=N/({fps}*TB),scale={width}:{height}:flags=lanczos:threads=0',
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

    fps = info(video_list[0])[2]
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

    orig_w, orig_h, fps, duration, is_vfr  = info(output_path)
    print(f'@concat : orig_w, orig_h, fps, duration, is_vfr {orig_w}, {orig_h}, {fps}, {duration}, {is_vfr} {output_path}')

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

    print(f"{progress_prefix} Frame check: left={left_count} right={right_count} fps={fps:.6f}")

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

def mask_overlay(source_video: str, mask_video: str, output_path: str, background_color: str = '0x00ff00', video_args: argparse.Namespace = None) -> str:

    resolved_path = overlay_path(source_video, output_path)
    src_w, src_h, src_fps, src_duration, is_vfr = info(source_video)
    mask_w, mask_h, mask_fps, mask_duration, is_vfr = info(mask_video)

    orig_filter = f"format=rgba,fps=fps={src_fps},setpts=N/({src_fps}*TB),scale={src_w}:{src_h}:flags=lanczos"
    mask_filter = f"format=gray,fps=fps={src_fps},setpts=N/({src_fps}*TB),scale={src_w}:{src_h}:flags=lanczos,lut=a=val/255"
    bg_filter = f"format=rgba,fps=fps={src_fps},setpts=N/({src_fps}*TB),scale={src_w}:{src_h}:flags=lanczos" 

    filter_complex = (
        f"[0:v]{orig_filter}[orig];"
        f"[1:v]{mask_filter}[mask_alpha];"
        f"[orig][mask_alpha]alphamerge[alphaed];"
        f"[2:v]{bg_filter}[bg];"
        f"[bg][alphaed]overlay=shortest=1:format=auto[out]"
    )

    os.makedirs(os.path.dirname(os.path.abspath(resolved_path)) or '.', exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'auto',
        '-i', source_video,
        '-i', mask_video,
        '-f', 'lavfi', '-i', f'color=c={background_color}',  
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-t', str(src_duration),

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
        circle_img = Image.new("L", (size_hr, size_hr), 0)

        draw = ImageDraw.Draw(circle_img)
        draw.ellipse([0, 0, size_hr - 1, size_hr - 1], fill=255)

        circle_img = circle_img.resize((size, size), Image.Resampling.BILINEAR)
        circle_img = circle_img.filter(ImageFilter.GaussianBlur(radius=1))
        circle_img.save(str(mask_path))

    except ImportError:
        cmd = [

            "ffmpeg", "-y", "-f", "lavfi", "-i",
            f"color=c=white:s={size}x{size}:d=1,format=gray",
            "-vf",
            "geq=lum='if(lte(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W,H)/2,2)),255,0)'",
            "-frames:v", "1",
            str(mask_path),
        ]
        
        subprocess.run(cmd, capture_output=True, text=True)

    return str(mask_path)

def discover_input_pairs(input_path: str) -> list[tuple[Path, Path]]:
    path = Path(input_path).expanduser().resolve()

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

def create_alpha_pack_command(
    video_path: str,
    mask_path: str,
    output_path: str,
    video_dims: tuple[int, int],
) -> list[str]:

    video_w, video_h = video_dims
    out_h = _ceil_to(video_h, 32)

    if video_w == 2 * video_h:
        out_w = 2 * out_h
    else:
        out_w = _ceil_to(video_w, 32)

    if (out_w, out_h) != (video_w, video_h):
        print(f"NVENC-aligned output: {out_w}x{out_h}")

    overlay_size = int(out_h * 0.4)
    overlay_size = (overlay_size // 4) * 4
    half_overlay = overlay_size // 2

    if video_h <= 2400:
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

    circle_mask = get_circle_mask(overlay_size)

    filter_parts: list[str] = [

        f"[0:v]scale=w={out_w}:h={out_h}:flags=bicubic,format=yuv420p[vid]",
        "[1:v]split=2[mask1][mask2]",
        "[2:v]format=gray,split=2[circle_l][circle_r]",

        (
            f"[mask1]crop=ih:ih:0:0,"
            f"scale={overlay_size}:{overlay_size}:flags=bicubic,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[left_scaled]"
        ),
        "[left_scaled][circle_l]alphamerge,format=rgba[left_circle]",

        (
            f"[mask2]crop=ih:ih:iw-ih:0,"
            f"scale={overlay_size}:{overlay_size}:flags=bicubic,"
            f"{erosion_filter}"
            f"gblur=sigma={sigma},eq=contrast={contrast}:gamma={gamma},"
            "format=gbrp[right_scaled]"
        ),
        "[right_scaled][circle_r]alphamerge,format=rgba[right_circle]",

        "[left_circle]split=2[left_for_top][left_for_bottom]",
        f"[left_for_top]crop={overlay_size}:{half_overlay}:0:0,format=yuva420p[left_top]",
        f"[left_for_bottom]crop={overlay_size}:{half_overlay}:0:{half_overlay},format=yuva420p[left_bottom]",

        "[right_circle]split=4[r1][r2][r3][r4]",
        f"[r1]crop={half_overlay}:{half_overlay}:0:0,format=yuva420p[right_tl]",
        f"[r2]crop={half_overlay}:{half_overlay}:{half_overlay}:0,format=yuva420p[right_tr]",
        f"[r3]crop={half_overlay}:{half_overlay}:0:{half_overlay},format=yuva420p[right_bl]",
        f"[r4]crop={half_overlay}:{half_overlay}:{half_overlay}:{half_overlay},format=yuva420p[right_br]",

        "[vid][left_top]overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h[v1]",
        "[v1][left_bottom]overlay=x=(main_w-overlay_w)/2:y=0[v2]",
        "[v2][right_tl]overlay=x=main_w-overlay_w:y=main_h-overlay_h[v3]",
        "[v3][right_tr]overlay=x=0:y=main_h-overlay_h[v4]",
        "[v4][right_bl]overlay=x=main_w-overlay_w:y=0[v5]",
        "[v5][right_br]overlay=x=0:y=0[out]",
    ]

    filter_complex = ";".join(filter_parts)

    cmd: list[str] = [

        'ffmpeg', '-y', '-hide_banner',
        "-filter_threads", "0",
        "-threads", "0",
        "-i", video_path,
        "-i", mask_path,
        "-loop", "1",
        "-i", circle_mask,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-shortest",
        *encoder_args(),
        output_path,
    ]

    return cmd

def pack_video(
        
    video_path: str,
    mask_path: str,
    output_path: str | None = None,
    sync_frames = None,
    progress_prefix: str = "[ALPHA] "

) -> int:
  
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

    video_w, video_h, *_ = info(video_path)
    mask_w, mask_h, *_ = info(actual_mask)

    encoder = "hevc_nvenc"

    print(f"Video: {video_w}x{video_h}")
    print(f"Encoder: {encoder}")
    print(f"Mask: {mask_w}x{mask_h}")

    cmd = create_alpha_pack_command(video_path, actual_mask, output_path, (video_w, video_h))
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

    print(f" Failed to pack {video_path}")

    return output_path

def packer(input_path: str, sync_frames=None) -> int:
    input_pairs = discover_input_pairs(input_path)

    processed = []
    
    for index, (video_path, mask_path) in enumerate(input_pairs, 1):

        print(f"[{index}/{len(input_pairs)}] Processing: {video_path.name} <- {mask_path.name}")
        rc = pack_video(str(video_path), str(mask_path), sync_frames=None)

        if rc == 0:
            processed.append((str(video_path), str(mask_path)))
        print()

    if not processed:
        print(" No files were packed successfully")
        return 1

    for video_path, mask_path in processed:
        print(f"{video_path} <- {mask_path}")

    return 0

def sync_mask_to_video(mask_path: str, fps: float, frame_offset: int = 0) -> str:

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
        *encoder_args(),
        synced_path,
    ]

    ffmpeg_progress(cmd)

    return synced_path

FISHEYE180_PIPELINE_MODE = '__NO_MASK__'

def fisheye180(input_video: str, mask_path: str | None = None) -> str:

    print('Starting FISHEYE180 conversion...')

    input_video = str(Path(input_video).expanduser().resolve())
    filename, ext = os.path.splitext(input_video)
    output_video = f'{filename}_FISHEYE180{ext}'
    target_w, target_h, fps, duration, is_vfr  = info(input_video)
    eye_w = target_w // 2

    if eye_w <= 0 or target_h <= 0:
        raise RuntimeError(f'Invalid input dimensions for fisheye conversion: {target_w}x{target_h}')

    filter_parts = [
        f'[0:v]fps={fps},setpts=N/({fps}*TB),split=2[left_src][right_src]',
        f'[left_src]crop=iw/2:ih:0:0,v360=hequirect:fisheye:w={eye_w}:h={target_h}[left]',
        f'[right_src]crop=iw/2:ih:iw/2:0,v360=hequirect:fisheye:w={eye_w}:h={target_h}[right]',
        f'[left][right]hstack,scale=w={target_w}:h={target_h}:flags=bilinear,format=yuv420p[stacked]',
    ]

    if mask_path is not None:
        mask_path = str(Path(mask_path).expanduser().resolve())
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f'Fisheye mask not found: {mask_path}')
        if Path(mask_path).suffix.lower() != '.png':
            raise RuntimeError(f'Fisheye mask must be a PNG image: {mask_path}')

        filter_parts.extend([
            '[1:v]format=rgba[mask_src]',
            '[mask_src][stacked]scale2ref[mask][stacked_ref]',
            '[stacked_ref][mask]overlay=0:0:format=auto[out]',
        ])
    else:
        filter_parts.append('[stacked]copy[out]')

    filter_complex = ';'.join(filter_parts)

    cmd = [
        'ffmpeg', '-y', '-hide_banner',
        '-i', input_video,
    ]

    if mask_path is not None:
        cmd.extend(['-i', mask_path])

    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-map', '0:a?',
        *encoder_args(),
        output_video,
    ])

    rc, _ = ffmpeg_progress(cmd)

    if rc != 0:
        raise RuntimeError(f'FFmpeg failed with exit code {rc}')

    print(f'FISHEYE180 output: {output_video}')
    return output_video

def run_fisheye180_mode(input_path: str, mask_path: str | None = None) -> int:
    video_paths = _input_videos(input_path)
    outputs: list[str] = []

    for index, video_path in enumerate(video_paths, 1):
        print(f'[{index}/{len(video_paths)}] FISHEYE180: {video_path}')
        output_path = fisheye180(str(video_path), mask_path=mask_path)
        outputs.append(output_path)
        print()

    print('=' * 60)
    print('FISHEYE180 conversion complete')
    print('=' * 60)
    for output_path in outputs:
        print(output_path)

    return 0
