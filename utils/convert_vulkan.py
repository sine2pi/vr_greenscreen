import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, threading, subprocess, json, functools, shutil

FFMPEG_BIN = shutil.which('ffmpeg') or 'ffmpeg'
FFPROBE_BIN = shutil.which('ffprobe') or 'ffprobe'

def have(a):
    return a is not None and str(a).strip() != ""

def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, b) if not have(c) else c

@functools.lru_cache(maxsize=128)
def get_video_info(file_path):
    cmd = [
        FFPROBE_BIN,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_name,r_frame_rate',
        '-of', 'json',
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    if 'streams' not in info or not info['streams']:
        raise ValueError(f"No video streams found in {file_path}")
    stream = info['streams'][0]
    fps_str = stream.get('r_frame_rate', '30/1')
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom if denom != 0 else 30.0
    codec = stream.get('codec_name', '')
    width = int(stream.get('width', 0))
    height = int(stream.get('height', 0))

    return {
        'fps': fps,
        'codec': codec,
        'width': width,
        'height': height
    }

def frame_rate(file_path):
    return get_video_info(file_path)['fps']

def vcodec(file_path):
    return get_video_info(file_path)['codec']

def vresolution(file_path):
    info = get_video_info(file_path)
    return info['width'], info['height']

def run_process(cmd, log_callback, process_callback=None):
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    else:
        print(f"Executing: {' '.join(cmd)}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, errors='replace')
    if process_callback: process_callback(process)

    for line in process.stdout:
        if log_callback: log_callback(line.strip())
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with code {process.returncode}")

CODEC_OPTIONS_DICT = {}
json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video-encoders.json')
if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            encoders_data = json.load(f)
            for enc in encoders_data:
                if 'bitrateOpts' in enc and 'cbr' not in enc['bitrateOpts']:
                    continue
                if 'os' in enc and 'windows' not in enc['os']:
                    continue
                CODEC_OPTIONS_DICT[enc['id']] = enc
    except Exception as e:
        print(f"Failed to load video-encoders.json: {e}")

CODEC_OPTIONS = list(CODEC_OPTIONS_DICT.keys()) if CODEC_OPTIONS_DICT else ["h265-main10-win-nvidia"]
DEFAULT_CODEC = "h265-main10-win-nvidia"
if DEFAULT_CODEC not in CODEC_OPTIONS:
    DEFAULT_CODEC = CODEC_OPTIONS[0]

def _ffmpeg_has_encoder(encoder_name):
    try:
        result = subprocess.run([FFMPEG_BIN, '-hide_banner', '-encoders'], capture_output=True, text=True, check=False)
    except Exception:
        return True
    output = (result.stdout or '') + (result.stderr or '')
    return encoder_name in output

def _ffmpeg_has_filter(filter_name):
    try:
        result = subprocess.run([FFMPEG_BIN, '-hide_banner', '-filters'], capture_output=True, text=True, check=False)
    except Exception:
        return True
    output = (result.stdout or '') + (result.stderr or '')
    return filter_name in output

def get_v360_filter_name(use_vulkan_filter=False):
    if use_vulkan_filter and _ffmpeg_has_filter("v360_vulkan"):
        return "v360_vulkan"
    return "v360"

def get_codec_opts(out_codec, bitrate, include_audio=False):
    opts = ["-fps_mode", "cfr"]

    if out_codec in CODEC_OPTIONS_DICT:
        enc = CODEC_OPTIONS_DICT[out_codec]
        ffmpeg_opts = enc.get('ffmpegOpts', '')
        if ffmpeg_opts:
            opts.extend(ffmpeg_opts.split())

        if 'bitrateOpts' in enc and 'cbr' in enc['bitrateOpts']:
            cbr_opts = enc['bitrateOpts']['cbr'].replace("<CONST_BITRATE_VALUE>", bitrate)
            opts.extend(cbr_opts.split())
    else:
        if _ffmpeg_has_encoder("hevc_nvenc"):
            opts.extend(["-c:v", "hevc_nvenc", "-profile:v", "main10", "-pix_fmt", "p010le", "-b_ref_mode", "disabled", "-tag:v", "hvc1", "-g", "30", "-rc", "cbr", "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate])
        elif _ffmpeg_has_encoder("libx265"):
            opts.extend(["-c:v", "libx265", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p"])
        else:
            opts.extend(["-c:v", "hevc", "-pix_fmt", "yuv420p"])

    if include_audio:
        opts.extend(["-c:a", "aac", "-b:a", "256k"])

    opts.extend(["-avoid_negative_ts", "make_zero", "-fflags", "+genpts"])
    return opts

def get_ext_from_codec(out_codec, original_ext):
    if out_codec in CODEC_OPTIONS_DICT:
        return "." + CODEC_OPTIONS_DICT[out_codec].get("ext", [original_ext.lstrip('.')])[0]
    return original_ext

def split_video(input_path, mode, output_dir, conversion="none", log_callback=None, process_callback=None, bitrate="100M", fps=None, height=None, width=None, out_codec="h265-main10-win-nvidia", input_path2=None, pack_scale=0.40, padding=0, use_vulkan_filter=False):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fps = frame_rate(input_path) if not have(fps) else fps
    filename = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    ext = get_ext_from_codec(out_codec, ext)
    width = 'iw' if not have(width) else width
    height = 'ih' if not have(height) else height

    if log_callback: log_callback(f"Detecting codec for {input_path}...")
    codec = vcodec(input_path)
    w_in, h_in = vresolution(input_path)
    wi, hi = w_in, h_in

    if mode in ('left_and_right', 'left', 'right'):
        wi = w_in // 2 if w_in > 0 else 0
    elif mode in ('top_and_bottom', 'top', 'bottom'):
        wi = w_in // 2 if w_in > 0 else 0
        hi = h_in // 2 if h_in > 0 else 0
    elif mode == 'pack':
        wi = w_in // 2 if w_in > 0 else 0

    dim_str = f":w={wi}:h={hi}" if wi > 0 and hi > 0 else ""

    decoder_opts = []
    if codec == 'h264':
        decoder_opts = ["-hwaccel", "auto"]
    elif codec == 'hevc':
        decoder_opts = ["-hwaccel", "auto"]

    v360_filter = ""
    fisheye_suffix = ""
    filter_name = get_v360_filter_name(use_vulkan_filter)
    if conversion == "to_fisheye":
        v360_filter = f",{filter_name}=hequirect:fisheye{dim_str}"
        fisheye_suffix = "_FE180"
    elif conversion == "to_fisheye190":
        v360_filter = f",{filter_name}=hequirect:fisheye:d_fov=190{dim_str}"
        fisheye_suffix = "_FE190"
    elif conversion == "to_hequirect":
        v360_filter = f",{filter_name}=fisheye:hequirect{dim_str}"
        fisheye_suffix = "_180"
    elif conversion == "heq_to_flat":
        v360_filter = f",{filter_name}=hequirect:sg:v_fov=60:h_fov=60{dim_str}"
        fisheye_suffix = "_flat"
    elif conversion == "fish_to_flat":
        v360_filter = f",{filter_name}=fisheye:sg:v_fov=60:h_fov=60{dim_str}"
        fisheye_suffix = "_flat"

    if mode == 'pack':
        if not input_path2 or not os.path.exists(input_path2):
            raise ValueError("mode='pack' requires a valid input_path2 (mask SBS video)")

        codec2 = vcodec(input_path2)
        decoder_opts2 = []
        if codec2 in ('h264', 'hevc'):
            decoder_opts2 = ["-hwaccel", "auto"]

        eye_w = wi
        eye_h = hi
        sbs_w = eye_w * 2

        target_w = int(eye_w * pack_scale)
        target_h = int(eye_h * pack_scale)
        h_half = target_h // 2
        w_half = target_w // 2
        x_mid = sbs_w // 2 - target_w // 2

        ov_y_top = padding
        ov_y_bot = eye_h - padding - h_half
        ov_x_right = sbs_w - padding - w_half
        ov_x_left = padding

        parts = []
        parts.append(f"[0:v]fps={fps},setpts=N/({fps}*TB),split=2[b1][b2]")
        parts.append(f"[b1]crop=iw/2:ih:0:0{v360_filter}[bl]")
        parts.append(f"[b2]crop=iw/2:ih:iw/2:0{v360_filter}[br]")
        parts.append(f"[bl][br]hstack[base]")

        parts.append(f"[1:v]fps={fps},setpts=N/({fps}*TB),split=2[m1][m2]")
        parts.append(f"[m1]crop=iw/2:ih:0:0{v360_filter},scale={target_w}:{target_h}:flags=bicubic,split=2[ml1][ml2]")
        parts.append(f"[m2]crop=iw/2:ih:iw/2:0{v360_filter},scale={target_w}:{target_h}:flags=bicubic,split=4[mr1][mr2][mr3][mr4]")

        parts.append(f"[ml1]crop={target_w}:{h_half}:0:{h_half}[ml_bottom]")
        parts.append(f"[ml2]crop={target_w}:{h_half}:0:0[ml_top]")

        parts.append(f"[mr1]crop={w_half}:{h_half}:0:0[mr_tl]")
        parts.append(f"[mr2]crop={w_half}:{h_half}:{w_half}:0[mr_tr]")
        parts.append(f"[mr3]crop={w_half}:{h_half}:0:{h_half}[mr_bl]")
        parts.append(f"[mr4]crop={w_half}:{h_half}:{w_half}:{h_half}[mr_br]")

        parts.append(f"[base][ml_bottom]overlay={x_mid}:{ov_y_top}[o1]")
        parts.append(f"[o1][ml_top]overlay={x_mid}:{ov_y_bot}[o2]")
        parts.append(f"[o2][mr_bl]overlay={ov_x_right}:{ov_y_top}[o3]")
        parts.append(f"[o3][mr_br]overlay={ov_x_left}:{ov_y_top}[o4]")
        parts.append(f"[o4][mr_tl]overlay={ov_x_right}:{ov_y_bot}[o5]")
        parts.append(f"[o5][mr_tr]overlay={ov_x_left}:{ov_y_bot},scale=w={width}:h={height}:flags=bicubic[packed]")

        filter_complex = ";".join(parts)
        output_file = os.path.join(output_dir, f"{filename}_ALPHA{fisheye_suffix}{ext}")

        cmd = [FFMPEG_BIN, "-hide_banner", "-y"]
        cmd.extend(decoder_opts)
        cmd.extend(["-i", input_path])
        cmd.extend(decoder_opts2)
        cmd.extend(["-i", input_path2])
        cmd.extend(["-sws_flags", "bicubic+full_chroma_int+accurate_rnd+full_chroma_inp"])
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[packed]", "-map", "0:a?"])
        cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=True))
        cmd.extend(["-movflags", "+faststart+write_colr+use_metadata_tags"])
        cmd.extend([output_file])

        if log_callback: log_callback(f"Starting pack (conv={conversion}, scale={pack_scale}, padding={padding}) for {input_path}...")
        run_process(cmd, log_callback, process_callback)

    elif mode == 'left_and_right':

        output_left = os.path.join(output_dir, f"{filename}_L{fisheye_suffix}{ext}")
        output_right = os.path.join(output_dir, f"{filename}_R{fisheye_suffix}{ext}")

        filter_complex = f"[0:v]fps={fps},setpts=N/({fps}*TB),split=2[v1][v2];[v1]crop=iw/2:ih:0:(ih-iw/2)/2{v360_filter},scale=w={width}:h={height}:flags=bicubic[left];[v2]crop=iw/2:ih:iw/2:(ih-iw/2)/2{v360_filter},scale=w={width}:h={height}:flags=bicubic[right]"

        cmd = [FFMPEG_BIN, "-hide_banner", "-y"]
        cmd.extend(decoder_opts)
        cmd.extend(["-i", input_path])
        cmd.extend(["-sws_flags", "bicubic+full_chroma_int+accurate_rnd+full_chroma_inp"])
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[left]", "-map", "0:a?"])
        cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=True))
        cmd.extend(["-movflags", "+faststart+write_colr+use_metadata_tags"])
        cmd.extend([output_left])

        cmd.extend(["-map", "[right]", "-map", "0:a?"])
        cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=True))
        cmd.extend(["-movflags", "+faststart+write_colr+use_metadata_tags"])
        cmd.extend([output_right])

        if log_callback: log_callback(f"Starting split (left_and_right, conv={conversion}) for {input_path}...")
        run_process(cmd, log_callback, process_callback)

    elif mode == 'top_and_bottom':
        output_top = os.path.join(output_dir, f"{filename}_T{fisheye_suffix}{ext}")
        output_bottom = os.path.join(output_dir, f"{filename}_B{fisheye_suffix}{ext}")

        filter_complex = f"[0:v]fps={fps},setpts=N/({fps}*TB),split=2[v1][v2];[v1]crop=ih/2:ih/2:(iw-ih/2)/2:0{v360_filter},scale=w={width}:h={height}:flags=bicubic[top];[v2]crop=ih/2:ih/2:(iw-ih/2)/2:ih/2{v360_filter},scale=w={width}:h={height}:flags=bicubic[bottom]"

        cmd = [FFMPEG_BIN, "-hide_banner", "-y"]
        cmd.extend(decoder_opts)
        cmd.extend(["-i", input_path])
        cmd.extend(["-sws_flags", "bicubic+full_chroma_int+accurate_rnd+full_chroma_inp"])
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[top]", "-map", "0:a?"])

        cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=True))
        cmd.extend(["-movflags", "+faststart+write_colr+use_metadata_tags"])
        cmd.extend([output_top])
        cmd.extend(["-map", "[bottom]", "-map", "0:a?"])

        cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=True))
        cmd.extend(["-movflags", "+faststart+write_colr+use_metadata_tags"])
        cmd.extend([output_bottom])

        if log_callback: log_callback(f"Starting split (top_and_bottom, conv={conversion}) for {input_path}...")
        run_process(cmd, log_callback, process_callback)

    else:
        if mode == 'left':
            crop_filter = f"crop=iw/2:ih:0:0{v360_filter}"
            suffix = '_L'
        elif mode == 'right':
            crop_filter = f"crop=iw/2:ih:iw/2:0{v360_filter}"
            suffix = '_R'
        elif mode == 'top':
            crop_filter = f"crop=iw/2:ih/2:iw/4:0{v360_filter}"
            suffix = '_T'
        elif mode == 'bottom':
            crop_filter = f"crop=ih/2:ih/2:(iw-ih/2)/2:ih/2{v360_filter}"
            suffix = '_B'
        else:
            raise ValueError(f"Unknown split mode: {mode}")

        output_file = os.path.join(output_dir, f"{filename}{suffix}{fisheye_suffix}{ext}")

        w_target = int(width) if have(width) and width != 'iw' else w_in
        if not have(width) or width == 'iw':
            if mode in ('left', 'right', 'top'): w_target = w_in // 2
            elif mode == 'bottom': w_target = h_in // 2

        h_target = int(height) if have(height) and height != 'ih' else h_in
        if not have(height) or height == 'ih':
            if mode in ('left', 'right'): h_target = h_in
            elif mode in ('top', 'bottom'): h_target = h_in // 2

        fps_filter = f"fps={fps}"
        audio_map = ["-c:a", "copy"]
        fps_opts = []
        ffmpeg_bin = FFMPEG_BIN

        if have(fps):
            filter_complex = f"[0:v]{crop_filter},{fps_filter},setpts=N/({fps}*TB),scale=w={width}:h={height}:flags=bicubic[v]"
        else:
            filter_complex = f"[0:v]{crop_filter},scale=w={width}:h={height}:flags=bicubic[v]"

        cmd = [ffmpeg_bin, "-hide_banner", "-y"]
        cmd.extend(decoder_opts)
        cmd.extend(["-i", input_path])
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[v]", "-map", "0:a?"])
        cmd.extend(audio_map)
        cmd.extend(fps_opts)
        cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=False))
        cmd.extend(["-movflags", "+faststart"])
        cmd.extend([output_file])

        if log_callback: log_callback(f"Starting split ({mode}, conv={conversion}) for {input_path}...")
        run_process(cmd, log_callback, process_callback)

    return True

def combine_video(input_path_1, input_path_2, mode, output_path, conversion="none", log_callback=None, process_callback=None, bitrate="100M", mask_path=None, fps=None, height=None, width=None, out_codec="h265-main10-win-nvidia", use_vulkan_filter=False):

    if log_callback: log_callback(f"Combining {input_path_1} and {input_path_2} into {output_path} (Mode: {mode}, Conv: {conversion})")

    fps = frame_rate(input_path_1) if not have(fps) else fps
    codec1 = vcodec(input_path_1)
    w_in, h_in = vresolution(input_path_1)

    # out_w = w_in * 2 if mode == "left_right" else w_in
    # out_h = h_in * 2 if mode == "top_bottom" else h_in

    width = 'iw' if not have(width) else width
    height = 'ih' if not have(height) else height

    dim_str = f":w={w_in}:h={h_in}" if w_in > 0 and h_in > 0 else ""

    input_opts_1 = []
    if codec1 in ('h264', 'hevc'):
        input_opts_1 = ["-hwaccel", "auto"]

    codec2 = vcodec(input_path_2)
    input_opts_2 = []
    if codec2 in ('h264', 'hevc'):
        input_opts_2 = ["-hwaccel", "auto"]

    cmd = [FFMPEG_BIN, "-hide_banner", "-y"]
    cmd.extend(input_opts_1)
    cmd.extend(["-i", input_path_1])
    if mask_path is not None:
        cmd.extend(["-i", mask_path])
    cmd.extend(input_opts_2)
    cmd.extend(["-i", input_path_2])

    v360_filter = ""
    filter_name = get_v360_filter_name(use_vulkan_filter)
    if conversion == "to_fisheye":
        v360_filter = f"{filter_name}=hequirect:fisheye{dim_str},"
    elif conversion == "to_fisheye190":
        v360_filter = f"{filter_name}=hequirect:fisheye:d_fov=190{dim_str},"
    elif conversion == "to_hequirect":
        v360_filter = f"{filter_name}=fisheye:hequirect{dim_str},"
    elif conversion == "heq_to_flat":
        v360_filter = f"{filter_name}=hequirect:sg:v_fov=60:h_fov=60{dim_str},"
    elif conversion == "fish_to_flat":
        v360_filter = f"{filter_name}=fisheye:sg:v_fov=60:h_fov=60{dim_str},"

    if mask_path is not None:
        idx2 = 2
    else:
        idx2 = 1

    if mode == 'left_right':
        base_filter = f"[0:v]{v360_filter}setpts=PTS-STARTPTS[left];[{idx2}:v]{v360_filter}setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2"
    elif mode == 'top_bottom':
        base_filter = f"[0:v]{v360_filter}setpts=PTS-STARTPTS[top];[{idx2}:v]{v360_filter}setpts=PTS-STARTPTS[bottom];[top][bottom]vstack=inputs=2"
    else:
        raise ValueError(f"Unknown combine mode: {mode}")

    fps_filter = f"fps={fps}"
    audio_map = ["-c:a", "copy"]
    fps_opts = []
    ffmpeg_bin = FFMPEG_BIN

    if have(fps):
        if mask_path is not None:
            filter_complex = f"{base_filter}[stacked];[1:v][stacked]scale2ref[mask][stacked_ref];[stacked_ref][mask]overlay=0:0,{fps_filter},setpts=N/({fps}*TB),scale=w={width}:h={height}:flags=bicubic[v]"
        else:
            filter_complex = f"{base_filter},{fps_filter},setpts=N/({fps}*TB),scale=w={width}:h={height}:flags=bicubic[v]"
    else:
        if mask_path is not None:
            filter_complex = f"{base_filter}[stacked];[1:v][stacked]scale2ref[mask][stacked_ref];[stacked_ref][mask]overlay=0:0,scale=w={width}:h={height}:flags=bicubic[v]"
        else:
            filter_complex = f"{base_filter},scale=w={width}:h={height}:flags=bicubic[v]"

    cmd = [ffmpeg_bin, "-hide_banner", "-y"]
    cmd.extend(input_opts_1)
    cmd.extend(["-i", input_path_1])
    if mask_path is not None:
        cmd.extend(["-i", mask_path])
    cmd.extend(input_opts_2)
    cmd.extend(["-i", input_path_2])

    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "[v]", "-map", "0:a?"])
    cmd.extend(audio_map)
    cmd.extend(fps_opts)

    cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=False))
    cmd.extend(["-movflags", "+faststart"])
    cmd.extend(["-color_range", "pc", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])

    cmd.extend([output_path])

    run_process(cmd, log_callback, process_callback)
    return True

def tb_to_sbs(input_path, output_path, conversion="none", log_callback=None, process_callback=None, bitrate="100M", operation_mode="custom_tb_to_sbs", mask_path=None, fps=None, height=None, width=None, out_codec="h265-main10-win-nvidia", use_vulkan_filter=False):
    if log_callback: log_callback(f"Converting video for {input_path} (conv={conversion}, op={operation_mode})")

    fps = frame_rate(input_path) if not have(fps) else fps
    codec = vcodec(input_path)
    w_in, h_in = vresolution(input_path)
    wi, hi = w_in, h_in

    if operation_mode == "sbs_to_sbs":
        wi = w_in // 2 if w_in > 0 else 0

    elif operation_mode in ("tb_to_tb", "tb_to_sbs"):

        wi = w_in // 2 if w_in > 0 else 0
        hi = h_in // 2 if h_in > 0 else 0
    elif operation_mode == "sbs_to_tb":
        wi = w_in // 2 if w_in > 0 else 0
    else:
        wi = w_in // 2 if w_in > 0 else 0
        hi = h_in // 2 if h_in > 0 else 0

    width = 'iw' if not have(width) else width
    height = 'ih' if not have(height) else height

    dim_str = f":w={wi}:h={hi}" if wi > 0 and hi > 0 else ""

    decoder_opts = []
    if codec == 'h264':
        decoder_opts = ["-hwaccel", "auto"]
    elif codec == 'hevc':
        decoder_opts = ["-hwaccel", "auto"]

    ffmpeg_bin = FFMPEG_BIN
    cmd = [ffmpeg_bin, "-hide_banner", "-y"]
    cmd.extend(decoder_opts)
    cmd.extend(["-i", input_path])
    if mask_path is not None:
        cmd.extend(["-i", mask_path])

    v360_filter = ""
    filter_name = get_v360_filter_name(use_vulkan_filter)
    if conversion == "to_fisheye":
        v360_filter = f",{filter_name}=hequirect:fisheye{dim_str}"
    elif conversion == "to_fisheye190":
        v360_filter = f",{filter_name}=hequirect:fisheye:d_fov=190{dim_str}"
    elif conversion == "to_hequirect":
        v360_filter = f",{filter_name}=fisheye:hequirect{dim_str}"
    elif conversion == "heq_to_flat":
        v360_filter = f",{filter_name}=hequirect:sg:v_fov=60:h_fov=60{dim_str}"
    elif conversion == "fish_to_flat":
        v360_filter = f",{filter_name}=fisheye:sg:v_fov=60:h_fov=60{dim_str}"

    base_filter = ""
    if operation_mode == "sbs_to_sbs":
        base_filter = (
            f"[0:v]crop=iw/2:iw/2:0:(ih-iw/2)/2{v360_filter}[left];"
            f"[0:v]crop=iw/2:iw/2:iw/2:(ih-iw/2)/2{v360_filter}[right];"
            f"[left][right]hstack=inputs=2"
        )
    elif operation_mode == "tb_to_tb":
        base_filter = (
            f"[0:v]crop=ih/2:ih/2:(iw-ih/2)/2:0{v360_filter}[top];"
            f"[0:v]crop=ih/2:ih/2:(iw-ih/2)/2:ih/2{v360_filter}[bottom];"
            f"[top][bottom]vstack=inputs=2"
        )
    elif operation_mode == "tb_to_sbs":
        base_filter = (
            f"[0:v]crop=ih/2:ih/2:(iw-ih/2)/2:0{v360_filter}[top];"
            f"[0:v]crop=ih/2:ih/2:(iw-ih/2)/2:ih/2{v360_filter}[bottom];"
            f"[top][bottom]hstack=inputs=2"
        )
    elif operation_mode == "sbs_to_tb":
        base_filter = (
            f"[0:v]crop=iw/2:iw/2:0:(ih-iw/2)/2{v360_filter}[left];"
            f"[0:v]crop=iw/2:iw/2:iw/2:(ih-iw/2)/2{v360_filter}[right];"
            f"[left][right]vstack=inputs=2"
        )
    else:
        base_filter = (
            f"[0:v]crop=ih/2:ih/2:(iw-ih/2)/2:0{v360_filter}[left];"
            f"[0:v]crop=ih/2:ih/2:(iw-ih/2)/2:ih/2{v360_filter}[right];"
            f"[left][right]hstack=inputs=2"
        )

    fps_filter = f"fps={fps}"
    audio_map = ["-map", "0:a?", "-c:a", "copy"]
    fps_opts = []

    if have(fps):
        fps_opts = ["-fps_mode", "cfr", "-r", str(fps)]

    if mask_path is not None:
        filter_complex = f"{base_filter}[stacked];[1:v][stacked]scale2ref[mask][stacked_ref];[stacked_ref][mask]overlay=0:0,{fps_filter},setpts=N/({fps}*TB),scale=w={width}:h={height}:flags=bicubic[v]"
    else:
        filter_complex = f"{base_filter},{fps_filter},setpts=N/({fps}*TB),scale=w={width}:h={height}:flags=bicubic[v]"

    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "[v]"])
    cmd.extend(audio_map)
    cmd.extend(fps_opts)

    cmd.extend(get_codec_opts(out_codec, bitrate, include_audio=False))
    cmd.extend(["-movflags", "+faststart"])

    cmd.extend(["-color_range", "pc", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])

    cmd.extend([output_path])

    run_process(cmd, log_callback, process_callback)
    return True

def batch_tb_to_sbs(input_dir, output_dir=None, conversion="none", log_callback=None, process_callback=None, bitrate="100M", operation_mode="custom_tb_to_sbs", mask_path=None, fps=None, height=None, width=None, out_codec="h265-main10-win-nvidia", use_vulkan_filter=False):

    width = 'iw' if not have(width) else width
    height = 'ih' if not have(height) else height

    import glob
    if log_callback: log_callback(f"Starting batch conversion in {input_dir}")
    mp4_files = glob.glob(os.path.join(input_dir, "*.mp4")) + glob.glob(os.path.join(input_dir, "*.mkv"))

    if not mp4_files:
        if log_callback: log_callback(f"No .mp4 or .mkv files found in {input_dir}")
        return False

    for input_path in mp4_files:
        if "_LR" in input_path or "_FE180" in input_path or "_FE190" in input_path or "_180" in input_path or ".restored" in input_path:
            continue

        filename = os.path.splitext(os.path.basename(input_path))[0]
        out_dir = output_dir if output_dir else input_dir

        suffix = ""
        if operation_mode in ("sbs_to_tb", "tb_to_tb"):
            suffix += "_TB"
        else:
            suffix += "_LR"

        if conversion == "to_fisheye":
            suffix = "_FE180" + suffix
        elif conversion == "to_fisheye190":
            suffix = "_FE190" + suffix
        elif conversion == "to_hequirect":
            suffix = "_180" + suffix
        elif conversion == "heq_to_flat" or conversion == "fish_to_flat":
            suffix = "_flat" + suffix

        ext_out = get_ext_from_codec(out_codec, ".mp4")
        output_path = os.path.join(out_dir, f"{filename}{suffix}{ext_out}")

        if os.path.exists(output_path):
            if log_callback: log_callback(f"Skipping {os.path.basename(input_path)}, output {os.path.basename(output_path)} already exists.")
            continue

        tb_to_sbs(input_path, output_path, conversion=conversion, log_callback=log_callback, process_callback=process_callback, bitrate=bitrate, operation_mode=operation_mode, mask_path=mask_path, fps=fps, height=height, width=width, out_codec=out_codec, use_vulkan_filter=use_vulkan_filter)

        get_video_info.cache_clear()

    if log_callback: log_callback("Batch conversion finished.")
    return True

LANG = 'en'

TRANSLATIONS = {
    'en': {
        'title': "VR Conversion Tool",
        'tab_split': " VR Split ",
        'tab_combine': " VR Combine ",
        'tab_convert': " VR Convert ",
        'log_title': "Log",
        'btn_back': "Back to Main Menu",

        'lbl_input_video': "Input Video:",
        'lbl_output_dir': "Output Dir (Optional):",
        'btn_browse': "Browse...",
        'grp_split_mode': "Split Mode",
        'mode_left': "Left Eye",
        'mode_right': "Right Eye",
        'mode_lr_sep': "Left & Right (Separate Files)",
        'mode_top': "Top Eye",
        'mode_bottom': "Bottom Eye",
        'mode_tb_sep': "Top & Bottom (Separate Files)",
        'mode_pack': "Pack (Overlay Masks)",
        'lbl_mask_video': "Mask Video (for Pack):",
        'lbl_pack_scale': "Pack Scale:",
        'lbl_padding': "Padding:",
        'btn_start': "Start Processing",
        'btn_stop': "Stop",
        'msg_stop': "Process stopped.",

        'lbl_input_1': "Input File 1 (Left/Top):",
        'lbl_input_2': "Input File 2 (Right/Bottom):",
        'lbl_mask_image': "Mask Image (Optional):",
        'lbl_output_file': "Output File (Optional):",
        'btn_save': "Save As...",
        'grp_combine_mode': "Combine Mode",
        'mode_sbs': "Side-by-Side (Left-Right)",
        'mode_ou': "Over-Under (Top-Bottom)",

        'msg_error_input_file': "Please select a valid input file.",
        'msg_error_input_1': "Please select Input File 1.",
        'msg_error_input_2': "Please select Input File 2.",
        'msg_error_output': "Please select output path.",
        'msg_success_split': "Split processing completed!",
        'msg_success_combine': "Combine processing completed!",
        'msg_task_complete': "Task Completed.",
        'msg_starting_split': "Starting Split... Mode: {}",
        'msg_starting_combine': "Starting Combine... Mode: {}",
        'msg_error_occurred': "An error occurred: {}",
        'title_error': "Error",
        'title_success': "Success",
        'lbl_conversion': "Projection Conversion:",
        'conv_none': "None (Keep Original)",
        'conv_to_fisheye': "To Fisheye (Hequirect -> Fisheye)",
        'conv_to_fisheye190': "To Fisheye 190 (Hequirect -> Fisheye 190°)",
        'conv_to_hequirect': "To Hequirect (Fisheye -> Hequirect)",
        'conv_heq_to_flat': "To Flat (Hequirect -> Rectilinear Wide)",
        'conv_fish_to_flat': "To Flat (Fisheye -> Rectilinear Wide)",
        'lbl_bitrate': "Bitrate:",
        'grp_convert_op': "Operation Mode",
        'op_custom_tb_sbs': "Custom Top-Bottom (25% cropped) to SBS",
        'op_sbs_sbs': "Standard SBS (Projection Convert Only)",
        'op_tb_tb': "Standard Top-Bottom (Projection Convert Only)",
        'op_tb_sbs': "Standard Top-Bottom to SBS",
        'op_sbs_tb': "Standard SBS to Top-Bottom"
    }
}

def get_text(key):
    return TRANSLATIONS.get(LANG, TRANSLATIONS['en']).get(key, key)

class VRSplitCombineApp:
    def __init__(self, root, on_return=None):
        self.root = root
        self.on_return = on_return
        self.root.title(get_text('title'))

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(header_frame, text=get_text('title'), font=('Arial', 14, 'bold')).pack(side='left')

        if self.on_return:
            ttk.Button(header_frame, text=get_text('btn_back'), command=self.go_back).pack(side='right')

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True)

        self.tab_split = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_split, text=get_text('tab_split'))
        self.setup_split_tab()

        self.tab_combine = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_combine, text=get_text('tab_combine'))
        self.setup_combine_tab()

        self.tab_convert = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_convert, text=get_text('tab_convert'))
        self.setup_convert_tab()

        log_frame = ttk.LabelFrame(self.main_frame, text=get_text('log_title'), padding=5)
        log_frame.pack(fill='both', expand=True, pady=10)
        self.log_text = tk.Text(log_frame, height=10, state='disabled')
        self.log_text.pack(fill='both', expand=True, side='left')
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)

    def go_back(self):
        if self.on_return:
            self.on_return()
        else:
            self.root.quit()

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert('end', message + "\n")
        self.log_text.see('end')
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def setup_split_tab(self):
        frame_input = ttk.Frame(self.tab_split)
        frame_input.pack(fill='x', pady=5)
        ttk.Label(frame_input, text=get_text('lbl_input_video')).pack(side='left')
        self.split_input_var = tk.StringVar()
        ttk.Entry(frame_input, textvariable=self.split_input_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_input, text=get_text('btn_browse'), command=lambda: self.browse_file(self.split_input_var)).pack(side='left')

        frame_mask = ttk.Frame(self.tab_split)
        frame_mask.pack(fill='x', pady=5)
        ttk.Label(frame_mask, text=get_text('lbl_mask_video')).pack(side='left')
        self.split_mask_var = tk.StringVar()
        ttk.Entry(frame_mask, textvariable=self.split_mask_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_mask, text=get_text('btn_browse'), command=lambda: self.browse_file(self.split_mask_var)).pack(side='left')

        frame_output = ttk.Frame(self.tab_split)
        frame_output.pack(fill='x', pady=5)
        ttk.Label(frame_output, text=get_text('lbl_output_dir')).pack(side='left')
        self.split_output_var = tk.StringVar()
        ttk.Entry(frame_output, textvariable=self.split_output_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_output, text=get_text('btn_browse'), command=lambda: self.browse_dir(self.split_output_var)).pack(side='left')

        frame_mode = ttk.LabelFrame(self.tab_split, text=get_text('grp_split_mode'), padding=10)
        frame_mode.pack(fill='x', pady=10)

        self.split_mode_var = tk.StringVar(value="left_and_right")

        modes = [
            (get_text('mode_left'), "left"),
            (get_text('mode_right'), "right"),
            (get_text('mode_lr_sep'), "left_and_right"),
            (get_text('mode_top'), "top"),
            (get_text('mode_bottom'), "bottom"),
            (get_text('mode_tb_sep'), "top_and_bottom"),
            (get_text('mode_pack'), "pack"),
        ]

        for text, val in modes:
            ttk.Radiobutton(frame_mode, text=text, variable=self.split_mode_var, value=val).pack(anchor='w')

        frame_conv = ttk.LabelFrame(self.tab_split, text=get_text('lbl_conversion'), padding=10)
        frame_conv.pack(fill='x', pady=10)
        self.split_conv_var = tk.StringVar(value="none")
        ttk.Radiobutton(frame_conv, text=get_text('conv_none'), variable=self.split_conv_var, value="none").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_fisheye'), variable=self.split_conv_var, value="to_fisheye").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_fisheye190'), variable=self.split_conv_var, value="to_fisheye190").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_hequirect'), variable=self.split_conv_var, value="to_hequirect").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_heq_to_flat'), variable=self.split_conv_var, value="heq_to_flat").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_fish_to_flat'), variable=self.split_conv_var, value="fish_to_flat").pack(anchor='w')

        frame_bitrate = ttk.Frame(self.tab_split)
        frame_bitrate.pack(fill='x', pady=5)
        ttk.Label(frame_bitrate, text=get_text('lbl_bitrate')).pack(side='left')
        self.split_bitrate_var = tk.StringVar(value="100M")
        ttk.Entry(frame_bitrate, textvariable=self.split_bitrate_var, width=10).pack(side='left', padx=5)

        ttk.Label(frame_bitrate, text="Output Codec:").pack(side='left', padx=(10, 0))
        self.split_codec_var = tk.StringVar(value=DEFAULT_CODEC)
        ttk.OptionMenu(frame_bitrate, self.split_codec_var, DEFAULT_CODEC, *CODEC_OPTIONS).pack(side='left', padx=5)

        frame_params = ttk.Frame(self.tab_split)
        frame_params.pack(fill='x', pady=5)
        self.use_vulkan_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_params, text="Use Vulkan 360 filter (if available)", variable=self.use_vulkan_filter_var).pack(side='left', padx=(10, 0))
        ttk.Label(frame_params, text="FPS (Optional):").pack(side='left')
        self.split_fps_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.split_fps_var, width=8).pack(side='left', padx=5)

        ttk.Label(frame_params, text="Width (Optional):").pack(side='left')
        self.split_width_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.split_width_var, width=8).pack(side='left', padx=5)

        ttk.Label(frame_params, text="Height (Optional):").pack(side='left')
        self.split_height_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.split_height_var, width=8).pack(side='left', padx=5)

        frame_pack_params = ttk.Frame(self.tab_split)
        frame_pack_params.pack(fill='x', pady=5)
        ttk.Label(frame_pack_params, text=get_text('lbl_pack_scale')).pack(side='left')
        self.split_pack_scale_var = tk.StringVar(value="0.40")
        ttk.Entry(frame_pack_params, textvariable=self.split_pack_scale_var, width=8).pack(side='left', padx=5)
        ttk.Label(frame_pack_params, text=get_text('lbl_padding')).pack(side='left')
        self.split_padding_var = tk.StringVar(value="0")
        ttk.Entry(frame_pack_params, textvariable=self.split_padding_var, width=8).pack(side='left', padx=5)

        btn_frame = ttk.Frame(self.tab_split)
        btn_frame.pack(pady=10)

        self.btn_split_start = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_split)
        self.btn_split_start.pack(side='left', padx=5)

        self.btn_split_stop = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_split, state='disabled')
        self.btn_split_stop.pack(side='left', padx=5)

        self.proc_split = None

    def setup_combine_tab(self):
        frame_in1 = ttk.Frame(self.tab_combine)
        frame_in1.pack(fill='x', pady=5)
        ttk.Label(frame_in1, text=get_text('lbl_input_1')).pack(side='left')
        self.combine_in1_var = tk.StringVar()
        ttk.Entry(frame_in1, textvariable=self.combine_in1_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_in1, text=get_text('btn_browse'), command=lambda: self.browse_file(self.combine_in1_var)).pack(side='left')

        frame_in2 = ttk.Frame(self.tab_combine)
        frame_in2.pack(fill='x', pady=5)
        ttk.Label(frame_in2, text=get_text('lbl_input_2')).pack(side='left')
        self.combine_in2_var = tk.StringVar()
        ttk.Entry(frame_in2, textvariable=self.combine_in2_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_in2, text=get_text('btn_browse'), command=lambda: self.browse_file(self.combine_in2_var)).pack(side='left')

        frame_mask = ttk.Frame(self.tab_combine)
        frame_mask.pack(fill='x', pady=5)
        ttk.Label(frame_mask, text=get_text('lbl_mask_image')).pack(side='left')
        self.combine_mask_var = tk.StringVar()
        ttk.Entry(frame_mask, textvariable=self.combine_mask_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_mask, text=get_text('btn_browse'), command=lambda: self.browse_file(self.combine_mask_var)).pack(side='left')

        frame_out = ttk.Frame(self.tab_combine)
        frame_out.pack(fill='x', pady=5)
        ttk.Label(frame_out, text=get_text('lbl_output_file')).pack(side='left')
        self.combine_out_var = tk.StringVar()
        ttk.Entry(frame_out, textvariable=self.combine_out_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_out, text=get_text('btn_save'), command=lambda: self.save_file(self.combine_out_var)).pack(side='left')

        frame_mode = ttk.LabelFrame(self.tab_combine, text=get_text('grp_combine_mode'), padding=10)
        frame_mode.pack(fill='x', pady=10)
        self.combine_mode_var = tk.StringVar(value="left_right")
        ttk.Radiobutton(frame_mode, text=get_text('mode_sbs'), variable=self.combine_mode_var, value="left_right").pack(anchor='w')
        ttk.Radiobutton(frame_mode, text=get_text('mode_ou'), variable=self.combine_mode_var, value="top_bottom").pack(anchor='w')

        frame_conv = ttk.LabelFrame(self.tab_combine, text=get_text('lbl_conversion'), padding=10)
        frame_conv.pack(fill='x', pady=10)
        self.combine_conv_var = tk.StringVar(value="none")
        ttk.Radiobutton(frame_conv, text=get_text('conv_none'), variable=self.combine_conv_var, value="none").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_fisheye'), variable=self.combine_conv_var, value="to_fisheye").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_fisheye190'), variable=self.combine_conv_var, value="to_fisheye190").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_hequirect'), variable=self.combine_conv_var, value="to_hequirect").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_heq_to_flat'), variable=self.combine_conv_var, value="heq_to_flat").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_fish_to_flat'), variable=self.combine_conv_var, value="fish_to_flat").pack(anchor='w')

        frame_bitrate = ttk.Frame(self.tab_combine)
        frame_bitrate.pack(fill='x', pady=5)
        ttk.Label(frame_bitrate, text=get_text('lbl_bitrate')).pack(side='left')
        self.combine_bitrate_var = tk.StringVar(value="100M")
        ttk.Entry(frame_bitrate, textvariable=self.combine_bitrate_var, width=10).pack(side='left', padx=5)

        ttk.Label(frame_bitrate, text="Output Codec:").pack(side='left', padx=(10, 0))
        self.combine_codec_var = tk.StringVar(value=DEFAULT_CODEC)
        ttk.OptionMenu(frame_bitrate, self.combine_codec_var, DEFAULT_CODEC, *CODEC_OPTIONS).pack(side='left', padx=5)

        frame_params = ttk.Frame(self.tab_combine)
        frame_params.pack(fill='x', pady=5)
        self.use_vulkan_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_params, text="Use Vulkan 360 filter (if available)", variable=self.use_vulkan_filter_var).pack(side='left', padx=(10, 0))
        ttk.Label(frame_params, text="FPS (Optional):").pack(side='left')
        self.combine_fps_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.combine_fps_var, width=8).pack(side='left', padx=5)

        ttk.Label(frame_params, text="Width (Optional):").pack(side='left')
        self.combine_width_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.combine_width_var, width=8).pack(side='left', padx=5)

        ttk.Label(frame_params, text="Height (Optional):").pack(side='left')
        self.combine_height_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.combine_height_var, width=8).pack(side='left', padx=5)

        btn_frame = ttk.Frame(self.tab_combine)
        btn_frame.pack(pady=10)

        self.btn_combine_start = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_combine)
        self.btn_combine_start.pack(side='left', padx=5)

        self.btn_combine_stop = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_combine, state='disabled')
        self.btn_combine_stop.pack(side='left', padx=5)

        self.proc_combine = None

    def setup_convert_tab(self):
        frame_in = ttk.Frame(self.tab_convert)
        frame_in.pack(fill='x', pady=5)
        ttk.Label(frame_in, text="Input File or Directory:").pack(side='left')
        self.convert_in_var = tk.StringVar()
        ttk.Entry(frame_in, textvariable=self.convert_in_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_in, text="Browse File...", command=lambda: self.browse_file(self.convert_in_var)).pack(side='left', padx=2)
        ttk.Button(frame_in, text="Browse Dir...", command=lambda: self.browse_dir(self.convert_in_var)).pack(side='left', padx=2)

        frame_mask = ttk.Frame(self.tab_convert)
        frame_mask.pack(fill='x', pady=5)
        ttk.Label(frame_mask, text=get_text('lbl_mask_image')).pack(side='left')
        self.convert_mask_var = tk.StringVar()
        ttk.Entry(frame_mask, textvariable=self.convert_mask_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_mask, text="Browse File...", command=lambda: self.browse_file(self.convert_mask_var)).pack(side='left', padx=2)

        frame_out = ttk.Frame(self.tab_convert)
        frame_out.pack(fill='x', pady=5)
        ttk.Label(frame_out, text="Output File or Directory (Optional):").pack(side='left')
        self.convert_out_var = tk.StringVar()
        ttk.Entry(frame_out, textvariable=self.convert_out_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(frame_out, text="Browse File/Dir...", command=lambda: self.save_file(self.convert_out_var)).pack(side='left', padx=2)
        ttk.Button(frame_out, text="Browse Dir...", command=lambda: self.browse_dir(self.convert_out_var)).pack(side='left', padx=2)

        frame_conv = ttk.LabelFrame(self.tab_convert, text=get_text('lbl_conversion'), padding=10)
        frame_conv.pack(fill='x', pady=10)
        self.convert_conv_var = tk.StringVar(value="none")
        ttk.Radiobutton(frame_conv, text=get_text('conv_none'), variable=self.convert_conv_var, value="none").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_fisheye'), variable=self.convert_conv_var, value="to_fisheye").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_fisheye190'), variable=self.convert_conv_var, value="to_fisheye190").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_to_hequirect'), variable=self.convert_conv_var, value="to_hequirect").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_heq_to_flat'), variable=self.convert_conv_var, value="heq_to_flat").pack(anchor='w')
        ttk.Radiobutton(frame_conv, text=get_text('conv_fish_to_flat'), variable=self.convert_conv_var, value="fish_to_flat").pack(anchor='w')

        frame_bitrate = ttk.Frame(self.tab_convert)
        frame_bitrate.pack(fill='x', pady=5)
        ttk.Label(frame_bitrate, text=get_text('lbl_bitrate')).pack(side='left')
        self.convert_bitrate_var = tk.StringVar(value="100M")
        ttk.Entry(frame_bitrate, textvariable=self.convert_bitrate_var, width=10).pack(side='left', padx=5)

        ttk.Label(frame_bitrate, text="Output Codec:").pack(side='left', padx=(10, 0))
        self.convert_codec_var = tk.StringVar(value=DEFAULT_CODEC)
        ttk.OptionMenu(frame_bitrate, self.convert_codec_var, DEFAULT_CODEC, *CODEC_OPTIONS).pack(side='left', padx=5)

        frame_params = ttk.Frame(self.tab_convert)
        frame_params.pack(fill='x', pady=5)
        self.use_vulkan_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_params, text="Use Vulkan 360 filter (if available)", variable=self.use_vulkan_filter_var).pack(side='left', padx=(10, 0))
        ttk.Label(frame_params, text="FPS (Optional):").pack(side='left')
        self.convert_fps_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.convert_fps_var, width=8).pack(side='left', padx=5)

        ttk.Label(frame_params, text="Width (Optional):").pack(side='left')
        self.convert_width_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.convert_width_var, width=8).pack(side='left', padx=5)

        ttk.Label(frame_params, text="Height (Optional):").pack(side='left')
        self.convert_height_var = tk.StringVar()
        ttk.Entry(frame_params, textvariable=self.convert_height_var, width=8).pack(side='left', padx=5)

        frame_op = ttk.LabelFrame(self.tab_convert, text=get_text('grp_convert_op'), padding=10)
        frame_op.pack(fill='x', pady=5)
        self.convert_op_var = tk.StringVar(value="custom_tb_to_sbs")

        ops = [
            (get_text('op_custom_tb_sbs'), "custom_tb_to_sbs"),
            (get_text('op_sbs_sbs'), "sbs_to_sbs"),
            (get_text('op_tb_tb'), "tb_to_tb"),
            (get_text('op_tb_sbs'), "tb_to_sbs"),
            (get_text('op_sbs_tb'), "sbs_to_tb")
        ]
        for text, val in ops:
            ttk.Radiobutton(frame_op, text=text, variable=self.convert_op_var, value=val).pack(anchor='w')

        btn_frame = ttk.Frame(self.tab_convert)
        btn_frame.pack(pady=10)

        self.btn_convert_start = ttk.Button(btn_frame, text=get_text('btn_start'), command=self.run_convert)
        self.btn_convert_start.pack(side='left', padx=5)

        self.btn_convert_stop = ttk.Button(btn_frame, text=get_text('btn_stop'), command=self.stop_convert, state='disabled')
        self.btn_convert_stop.pack(side='left', padx=5)

        self.proc_convert = None

    def browse_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")])
        if f: var.set(f)

    def browse_dir(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)

    def save_file(self, var):
        f = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("MOV files", "*.mov"), ("All files", "*.*")])
        if f: var.set(f)

    def run_split(self):
        input_path = self.split_input_var.get()
        output_dir = self.split_output_var.get()
        mode = self.split_mode_var.get()

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_file'))
            return
        if not output_dir:
            output_dir = os.path.dirname(input_path)

        self.log(get_text('msg_starting_split').format(mode))

        self.btn_split_start.config(state='disabled')
        self.btn_split_stop.config(state='normal')

        def task():
            mask_val = self.split_mask_var.get()
            pack_scale_val = float(self.split_pack_scale_var.get()) if self.split_pack_scale_var.get() else 0.40
            padding_val = int(self.split_padding_var.get()) if self.split_padding_var.get() else 0
            split_video(
                input_path,
                mode,
                output_dir,
                conversion=self.split_conv_var.get(),
                log_callback=self.log,
                process_callback=lambda p: setattr(self, 'proc_split', p),
                bitrate=self.split_bitrate_var.get(),
                fps=self.split_fps_var.get(),
                width=self.split_width_var.get(),
                height=self.split_height_var.get(),
                out_codec=self.split_codec_var.get(),
                input_path2=mask_val if mask_val and os.path.exists(mask_val) else None,
                pack_scale=pack_scale_val,
                padding=padding_val,
                use_vulkan_filter=self.use_vulkan_filter_var.get()
            )
            self.log(get_text('msg_task_complete'))
            messagebox.showinfo(get_text('title_success'), get_text('msg_success_split'))
            self.root.after(0, lambda: self.btn_split_start.config(state='normal'))
            self.root.after(0, lambda: self.btn_split_stop.config(state='disabled'))
            self.proc_split = None

        threading.Thread(target=task).start()

    def stop_split(self):
        if self.proc_split:
            self.proc_split.kill()
            self.proc_split = None
            self.log(get_text('msg_stop'))

    def run_combine(self):
        in1 = self.combine_in1_var.get()
        in2 = self.combine_in2_var.get()
        mask_val = self.combine_mask_var.get()
        out = self.combine_out_var.get()
        mode = self.combine_mode_var.get()

        if not in1 or not os.path.exists(in1):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_1'))
            return
        if not in2 or not os.path.exists(in2):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_2'))
            return
        if not out:
             dirname = os.path.dirname(in1)
             filename = os.path.splitext(os.path.basename(in1))[0]
             for s in ["_L", "_R", "_T", "_B", "_l", "_r", "_t", "_b"]:
                 if filename.endswith(s):
                     filename = filename[:-len(s)]
                     break

             suffix = "_LR" if mode == "left_right" else "_TB"
             ext_out = get_ext_from_codec(self.combine_codec_var.get(), ".mp4")
             out = os.path.join(dirname, f"{filename}{suffix}{ext_out}")

        self.log(get_text('msg_starting_combine').format(mode))

        self.btn_combine_start.config(state='disabled')
        self.btn_combine_stop.config(state='normal')

        def task():
            combine_video(
                in1,
                in2,
                mode,
                out,
                conversion=self.combine_conv_var.get(),
                log_callback=self.log,
                process_callback=lambda p: setattr(self, 'proc_combine', p),
                bitrate=self.combine_bitrate_var.get(),
                mask_path=mask_val if mask_val and os.path.exists(mask_val) else None,
                fps=self.combine_fps_var.get(),
                width=self.combine_width_var.get(),
                height=self.combine_height_var.get(),
                out_codec=self.combine_codec_var.get(),
                use_vulkan_filter=self.use_vulkan_filter_var.get()
            )
            self.log(get_text('msg_task_complete'))
            messagebox.showinfo(get_text('title_success'), get_text('msg_success_combine'))
            self.root.after(0, lambda: self.btn_combine_start.config(state='normal'))
            self.root.after(0, lambda: self.btn_combine_stop.config(state='disabled'))
            self.proc_combine = None

        threading.Thread(target=task).start()

    def stop_combine(self):
        if self.proc_combine:
            self.proc_combine.kill()
            self.proc_combine = None
            self.log(get_text('msg_stop'))

    def run_convert(self):
        in_path = self.convert_in_var.get()
        out_path = self.convert_out_var.get()
        conversion = self.convert_conv_var.get()
        operation_mode = self.convert_op_var.get()
        mask_val = getattr(self, "convert_mask_var", tk.StringVar()).get() if hasattr(self, "convert_mask_var") else ""

        if not in_path or not os.path.exists(in_path):
            messagebox.showerror(get_text('title_error'), get_text('msg_error_input_file'))
            return

        self.log(f"Starting Conversion for: {in_path}")

        self.btn_convert_start.config(state='disabled')
        self.btn_convert_stop.config(state='normal')

        def task():
            if os.path.isdir(in_path):
                output_dir = out_path if out_path and os.path.isdir(out_path) else None
                batch_tb_to_sbs(
                    in_path,
                    output_dir=output_dir,
                    conversion=conversion,
                    log_callback=self.log,
                    process_callback=lambda p: setattr(self, 'proc_convert', p),
                    bitrate=self.convert_bitrate_var.get(),
                    operation_mode=operation_mode,
                    mask_path=mask_val if mask_val and os.path.exists(mask_val) else None,
                    fps=self.convert_fps_var.get(),
                    width=self.convert_width_var.get(),
                    height=self.convert_height_var.get(),
                    out_codec=self.convert_codec_var.get(),
                    use_vulkan_filter=self.use_vulkan_filter_var.get()
                )
            else:
                out_file = out_path
                if not out_file or os.path.isdir(out_file):
                     dirname = out_file if out_file else os.path.dirname(in_path)
                     filename = os.path.splitext(os.path.basename(in_path))[0]
                     suffix = ""
                     if operation_mode in ("sbs_to_tb", "tb_to_tb"):
                         suffix += "_tb"
                     else:
                         suffix += "_LR"

                     if conversion == "to_fisheye":
                         suffix = "_FE180" + suffix
                     elif conversion == "to_fisheye190":
                         suffix = "_FE190" + suffix
                     elif conversion == "to_hequirect":
                         suffix = "_180" + suffix
                     elif conversion == "heq_to_flat" or conversion == "fish_to_flat":
                         suffix = "_flat" + suffix

                     ext_out = get_ext_from_codec(self.convert_codec_var.get(), ".mp4")
                     out_file = os.path.join(dirname, f"{filename}{suffix}{ext_out}")

                tb_to_sbs(
                    in_path,
                    out_file,
                    conversion=conversion,
                    log_callback=self.log,
                    process_callback=lambda p: setattr(self, 'proc_convert', p),
                    bitrate=self.convert_bitrate_var.get(),
                    operation_mode=operation_mode,
                    mask_path=mask_val if mask_val and os.path.exists(mask_val) else None,
                    fps=self.convert_fps_var.get(),
                    width=self.convert_width_var.get(),
                    height=self.convert_height_var.get(),
                    out_codec=self.convert_codec_var.get(),
                    use_vulkan_filter=self.use_vulkan_filter_var.get()
                )
            self.log(get_text('msg_task_complete'))
            messagebox.showinfo(get_text('title_success'), "Conversion completed!")
            self.root.after(0, lambda: self.btn_convert_start.config(state='normal'))
            self.root.after(0, lambda: self.btn_convert_stop.config(state='disabled'))
            self.proc_convert = None

        threading.Thread(target=task).start()

    def stop_convert(self):
        if self.proc_convert:
            self.proc_convert.kill()
            self.proc_convert = None
            self.log(get_text('msg_stop'))

if __name__ == "__main__":
    root = tk.Tk()
    app = VRSplitCombineApp(root)
    root.mainloop()
