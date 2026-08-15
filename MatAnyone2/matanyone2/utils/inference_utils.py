import os
import cv2
import random
import numpy as np

import imageio.v3 as iio
import torch

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

def _normalize_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.shape[-1] > 3:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame)

def get_video_stream_info(frame_root):
    if not frame_root.endswith(VIDEO_EXTENSIONS):
        raise RuntimeError(
            f"Frame-folder input requires explicit FPS and is unsupported in strict mode: {frame_root}"
        )

    video_name = os.path.splitext(os.path.basename(frame_root))[0]
    meta = iio.immeta(frame_root, plugin="pyav")

    fps_raw = meta.get('fps', None)
    if fps_raw is None:
        raise RuntimeError(f"Missing FPS metadata for video: {frame_root}")

    fps = float(fps_raw)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS metadata for video {frame_root}: {fps_raw}")

    try:
        props = iio.improps(frame_root, plugin="pyav")
        length = props.n_images
        if length < 0:
            length = None
    except Exception:
        length = None

    return fps, length, video_name

def iter_video_frames(frame_root):
    if not frame_root.endswith(VIDEO_EXTENSIONS):
        raise RuntimeError(
            f"Frame-folder input requires explicit FPS and is unsupported in strict mode: {frame_root}"
        )

    reader = iio.imiter(frame_root, plugin="pyav")
    try:
        for frame in reader:
            yield _normalize_video_frame(frame)
    finally:
        reader.close()

def read_frame_from_videos(frame_root):
    fps, _, video_name = get_video_stream_info(frame_root)
    frames_list = [frame.copy() for frame in iter_video_frames(frame_root)]
    if not frames_list:
        raise RuntimeError(f"No frames could be read from video: {frame_root}")

    frames = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).contiguous()  # TCHW
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
