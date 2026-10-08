# import os
# import cv2
# import random
# import numpy as np
# import av

# import imageio.v2 as imageio
# import torch

# IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
# VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

# def read_frame_from_videos(frame_root):
#     if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path

#         video_name = os.path.basename(frame_root)[:-4]
#         container = av.open(frame_root)
#         stream = container.streams.video[0]
#         fps = float(stream.average_rate)
#         frames_list = []
#         for frame in container.decode(stream):
#             arr = frame.to_ndarray(format='rgb24')  # HWC uint8
#             frames_list.append(arr)
#         container.close()
#         frames = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).contiguous()  # TCHW uint8'''

#         video_name = os.path.splitext(os.path.basename(frame_root))[0]

#         reader = imageio.get_reader(frame_root)
#         try:
#             meta = reader.get_meta_data()
#             fps_raw = (meta or {}).get('fps', None)
#             if fps_raw is None:
#                 raise RuntimeError(f"Missing FPS metadata for video: {frame_root}")
#             fps = float(fps_raw)
#             if fps <= 0:
#                 raise RuntimeError(f"Invalid FPS metadata for video {frame_root}: {fps_raw}")
#             frames_list = []
#             for frame in reader:
#                 frame = np.asarray(frame)
#                 if frame.ndim == 2:
#                     frame = np.repeat(frame[..., None], 3, axis=2)
#                 elif frame.shape[-1] > 3:
#                     frame = frame[..., :3]
#                 frames_list.append(frame.copy())
#         finally:
#             reader.close()

#         if not frames_list:
#             raise RuntimeError(f"No frames could be read from video: {frame_root}")

#         frames = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).contiguous()  # TCHW
#     else:
#         raise RuntimeError(
#             f"Frame-folder input requires explicit FPS and is unsupported in strict mode: {frame_root}"
#         )

#     length = frames.shape[0]

#     return frames, fps, length, video_name

# def get_video_paths(input_root):
#     video_paths = []
#     for root, _, files in os.walk(input_root):
#         for file in files:
#             if file.lower().endswith(VIDEO_EXTENSIONS):
#                 video_paths.append(os.path.join(root, file))
#     return sorted(video_paths)

# def str_to_list(value):
#     return list(map(int, value.split(',')))

# def gen_dilate(alpha, min_kernel_size, max_kernel_size):
#     kernel_size = random.randint(min_kernel_size, max_kernel_size)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
#     fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
#     dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
#     return dilate.astype(np.float32)

# def gen_erosion(alpha, min_kernel_size, max_kernel_size):
#     kernel_size = random.randint(min_kernel_size, max_kernel_size)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
#     fg = np.array(np.equal(alpha, 255).astype(np.float32))
#     erode = cv2.erode(fg, kernel, iterations=1)*255
#     return erode.astype(np.float32)

import os
import cv2
import random
import numpy as np

import torch
import av

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI')

def read_frame_from_videos(frame_root):
    if frame_root.endswith(VIDEO_EXTENSIONS):  # Video file path

        video_name = os.path.basename(frame_root)[:-4]
        container = av.open(frame_root)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frames_list = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format='rgb24')  # HWC uint8
            frames_list.append(arr)
        container.close()
        frames = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2).contiguous()  # TCHW uint8'''

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

# def gen_dilate(alpha, min_kernel_size, max_kernel_size): 
#     kernel_size = random.randint(min_kernel_size, max_kernel_size)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
#     fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
#     dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
#     return dilate.astype(np.float32)

# def gen_erosion(alpha, min_kernel_size, max_kernel_size): 
#     kernel_size = random.randint(min_kernel_size, max_kernel_size)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
#     fg = np.array(np.equal(alpha, 255).astype(np.float32))
#     erode = cv2.erode(fg, kernel, iterations=1)*255
#     return erode.astype(np.float32)

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
        padding=kernel_size // 2,
    )
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
        value=0,
    )
    eroded = F.conv2d(padded_foreground, kernel.unsqueeze(0).unsqueeze(0))
    return (eroded[0, 0, :alpha.shape[-2], :alpha.shape[-1]] == kernel.sum()).to(alpha.dtype) * 255