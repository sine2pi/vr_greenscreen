import shutil, gc, os, torch, cv2, numpy as np, glob, matplotlib.pyplot as plt, torch.nn.functional as F, argparse, imageio
from pathlib import Path
from PIL import Image, ImageFilter
from huggingface_hub import snapshot_download
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox, plot_results
from torchvision.transforms import v2
from typing import List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
_setup_tf32()

SAM3_REPO_ID = "sin2piusc/sam3_fta"

SAM3_BOX_CXCYWH_NORM = (0.5, 0.4, 0.5, 0.5)
SAM3_BOX2_CXCYWH_NORM = (0.5, 0.9, 0.9, 0.18)
# SAM3_BOX3_CXCYWH_NORM = (0.1, 0.9, 0.9, 0.18)

SAPIENS_REPO_ID = "facebook/sapiens2-matting-1b"
SAPIENS_CHECKPOINT = "sapiens2_1b_matting.safetensors"
SAPIENS_CONFIG = "assets/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py"

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

def build_sam3_video_predictor(*model_args,
                               checkpoint_path=None,
                               gpus_to_use=None,
                               is_sbs=False,
                               max_num_objects=1,
                               num_obj_for_compile=1,
                               strict_state_dict_loading=False,
                               **model_kwargs):

    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

    return Sam3VideoPredictorMultiGPU(*model_args, 
    checkpoint_path=checkpoint_path, 
    gpus_to_use=gpus_to_use, 
    is_sbs=is_sbs, 
    max_num_objects=max_num_objects, 
    num_obj_for_compile=num_obj_for_compile, 
    strict_state_dict_loading=strict_state_dict_loading, 
    **model_kwargs)

class sam3_video_inference:

    def __init__(self, video_path, prompt, sam31, output_size, video_args):

        self.video_path = video_path
        self.prompt = prompt
        self.seg_length = video_args.segment_length
        self.output_size = video_args.mask_height
        self.sam31 = sam31

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
                async_loading_frames  = True,
                num_obj_for_compile=1,

                )

        else:

            self.predictor = build_sam3_video_predictor(

                bpe_path = None,
                gpus_to_use = None,
                has_presence_token = False,
                geo_encoder_use_img_cross_attn = False,
                strict_state_dict_loading = False,
                async_loading_frames = True,
                video_loader_type = "cv2",
                apply_temporal_disambiguation = True,
                compile = False,
                is_sbs = False,
                max_num_objects=1,
                num_obj_for_compile=1,
                use_fa3 = False

                )

    def propagate_in_video(self, predictor=None, session_id=None, max_frame_num_to_track=None):

        print()
        print(f"Sam3 inference. ... ♩ ♪ ♫ ♬")

        max_frame_num_to_track=int(self.seg_length * 60) # assumes 60 fps
        predictor=self.predictor
        outputs = {}

        if self.sam31:
            for response in predictor.handle_stream_request(

                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction="forward",
                        output_prob_thresh = 0.4,

                    )):

                outputs[response["frame_idx"]] = response["outputs"]
        else:
            for response in predictor.handle_stream_request(

                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction="forward",
                        output_prob_thresh = 0.4,
                        max_frame_num_to_track = None, #max_frame_num_to_track if max_frame_num_to_track != -1 else None, ## will results in blank masks if not correctly set. None is safest since it allows tracking all frames by default.

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

    def track(self, video_path = None, remove = False, add_box = False, sub_box = False, add_point = 0, show_plots = False):

        predictor, video_path, prompt = self.predictor, self.video_path, self.prompt

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

        IMG_WIDTH, IMG_HEIGHT = image.size

        response = predictor.handle_request(

            request=dict(
                type="start_session",
                resource_path=video_path))

        session_id = response["session_id"]

        _ = predictor.handle_request(

            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        frame_idx = 0
        response = predictor.handle_request(

            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_idx=frame_idx,
                text=prompt,
            )
        )

        out = response["outputs"]
        outputs = self.propagate_in_video(predictor, session_id)

        if add_box:

            boxes = torch.tensor(np.array([[0.1, 0.1, 0.8, 0.7]]), dtype=torch.float32)
            labels = torch.tensor(np.array([1]), dtype=torch.int32)

        else:
            boxes, labels = None, None

        frame_idx = 0
        response = predictor.handle_request(

            request=dict(

                type="add_prompt",
                session_id=session_id,
                frame_idx=frame_idx,
                text=prompt,
                bounding_boxes = boxes,
                bounding_box_labels = labels
                
                ))

        out = response["outputs"]
        outputs = self.propagate_in_video(predictor, session_id)

        if add_point == 1:

            frame_idx = 0
            obj_id = 0

            points_tensor = torch.tensor(self.abs_to_rel_coords(np.array([[350, 350]]), IMG_WIDTH, IMG_HEIGHT, coord_type="point"), dtype=torch.float32)
            points_labels_tensor = torch.tensor(np.array([1]), dtype=torch.int32)

            response = predictor.handle_request(

                request=dict(

                    type="add_prompt",
                    session_id=session_id,
                    frame_idx=frame_idx,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    obj_id=obj_id

                    ))

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

        if add_point == 4:

            frame_idx = 0
            obj_id = 0
            points_abs = np.array(

                [
                    [740, 450],  # +
                    [760, 630],  # -
                    [840, 640],  # -
                    [760, 550],  # +
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
                    obj_id=obj_id

                    ))

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

        if sub_box:

            box = [[0.1, 0.8, 0.8, 0.2]]
            boxes = torch.tensor(np.array(box), dtype=torch.float32)
            labels = torch.tensor(np.array([0]), dtype=torch.int32)

            frame_idx = 0
            response = predictor.handle_request(

                request=dict(
                    type = "add_prompt",
                    session_id = session_id,
                    frame_idx = frame_idx,
                    text = prompt if prompt else None,
                    bounding_boxes = boxes,
                    bounding_box_labels = labels,

                )
            )

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

        if remove:

            obj_id = 0
            response = predictor.handle_request(

                request=dict(
                    type="remove_object",
                    session_id=session_id,
                    obj_id=obj_id,
                )
            )

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

        if show_plots:

            plt.close("all")
            visualize_formatted_frame_output(
                frame_idx,
                frames,
                outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
                titles=["SAM 3 Dense Tracking outputs"],
                figsize=(6, 4),
            )

        _ = predictor.handle_request(

            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        predictor.shutdown()
        return outputs

def _sam3_video_inference(frames_dir, prompt, sam31, output_size, video_args) -> None:

    output_size = video_args.mask_height
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

    soft_masks: list[np.ndarray] = []
    valid_flags: list[bool] = []

    min_valid_pixels = int(output_size * 0.8)

    for i, frame_path in enumerate(image_files):

        out_path = frame_path.parent / f"{frame_path.stem}_mask.png"
        output_paths.append(out_path)

        raw = Image.open(frame_path)
        image = raw.convert("RGB")
        raw.close()

        if image.height != output_size:

            full = image
            image = full.resize((output_size, output_size), Image.Resampling.BILINEAR)
            full.close()

        frame_shapes.append((image.height, image.width))
        image.save(seq_dir / f"{i:06d}.jpg", format="JPEG", quality=100)
        image.close()

    tracker = sam3_video_inference(video_path=str(seq_dir), prompt=prompt, sam31=sam31, output_size=output_size, video_args=video_args)
    inference_state = tracker.track()

    with torch.inference_mode():

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
                best_soft = np.zeros((image.height, image.width), dtype=np.float32)

                print(f"No SAM3 masks/scores for {frame_path.name}; marking as missing for temporal fill")

            else:
                best_idx = int(np.argmax(scores))
                best_soft = masks[best_idx]

                if len(best_soft.shape) == 3:
                    best_soft = best_soft[0]

                best_soft = np.asarray(best_soft, dtype=np.float32)

                print("Confidence:", scores[best_idx])

            best_soft = np.clip(best_soft, 0.0, 1.0)
            soft_masks.append(best_soft)
            valid_flags.append(np.count_nonzero(best_soft >= 0.5) >= min_valid_pixels)

    filled_masks, filled_count = fill_soft(soft_masks, valid_flags, max_interp_gap=6)

    if filled_count > 0:
        print(f"Filled {filled_count} missing/weak SAM3 masks using temporal soft-mask interpolation")

    for out_path, soft_mask in zip(output_paths, filled_masks):

        mask = (soft_mask > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask, mode='L').save(out_path)

    if seq_dir.exists():
        shutil.rmtree(seq_dir)

    del tracker, inference_state
    gc.collect()
    torch.cuda.empty_cache()

def _fill_soft(

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

def fill_soft(*args, **kwargs):
    return _fill_soft(*args, **kwargs)

def _sam3_inference(frames_dir, prompt, sam31, output_size, video_args, show_plots=False) -> None:

    folder = Path(frames_dir)

    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    repo_path = snapshot_download(repo_id=SAM3_REPO_ID, local_files_only=False)
    model_path = os.path.join(repo_path, "sam3.pth")

    model = build_sam3_image_model(load_from_HF = True, enable_inst_interactivity = False, enable_segmentation = True, compile = False)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'), strict=False)

    processor = Sam3Processor(model, confidence_threshold=0.2, device="cuda" if torch.cuda.is_available() else "cpu")

    if not image_files:
        return

    mask_paths = []
    soft_masks = []
    valid_flags = []

    min_valid_pixels = int(output_size * 0.8)

    with torch.inference_mode():

        for frame_path in image_files:

            output_path = frame_path.parent / f"{frame_path.stem}_mask.png"
            mask_paths.append(output_path)
            raw = Image.open(frame_path)
            image = raw.convert("RGB")
            raw.close()

            width, height = image.width, image.height

            inference_state = processor.set_text_prompt(state=processor.set_image(image), prompt=prompt)
            box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor([sam3_box(width, height), sam3_box2(width, height)]).view(-1, 4)).view(-1,4)

            for box, label in zip(normalize_bbox(box_input_cxcywh, width, height).tolist(), [True, False]):
                inference_state = processor.add_geometric_prompt(state=inference_state, box=box, label=label)

            if show_plots:
                plot_results(image, inference_state)
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

                print("Confidence:", scores[best_idx])

            soft_masks.append(best_soft)
            valid_flags.append(np.count_nonzero(best_soft >= 0.5) >= min_valid_pixels)

            del inference_state
            image.close()

    filled_masks, filled_count = _fill_soft(soft_masks, valid_flags, max_interp_gap=6)

    if filled_count > 0:
        print(f"Filled {filled_count} missing/weak SAM3 masks using temporal soft-mask interpolation")

    for out_path, soft_mask in zip(mask_paths, filled_masks):

        hard_mask = (soft_mask >= 0.5).astype(np.uint8) * 255
        Image.fromarray(hard_mask, mode='L').save(out_path)

    del processor, model

    gc.collect()
    torch.cuda.empty_cache()

def sam3_inference(frames_dir, prompt, sam31, output_size, video_args, show_plots=False) -> None:
    return _sam3_inference(frames_dir, prompt, sam31, output_size, video_args, show_plots=show_plots)

def seed_mask_batch(

    frames_dir: str,
    prompt: str,
    sam31: bool,
    output_size: int | None,
    video_args: argparse.Namespace,
    seed_model: str,
    sapiens_threshold: float,
    gate_dilate: int,

    ) -> None:

    seed_model = (seed_model or "sam3").lower()

    if seed_model == "sam3":

        sam3_inference(
            frames_dir, 
            prompt=prompt, 
            sam31=sam31, 
            output_size=output_size, 
            video_args=video_args
            )

    elif seed_model == "sam3video":

        _sam3_video_inference(
            frames_dir,
            prompt=prompt,
            sam31=False,
            output_size=output_size,
            video_args=video_args,
            )

    elif seed_model == "sam31video":

        _sam3_video_inference(
            frames_dir,
            prompt=prompt,
            sam31=True,
            output_size=output_size,
            video_args=video_args,
            )

    elif seed_model == "sapiens":

        sapiens_inference(
            frames_dir, 
            prompt=prompt, 
            sam31=sam31, 
            output_size=output_size, 
            video_args=video_args
            )

    elif seed_model == "hybrid":

        sam_sapiens(
            frames_dir,
            prompt=prompt,
            sam31=sam31,
            output_size=output_size,
            video_args=video_args,
            threshold=sapiens_threshold,
            gate_dilate=gate_dilate,
            )

    else:
        raise ValueError(f"Unsupported seed model: {seed_model}")

def sam_sapiens(

        frames_dir: str,
        prompt: str,
        sam31: bool,
        output_size: int | None,
        video_args: argparse.Namespace,
        threshold: float,
        gate_dilate: int,

        ) -> None:

    _sam3_video_inference(frames_dir, prompt=prompt, sam31=False, output_size=output_size, video_args=video_args)

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

            image_rgb = np.array(image)
            image_bgr = image_rgb[:, :, ::-1]
            alpha = estimate_alpha(image_bgr, model)

            sapiens_mask = (alpha >= threshold).astype(np.uint8) * 255
            sam3_mask = np.array(Image.open(sam3_mask_path).convert('L'))

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

def sapiens_inference(frames_dir, prompt, sam31, output_size, video_args, threshold: float = 0.5, gate_dilate: int = 5):

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

            image_rgb = np.array(image)
            image_bgr = image_rgb[:, :, ::-1]

            alpha = estimate_alpha(image_bgr, model)
            mask = (alpha >= threshold).astype(np.uint8) * 255

            Image.fromarray(mask, mode='L').save(output_path)
            image.close()

            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

def estimate_alpha(image_bgr, model):

    h0, w0 = image_bgr.shape[:2]
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs = data["inputs"]

    with torch.no_grad():
        outputs = model(inputs)

    outputs = F.interpolate(
        outputs,
        size=(h0, w0),
        mode="bilinear",
        align_corners=False,
    )

    outputs = outputs.squeeze(0).float().cpu().numpy()

    if outputs.size == 3:
        alpha = outputs[2].clip(0.0, 1.0)

    else:
        alpha = outputs[3].clip(0.0, 1.0)

    return alpha

def _sam3_masks(
    mask_segments,
    frames_dir: Path,
    masks_dir: Path,
    mask_square: int,
    prompt: str | None,
    seed_model: str,
    sapiens_threshold: float,
    gate_dilate: int,
    video_args: argparse.Namespace,
    sam31: bool = False,
    ):

    print(f"Seed model: {seed_model}")

    seed_mask_batch(

        str(frames_dir),
        output_size=mask_square,
        prompt=prompt,
        sam31=sam31,
        seed_model=seed_model,
        sapiens_threshold=sapiens_threshold,
        gate_dilate=gate_dilate,
        video_args=video_args,

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

def sam3_masks(*args, **kwargs):
    return _sam3_masks(*args, **kwargs)

def read_frames(video_path: str):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for SAM3 tracking: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    frames_rgb = []

    while True:
        ok, frame_bgr = cap.read()

        if not ok:
            break

        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames_rgb:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames_rgb, fps

def tracker_ouputs(outputs, out_h: int, out_w: int) -> np.ndarray:

    if not outputs:
        return np.zeros((out_h, out_w), dtype=np.float32)

    masks = outputs.get("out_binary_masks", None)

    if masks is None:
        masks = outputs.get("out_masks", None)

    scores = outputs.get("out_probs", None)

    if masks is None:
        return np.zeros((out_h, out_w), dtype=np.float32)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().float().cpu().numpy()

    else:
        masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.shape[0] == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()

    elif scores is None:
        scores = None

    else:
        scores = np.asarray(scores)

    if scores is None or np.size(scores) == 0:
        best_idx = 0

    else:
        best_idx = int(np.argmax(scores))

    if best_idx < 0 or best_idx >= masks.shape[0]:
        best_idx = 0

    best_soft = np.asarray(masks[best_idx], dtype=np.float32)

    if best_soft.ndim == 3:
        best_soft = best_soft[0]

    if best_soft.shape != (out_h, out_w):
        best_soft = cv2.resize(best_soft, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if best_soft.min() < 0.0:
        x = np.clip(best_soft, -20.0, 20.0)
        best_soft = 1.0 / (1.0 + np.exp(-x))

    elif best_soft.max() > 1.0:
        best_soft = best_soft / 255.0

    return np.clip(best_soft, 0.0, 1.0).astype(np.float32)

def sam3_track(

    video_path: str,
    prompt: str,
    sam31: bool,
    output_size: int,
    video_args: argparse.Namespace,
    frame_count: int,
    out_h: int,
    out_w: int,

) -> list[np.ndarray]:

    tracker = sam3_video_inference(
        video_path=video_path,
        prompt=prompt,
        sam31=sam31,
        output_size=output_size,
        video_args=video_args,
    )

    inference_state = tracker.track()

    soft_masks = []
    valid_flags = []

    min_valid_pixels = max(32, int(min(out_h, out_w) * 0.8))

    for frame_idx in range(frame_count):

        outputs = inference_state.get(frame_idx, None)
        best_soft = tracker_ouputs(outputs, out_h, out_w)

        soft_masks.append(best_soft)
        valid_flags.append(np.count_nonzero(best_soft >= 0.5) >= min_valid_pixels)

    filled_masks, _ = fill_soft(soft_masks, valid_flags, max_interp_gap=6)

    del tracker, inference_state
    gc.collect()
    torch.cuda.empty_cache()

    return filled_masks

def refinement(

    sam_soft,
    fg_thr: float,
    bg_thr: float,
    unknown_dilate_px: int,

) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    p = np.clip(np.asarray(sam_soft, dtype=np.float32), 0.0, 1.0)
    has_soft_band = bool(np.any((p > bg_thr) & (p < fg_thr)))

    if has_soft_band:
        sure_fg = p >= float(fg_thr)
        sure_bg = p <= float(bg_thr)
        unknown = ~(sure_fg | sure_bg)

    else:
        fg = p > 0.5
        radius = max(1, int(unknown_dilate_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

        sure_fg = cv2.erode((fg.astype(np.uint8) * 255), kernel, iterations=1) > 0
        sure_bg = cv2.dilate((fg.astype(np.uint8) * 255), kernel, iterations=1) == 0
        unknown = ~(sure_fg | sure_bg)

    if int(unknown_dilate_px) > 0:
        radius = int(unknown_dilate_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        unknown = cv2.dilate((unknown.astype(np.uint8) * 255), kernel, iterations=1) > 0
        unknown = unknown & ~sure_fg
        sure_bg = sure_bg & ~unknown

    return sure_fg, sure_bg, unknown

def refine_edges(

    frame_rgb,
    sam_soft,
    sapiens_model,
    fg_thr: float,
    bg_thr: float,
    unknown_dilate_px: int,
    gate_dilate_px: int,

) -> np.ndarray:

    frame_bgr = frame_rgb[:, :, ::-1]
    alpha = estimate_alpha(frame_bgr, sapiens_model)

    sure_fg, sure_bg, unknown = refinement(sam_soft, fg_thr, bg_thr, unknown_dilate_px)

    refined = np.zeros_like(alpha, dtype=np.float32)
    refined[sure_fg] = 1.0
    refined[unknown] = alpha[unknown]
    refined[sure_bg] = 0.0

    if int(gate_dilate_px) > 0:
        radius = int(gate_dilate_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        gate = cv2.dilate(((sam_soft > bg_thr).astype(np.uint8) * 255), kernel, iterations=1) > 0
        refined = np.where(gate, refined, 0.0)

    return np.clip(refined, 0.0, 1.0).astype(np.float32)

def masks_video(output_file: str, soft_masks: list[np.ndarray], fps: float) -> str:

    if not soft_masks:
        raise RuntimeError("No masks available to write video")

    alpha = np.stack(soft_masks, axis=0)

    if alpha.ndim == 3:
        alpha = alpha[..., None]

    alpha_u8 = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    imageio.mimwrite(output_file, alpha_u8, fps=float(fps), quality=10)

    return output_file

def load_sapiens():

    from huggingface_hub import hf_hub_download
    from sapiens.dense.src.models.core.matting_estimator import MattingEstimator
    from sapiens.dense.src.models.init_model import init_model

    _ = MattingEstimator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = hf_hub_download(repo_id=SAPIENS_REPO_ID, filename=SAPIENS_CHECKPOINT)

    return init_model(SAPIENS_CONFIG, ckpt, device=device)

def sam3_process(job: dict, sapiens_model, video_args) -> str:

    input_path = job['input_path']
    output_path = job['output_path']
    prompt = str(job.get('prompt', 'one woman'))
    output_size = int(job.get('mask_height', video_args.mask_height))
    sam31 = bool(job.get('sam31', False))
    refine_with_sapiens = bool(job.get('refine_with_sapiens', False))
    fg_thr = float(job.get('refine_fg_threshold', 0.85))
    bg_thr = float(job.get('refine_bg_threshold', 0.05))
    unknown_dilate_px = int(job.get('refine_unknown_dilate', 5))
    gate_dilate_px = int(job.get('gate_dilate', 5))

    frames_rgb, fps = read_frames(input_path)
    out_h, out_w = frames_rgb[0].shape[:2]

    soft_masks = sam3_track(
        input_path,
        prompt=prompt,
        sam31=sam31,
        output_size=output_size,
        video_args=video_args,
        frame_count=len(frames_rgb),
        out_h=out_h,
        out_w=out_w,
    )

    if refine_with_sapiens:

        if sapiens_model is None:
            raise RuntimeError("Sapiens model is required for SAM3+Sapiens refinement")

        refined_masks = []

        for frame_rgb, sam_soft in zip(frames_rgb, soft_masks):
            refined_masks.append(

                refine_edges(
                    frame_rgb,
                    sam_soft,
                    sapiens_model,
                    fg_thr=fg_thr,
                    bg_thr=bg_thr,
                    unknown_dilate_px=unknown_dilate_px,
                    gate_dilate_px=gate_dilate_px,
                )
            )

        final_masks = refined_masks

    else:
        final_masks = soft_masks

    os.makedirs(output_path, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_path, f'{video_name}_pha.mp4')

    masks_video(output_file, final_masks, fps)

    del frames_rgb, soft_masks, final_masks
    gc.collect()
    torch.cuda.empty_cache()

    return output_file

def sam3_track_inference(
    jobs,
    on_segment_done,
    video_args
):

    if not jobs:
        return []

    needs_sapiens = any(bool(job.get('refine_with_sapiens', False)) for job in jobs)
    sapiens_model = None

    if needs_sapiens:
        print("Loading Sapiens edge-refinement model...")
        sapiens_model = load_sapiens()

    completed= []

    try:

        total_ops = len(jobs)

        for i, job in enumerate(jobs, 1):

            label = str(job.get('label', os.path.basename(str(job.get('input_path', 'job')))))
            op_num = int(job.get('op_num', i))
            total = int(job.get('total_ops', total_ops))

            print(f"[{op_num}/{total}] {label}")
            output_file = sam3_process(job, sapiens_model=sapiens_model, video_args=video_args)
            completed.append(output_file)

            if on_segment_done:
                on_segment_done(output_file)

    finally:

        if sapiens_model is not None:
            del sapiens_model

        gc.collect()
        torch.cuda.empty_cache()

    return completed
