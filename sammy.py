import shutil, gc, os, torch, cv2, numpy as np, glob, matplotlib.pyplot as plt, torch.nn.functional as F, argparse
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox, plot_results
from torchvision.transforms import v2
from typing import List
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
SAM3_MAX = 1008
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
    return Sam3VideoPredictorMultiGPU(*model_args, checkpoint_path=checkpoint_path, gpus_to_use=gpus_to_use, is_sbs=is_sbs, max_num_objects= max_num_objects, num_obj_for_compile=num_obj_for_compile, strict_state_dict_loading=strict_state_dict_loading, **model_kwargs)

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
                num_obj_for_compile=1
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
                is_sbs = None,
                max_num_objects=1,
                num_obj_for_compile=1,
                use_fa3 = False

                )

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(output_size, output_size)),
                v2.ToDtype(torch.float32, scale=True),

            ]
        )

    def propagate_in_video(self, predictor=None, session_id=None, max_frame_num_to_track=None):

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
                        max_frame_num_to_track = max_frame_num_to_track if max_frame_num_to_track != -1 else None,
                        
                    )):
                
                outputs[response["frame_idx"]] = response["outputs"]

        return outputs

    def abs_to_rel_coords(self, coords=None, IMG_WIDTH=None, IMG_HEIGHT=None, coord_type="box"):

        if coord_type == "point":
            return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]

        elif coord_type == "box":

            return [

                [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
                for x, y, w, h in coords
            ]

        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

    def track(self, video_path = None, remove = False, add_box = True, sub_box = True, add_point = 0, show_plots = True):

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
                frames.sort(
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                )

            except ValueError:
                print(
                    f'frame names are not in "<frame_idx>.jpg" format: {frames[:5]=}, '
                    f"falling back to lexicographic sort."
                )
                frames.sort()

        image = Image.fromarray(load_frame(frames[0]))

        if image.height > self.output_size:

            full = image
            image = full.resize((self.output_size, self.output_size), Image.Resampling.BILINEAR)
            full.close()

        IMG_WIDTH, IMG_HEIGHT = image.size

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

            boxes = torch.tensor(self.abs_to_rel_coords(np.array([[252, 152, 704, 704]]), IMG_WIDTH, IMG_HEIGHT, coord_type="box"), dtype=torch.float32)
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
                bounding_box_labels = labels,
             
            )
        )

        out = response["outputs"]
        outputs = self.propagate_in_video(predictor, session_id)

        if add_point == 1:

            frame_idx = 0
            obj_id = 0
            points_abs = np.array(

                [
                    [350, 350], 
                ]
            )

            labels = np.array([1])
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
                    obj_id=obj_id,
                )
            )

            out = response["outputs"]
            outputs = self.propagate_in_video(predictor, session_id)

        if sub_box:

            box = np.array([[800, 552, 180, 280]])
            frame_idx = 0
            boxes = torch.tensor(self.abs_to_rel_coords(box, IMG_WIDTH, IMG_HEIGHT, coord_type="box"), dtype=torch.float32)
            labels = torch.tensor(np.array([0]), dtype=torch.int32)

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

        if image.height > output_size:

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

            masks = outputs.get("out_binary_masks", None)
            scores = outputs.get("out_probs", None)

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

            best_soft = np.clip((best_soft - 0.5) * 3.0 + 0.5, 0.0, 1.0)

            soft_masks.append(best_soft)
            valid_flags.append(min_valid_pixels <= np.count_nonzero(best_soft) <= output_size)

    filled_soft_masks, filled_count = _fill_soft_mask_gaps(soft_masks, valid_flags, max_interp_gap=2)

    if filled_count > 0:
        print(f"Filled {filled_count} missing/weak SAM3 masks using temporal soft-mask interpolation")

    for out_path, soft_mask in zip(output_paths, filled_soft_masks):

        hard_mask = (soft_mask).astype(np.uint8) * 255
        Image.fromarray(hard_mask, mode='L').save(out_path)

    if seq_dir.exists():
        shutil.rmtree(seq_dir)

    del tracker, inference_state
    gc.collect()
    torch.cuda.empty_cache()

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

def _sam3_inference(frames_dir, prompt, sam31, output_size, video_args, show_plots=False) -> None:

    folder = Path(frames_dir)

    image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
    image_files = [f for f in image_files if "_mask" not in f.stem]

    repo_path = snapshot_download(repo_id=SAM3_REPO_ID, local_files_only=False)
    model_path = os.path.join(repo_path, "sam3.pth")

    model = build_sam3_image_model(load_from_HF = False, enable_inst_interactivity = False, enable_segmentation = True, compile = False)
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])

    processor = Sam3Processor(model, confidence_threshold=0.4, device="cuda" if torch.cuda.is_available() else "cpu")

    # model = build_sam3_image_model(
    #     bpe_path=None,
    #     device = "cuda" if torch.cuda.is_available() else "cpu",
    #     eval_mode = True,
    #     checkpoint_path = None,
    #     load_from_HF= True,
    #     enable_segmentation = True,
    #     enable_inst_interactivity = False,
    #     compile = False,
    #     use_fa3 = False,
    # )

    if not image_files:
        return

    mask_paths: list[Path] = []
    soft_masks: list[np.ndarray] = []
    valid_flags: list[bool] = []

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
            valid_flags.append(min_valid_pixels <= np.count_nonzero(best_soft) <= output_size)

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

def seed_mask_batch(
        
    frames_dir: str,
    prompt: str = "one woman",
    sam31: bool=False,
    output_size: int | None = None,
    video_args: argparse.Namespace = None,
    seed_model: str = "sam3",
    sapiens_threshold: float = 0.5,
    gate_dilate: int = 5,

    ) -> None:

    seed_model = (seed_model or "sam3").lower()

    if seed_model == "sam3":

        _sam3_inference(frames_dir, prompt=prompt, sam31=sam31, output_size=output_size, video_args=video_args)

    elif seed_model == "sam3video":

        _sam3_video_inference(frames_dir, prompt=prompt, sam31=False, output_size=output_size, video_args=video_args)

    elif seed_model == "sam31video":

        _sam3_video_inference(frames_dir, prompt=prompt, sam31=True, output_size=output_size, video_args=video_args)

    elif seed_model == "sapiens":

        _sapiens_inference(frames_dir, prompt=prompt, sam31=sam31, output_size=output_size, video_args=video_args)

    elif seed_model == "hybrid":

        _sam_sapiens(

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

def _sam_sapiens(
        
        frames_dir: str, 
        prompt: str = "one woman", 
        sam31: bool = False,
        output_size: int | None = None, 
        video_args: argparse.Namespace = None, 
        threshold: float = 0.5, 
        gate_dilate: int = 5

        ) -> None:

    _sam3_inference(frames_dir, prompt=prompt, sam31=sam31, output_size=output_size, video_args=video_args)

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
                sam3_mask = cv2.resize(sam3_mask, (sapiens_mask.shape[1], sapiens_mask.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)

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

def _sapiens_inference(frames_dir: str, prompt: str = "one woman", sam31:bool = False, output_size: int | None = None, video_args: argparse.Namespace = None, threshold: float = 0.5, gate_dilate: int = 5) -> None:

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
        mode="bilinear",
        align_corners=False,
    )

    outputs = outputs.squeeze(0).float().cpu().numpy()

    if outputs.size == 3:
        alpha = outputs[2].clip(0.0, 1.0)

    else:
        alpha = outputs[3].clip(0.0, 1.0)

    return alpha

def sam3_masks(

    mask_segments: List[SegmentInfo],
    frames_dir: Path,
    masks_dir: Path,
    mask_square: int,
    prompt: str | None,
    seed_model: str,
    sapiens_threshold: float,
    gate_dilate: int,
    video_args: argparse.Namespace,
    sam31: bool = False,

    ) -> List[SegmentInfo]:

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
