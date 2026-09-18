import datetime
import logging
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sam3.logger import get_logger
from sam3.model.box_ops import fast_diag_box_iou
from sam3.model.data_misc import BatchedDatapoint, NestedTensor
from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores, mask_to_box
from sam3.model.sam3_video_base import (
    _associate_det_trk_compilable,
    LazyAssociateDetTrkResult,
    MaskletConfirmationStatus,
    realize_adt_result,
    RealizedAssociateDetTrkresult,
    Sam3VideoBase,
)
from sam3.perflib.masks_ops import mask_iou
from sam3.train.masks_ops import rle_encode
from torch import nn, Tensor

SAM3_COLLECTIVE_OP_TIMEOUT_SEC = int(os.getenv("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180"))

logger = get_logger(__name__)

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class Sam3MultiplexTrackerPredictor(nn.Module):
    def __init__(
        self,
        config_file,
        checkpoint_file=None,
        hydra_overrides=None,
        per_obj_inference=False,
        fill_hole_area=0,
        use_fa3=False,
        use_rope_real=False,
        keep_first_cond_frame=False,
        is_multiplex=False,
        is_multiplex_dynamic=False,
        use_memory_selection=False,
        max_num_objects=2,
        num_obj_for_compile=2,
    ):
        """
        Initialize the SAM2 predictor with the given configuration and checkpoint.
        Args:
            config_file (str): Path to the configuration file.
            checkpoint_file (str, optional): Path to the checkpoint file. If None, the model will be initialized without loading weights.
            hydra_overrides (list, optional): List of Hydra overrides to apply to the configuration.
            per_obj_inference (bool): If True, the model will perform per-object inference instead of bucketized batching.
        """

        super().__init__()
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        if not GlobalHydra().is_initialized():
            logger.info("Sam3MultiplexTrackerPredictor: GlobalHydra not initialized")
            GlobalHydra.instance().clear()
            initialize_config_module("sam3.config", version_base="1.2")

        if hydra_overrides is None:
            hydra_overrides = []
        self.is_multiplex = is_multiplex
        self.is_multiplex_dynamic = is_multiplex_dynamic
        self.per_obj_inference = per_obj_inference

        if self.is_multiplex:
            inference_model_class = "sam3.model.video_tracking_multiplex_demo.Sam3VideoTrackingMultiplexDemo"
        else:
            inference_model_class = (
                "sam3.model.video_tracking_with_prompt_demo_per_obj_inference.Sam3VideoTrackingWithPromptDemoPerObjInference"
                if per_obj_inference
                else "sam3.model.video_tracking_with_prompt_demo.Sam3VideoTrackingWithPromptDemo"
            )
        hydra_overrides = list(hydra_overrides)
        hydra_overrides.extend(
            [
                "launcher.experiment_log_dir=''",
                f"++trainer.model._target_={inference_model_class}",
                "++trainer.model.image_size=1008",
                "++trainer.model.backbone_stride=14",
                "++trainer.model.maskmem_backbone.mask_downsampler.interpol_size=[1152,1152]",
                "++trainer.model.backbone.forward_in_chunk_for_eval=false",
                "++trainer.model.always_start_from_first_ann_frame=false",
                "++trainer.model.non_overlap_masks_for_mem_enc=false",
                "++trainer.model.non_overlap_masks_for_output=false",
                "++trainer.model.max_cond_frames_in_attn=4",
                f"++trainer.model.keep_first_cond_frame={keep_first_cond_frame}",
                "++trainer.model.offload_output_to_cpu_for_eval=false",
                "++trainer.model.trim_past_non_cond_mem_for_eval=false",
                "++trainer.model.backbone.visual.trunk.weights_path=null",
                "++trainer.model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++trainer.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++trainer.model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++trainer.model.binarize_mask_from_pts_for_mem_enc=true",
                "++trainer.model.only_obj_ptrs_in_the_past_for_eval=true",
                "++trainer.model.clear_non_cond_mem_around_input=true",
                "++trainer.model.transformer.encoder.layer.self_attention.feat_sizes=[72,72]",
                "++trainer.model.transformer.encoder.layer.cross_attention.feat_sizes=[72,72]",
                f"++trainer.model.fill_hole_area={fill_hole_area}",
                f"++trainer.model.transformer.encoder.layer.self_attention.use_fa3={use_fa3}",
                f"++trainer.model.transformer.encoder.layer.cross_attention.use_fa3={use_fa3}",
                f"++trainer.model.transformer.encoder.layer.self_attention.use_rope_real={use_rope_real}",
                f"++trainer.model.transformer.encoder.layer.cross_attention.use_rope_real={use_rope_real}",
            ]
        )

        if self.is_multiplex or self.is_multiplex_dynamic:
            hydra_overrides.extend(
                [
                    f"++trainer.model.transformer.encoder.layer.self_attention_rope.use_fa3={use_fa3}",
                    f"++trainer.model.transformer.encoder.layer.cross_attention_rope.use_fa3={use_fa3}",
                    f"++trainer.model.transformer.encoder.layer.self_attention_rope.use_rope_real={use_rope_real}",
                    f"++trainer.model.transformer.encoder.layer.cross_attention_rope.use_rope_real={use_rope_real}",
                ]
            )

        hydra_overrides.extend(
            [f"++trainer.model.use_memory_selection={use_memory_selection}"]
        )

        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        model = instantiate(cfg.trainer.model, _recursive_=True)
        del model.backbone
        if checkpoint_file is not None:
            ckpt = torch.load(checkpoint_file, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
        self.model = model
        self.per_obj_inference = per_obj_inference
        self.fill_hole_area = fill_hole_area
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.bf16_context.__enter__()

    def __getattr__(self, name):
        model = super().__getattr__("model")
        if name == "model":
            return model
        return getattr(model, name)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use the sam2 predictor APIs instead. Check VideoTrackingWithPromptDemo class for details."
        )

    def add_output_per_object(self, *args, **kwargs):
        if self.per_obj_inference:
            return

        self._add_output_per_object(*args, **kwargs)

class Sam3MultiplexBase(Sam3VideoBase):
    def __init__(
        self,
        tracker,
        detector,
        ckpt_path=None,
        sam3_ckpt_path=None,
        score_threshold_detection=0.5,
        image_only_det_thresh=0.5,
        det_nms_thresh=0.0,
        det_nms_use_iom=False,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.5,
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=0,
        max_trk_keep_alive=8,
        min_trk_keep_alive=-4,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        allow_unoccluded_to_suppress: bool = False,
        decrease_trk_keep_alive_for_empty_masklets=False,
        o2o_matching_masklets_enable=False,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        sprinkle_removal_area=16,
        max_num_objects=1,
        num_obj_for_compile=1,
        max_num_kboxes=20,
        recondition_every_nth_frame=-1,
        use_iom_recondition=False,
        iom_thresh_recondition=0.8,
        iou_thresh_recondition=0.8,
        is_multiplex=False,
        masklet_confirmation_enable=False,
        masklet_confirmation_consecutive_det_thresh=3,
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.5,
        reapply_no_object_pointer: bool = False,
        running_in_prod=False,
        use_batched_grounding=False,
        batched_grounding_batch_size=1,
        **kwargs,
    ):
        nn.Module.__init__(self)
        assert isinstance(tracker, Sam3MultiplexTrackerPredictor)
        self.tracker = tracker
        assert isinstance(detector, Sam3MultiplexDetector)
        self.detector = detector
        if sam3_ckpt_path:
            ckpt = torch.load(sam3_ckpt_path, map_location="cpu", weights_only=True)
            self.detector.load_state_dict(ckpt["model"], strict=False)
        elif ckpt_path:
            self._load_checkpoint(ckpt_path, strict=False)
        self.score_threshold_detection = score_threshold_detection
        self.image_only_det_thresh = image_only_det_thresh
        self.det_nms_thresh = det_nms_thresh
        self.det_nms_use_iom = det_nms_use_iom
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh
        self.is_multiplex = is_multiplex
        self.running_in_prod = running_in_prod
        self.detector.running_in_prod = running_in_prod
        self.num_obj_for_compile = num_obj_for_compile
        self.max_num_objects = max_num_objects

        assert (
            self.is_multiplex == self.tracker.is_multiplex == self.detector.is_multiplex
        ), (
            f"is_multiplex must be the same for all models: {self.is_multiplex=}, {self.tracker.is_multiplex=}, {self.detector.is_multiplex=}"
        )

        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.allow_unoccluded_to_suppress = allow_unoccluded_to_suppress
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.sprinkle_removal_area = sprinkle_removal_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None

        self._profiler = None
        self._frame_count = 0
        self._profile_save_dir = os.getenv("PROFILE_SAVE_DIR", "/tmp/profiling")
        self._profiling_enabled = os.getenv("ENABLE_PROFILING", "0").lower() == "1"

        if max_num_objects > 0:
            multiplex_divisor = (
                self.tracker.multiplex_controller.allowed_bucket_capacity
                if self.is_multiplex
                else self.max_num_objects
            )
            num_obj_for_compile = math.ceil(
                max_num_objects / (self.world_size * multiplex_divisor)
            )
        else:
            max_num_objects = 10000
            num_obj_for_compile = 16
        logger.info(
            f"`setting max_num_objects` to {max_num_objects} -- creating {num_obj_for_compile=} objects for torch.compile cache"
        )
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.max_num_kboxes = max_num_kboxes
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.use_iom_recondition = use_iom_recondition
        self.iom_thresh_recondition = iom_thresh_recondition
        self.iou_thresh_recondition = iou_thresh_recondition
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score
        self.reapply_no_object_pointer = reapply_no_object_pointer

        self.use_batched_grounding = use_batched_grounding
        self.batched_grounding_batch_size = (
            batched_grounding_batch_size
        )

        if self.is_multiplex:
            assert not self.tracker.multiplex_controller.training, (
                "This model class should only be used for eval."
            )
            self.bucket_capacity: int = (
                self.tracker.multiplex_controller.allowed_bucket_capacity
            )

    def all_gather_cpu(self, tensor_list, tensor):
        if self._dist_pg_cpu is None:
            self._init_dist_pg_cpu()
        dist.broadcast(tensor_list, tensor, group=self._dist_pg_cpu)

    def all_gather_python_obj_cpu(self, object_list, python_obj):
        if self._dist_pg_cpu is None:
            self._init_dist_pg_cpu()
        dist.all_gather_object(object_list, python_obj, group=self._dist_pg_cpu)

    def broadcast_cpu(self, x, src):
        if self._dist_pg_cpu is None:
            self._init_dist_pg_cpu()
        dist.broadcast(x, src=src, group=self._dist_pg_cpu)

    def _start_profiling(self, frame_idx):
        self._profiling_enabled = os.getenv("ENABLE_PROFILING", "0").lower() == "1"
        self._profile_end_frame = int(os.getenv("PROFILE_END_FRAME", "-1"))
        """Start profiling for _det_track_one_frame if conditions are met."""
        if not self._profiling_enabled:
            return False

        if not getattr(self, "_warm_up_complete", False):
            return False

        if self._profiler is not None:
            return True

        os.makedirs(self._profile_save_dir, exist_ok=True)
        profile_path = os.path.join(
            self._profile_save_dir, f"det_track_frame_rank_{self.rank}.json.gz"
        )

        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            experimental_config=torch.profiler._ExperimentalConfig(
                profile_all_threads=True
            ),
        )
        self._profiler.start()
        self._current_profile_path = profile_path
        print(f"Started profiling frame on {frame_idx} on rank {self.rank}")
        return True

    def _stop_profiling(self):
        """Stop profiling and save trace."""
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler.export_chrome_trace(self._current_profile_path)
            print(f"Profiling trace saved to: {self._current_profile_path}")
            print(
                f"You can open this file in Perfetto (https://ui.perfetto.dev/) to visualize the trace"
            )
            self._profiler = None
            self._profiling_enabled = False
            os.environ["ENABLE_PROFILING"] = "0"

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
    ):
        profiling_enabled = self._start_profiling(frame_idx)

        try:
            return self._det_track_one_frame_impl(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                input_batch=input_batch,
                geometric_prompt=geometric_prompt,
                tracker_states_local=tracker_states_local,
                tracker_metadata_prev=tracker_metadata_prev,
                feature_cache=feature_cache,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                is_image_only=is_image_only,
            )
        finally:
            if profiling_enabled:
                if sys.exc_info()[0] is not None:
                    self._stop_profiling()
                else:
                    if (
                        (not reverse and frame_idx == num_frames - 1)
                        or (reverse and frame_idx == 0)
                        or self._profile_end_frame == frame_idx
                    ):
                        self._stop_profiling()

    def _det_track_one_frame_impl(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool,
    ):
        """
        This function handles one-step inference for the multiplex model in an SPMD manner.
        At a high-level, all GPUs execute the same function calls as if it's done on a single GPU,
        while under the hood, some function calls involve distributed computation based on sharded
        SAM2 states.

        - `input_batch` contains image and other inputs on the entire video; it should be identical across GPUs
        - `tracker_states_local` holds the local masklet information in this GPU shard
        - `tracker_metadata_prev` manages the metadata for SAM2 objects, such as which masklet is hold on which GPUs
          it contains both global and local masklet information
        """

        with torch.profiler.record_function("run_backbone_and_detection"):
            det_out, pos_pred_mask = self.run_backbone_and_detection(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                input_batch=input_batch,
                geometric_prompt=geometric_prompt,
                feature_cache=feature_cache,
                use_batched_grounding=self.use_batched_grounding,
                batched_grounding_batch_size=self.batched_grounding_batch_size,
            )

        with torch.profiler.record_function("run_tracker_propagation"):
            if tracker_metadata_prev == {}:
                tracker_metadata_prev.update(self._initialize_metadata())
            tracker_low_res_masks_global, tracker_obj_scores_global = (
                self.run_tracker_propagation(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    tracker_states_local=tracker_states_local,
                    tracker_metadata_prev=tracker_metadata_prev,
                )
            )

        with torch.profiler.record_function("GPU sync and filter"):
            assert pos_pred_mask.shape[0] == 1
            pos_pred_mask = pos_pred_mask.squeeze(0)
            det_out = {k: det_out[k][0] for k in det_out}
            pos_pred_mask_idx = pos_pred_mask.argsort(descending=True)
            pos_pred_mask = torch.index_select(
                pos_pred_mask, dim=0, index=pos_pred_mask_idx
            )
            det_out = {
                k: torch.index_select(det_out[k], dim=0, index=pos_pred_mask_idx)
                for k in det_out
            }

        with torch.profiler.record_function("run_tracker_update_planning_phase"):
            sam2_update_plan, tracker_metadata_new = (
                self.run_tracker_update_planning_phase(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_out=det_out,
                    det_keep=pos_pred_mask,
                    tracker_low_res_masks_global=tracker_low_res_masks_global,
                    tracker_obj_scores_global=tracker_obj_scores_global,
                    tracker_metadata_prev=tracker_metadata_prev,
                    tracker_states_local=tracker_states_local,
                    is_image_only=is_image_only,
                )
            )

        reconditioned_obj_ids = sam2_update_plan.get("reconditioned_obj_ids", set())
        det_to_matched_trk_obj_ids = sam2_update_plan.get(
            "det_to_matched_trk_obj_ids", {}
        )

        with torch.profiler.record_function("run_tracker_update_execution_phase"):
            tracker_states_local_new = self.run_tracker_update_execution_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_states_local=tracker_states_local,
                tracker_update_plan=sam2_update_plan,
                tracker_metadata_new=tracker_metadata_new,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        with torch.profiler.record_function("build_outputs"):
            if self.rank == 0:
                obj_id_to_mask = self.build_outputs(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_out=det_out,
                    tracker_low_res_masks_global=tracker_low_res_masks_global,
                    tracker_obj_scores_global=tracker_obj_scores_global,
                    tracker_metadata_prev=tracker_metadata_prev,
                    sam2_update_plan=sam2_update_plan,
                    orig_vid_height=orig_vid_height,
                    orig_vid_width=orig_vid_width,
                    reconditioned_obj_ids=reconditioned_obj_ids,
                    det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                )
                obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
            else:
                obj_id_to_mask, obj_id_to_score = {}, {}
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj_per_gpu"]),
            "num_obj_dropped": sam2_update_plan["num_obj_dropped_due_to_limit"],
        }
        if tracker_obj_scores_global.shape[0] > 0:
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid()
            sam2_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
            tracker_metadata_new["obj_id_to_sam2_score_frame_wise"][frame_idx].update(
                dict(zip(sam2_obj_ids, tracker_obj_scores_global))
            )

        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        num_frames: int,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        feature_cache: Dict,
        reverse: bool,
        use_batched_grounding: bool = False,
        batched_grounding_batch_size: int = 16,
    ):
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.detector.backbone.forward_text(
                input_batch.find_text_batch, device=self.device
            )
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        tracking_bounds = feature_cache.get("tracking_bounds", {})
        max_frame_num_to_track = tracking_bounds.get("max_frame_num_to_track")
        start_frame_idx = tracking_bounds.get("propagate_in_video_start_frame_idx")
        backbone_out = {
            "img_batch_all_stages": input_batch.img_batch,
            **text_outputs,
        }

        if use_batched_grounding:
            if "grounding_cache" not in feature_cache:
                feature_cache["grounding_cache"] = {}

            with torch.profiler.record_function(
                "forward_video_grounding_batched_multigpu"
            ):
                sam3_image_out, _ = (
                    self.detector.forward_video_grounding_batched_multigpu(
                        backbone_out=backbone_out,
                        find_inputs=input_batch.find_inputs,
                        geometric_prompt=geometric_prompt,
                        frame_idx=frame_idx,
                        num_frames=num_frames,
                        grounding_cache=feature_cache["grounding_cache"],
                        track_in_reverse=reverse,
                        return_sam2_backbone_feats=True,
                        run_nms=self.det_nms_thresh > 0.0,
                        nms_prob_thresh=self.score_threshold_detection,
                        nms_iou_thresh=self.det_nms_thresh,
                        nms_use_iom=self.det_nms_use_iom,
                        max_frame_num_to_track=max_frame_num_to_track,
                        propagate_in_video_start_frame_idx=start_frame_idx,
                        feature_cache=feature_cache,
                        batch_size=batched_grounding_batch_size,
                    )
                )
        else:
            if "multigpu_buffer" not in feature_cache:
                feature_cache["multigpu_buffer"] = {}

            with torch.profiler.record_function("forward_video_grounding_multigpu"):
                sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
                    backbone_out=backbone_out,
                    find_inputs=input_batch.find_inputs,
                    geometric_prompt=geometric_prompt,
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    multigpu_buffer=feature_cache["multigpu_buffer"],
                    track_in_reverse=reverse,
                    return_sam2_backbone_feats=True,
                    run_nms=self.det_nms_thresh > 0.0,
                    nms_prob_thresh=self.score_threshold_detection,
                    nms_iou_thresh=self.det_nms_thresh,
                    nms_use_iom=self.det_nms_use_iom,
                    max_frame_num_to_track=max_frame_num_to_track,
                    propagate_in_video_start_frame_idx=start_frame_idx,
                    feature_cache=feature_cache,
                )

        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]
        pos_pred_mask = pred_probs > self.score_threshold_detection

        if self.suppress_det_close_to_boundary:
            keep = self._suppress_detections_close_to_boundary(pred_boxes_xyxy)
            pos_pred_mask = pos_pred_mask & keep

        det_out = {
            "bbox": pred_boxes_xyxy,
            "mask": pred_masks,
            "scores": pred_probs,
        }

        backbone_cache = {}
        if self.is_multiplex:
            interaction_sam_mask_decoder = self.tracker.interactive_sam_mask_decoder
            interaction_backbone_fpn = [
                interaction_sam_mask_decoder.conv_s0(
                    sam3_image_out["interactive_backbone_fpn_0"]
                ),
                interaction_sam_mask_decoder.conv_s1(
                    sam3_image_out["interactive_backbone_fpn_1"]
                ),
                sam3_image_out[
                    "interactive_backbone_fpn_2"
                ],
            ]
            interaction_backbone_out = {
                "vision_features": interaction_backbone_fpn[-1],
                "vision_mask": None,
                "vision_pos_enc": sam3_image_out["interactive_backbone_pos_enc"],
                "backbone_fpn": [
                    NestedTensor(x, None) for x in interaction_backbone_fpn
                ],
            }
            backbone_cache["interactive"] = interaction_backbone_out
        sam_mask_decoder = self.tracker.sam_mask_decoder
        sam2_backbone_fpn = [
            sam_mask_decoder.conv_s0(sam3_image_out["sam2_backbone_fpn_0"]),
            sam_mask_decoder.conv_s1(sam3_image_out["sam2_backbone_fpn_1"]),
            sam3_image_out["sam2_backbone_fpn_2"],
        ]
        sam2_backbone_out = {
            "vision_features": sam2_backbone_fpn[-1],
            "vision_mask": None,
            "vision_pos_enc": sam3_image_out["sam2_backbone_pos_enc"],
            "backbone_fpn": [NestedTensor(x, None) for x in sam2_backbone_fpn],
        }
        backbone_cache["sam2_backbone_out"] = sam2_backbone_out

        with torch.profiler.record_function("run_backbone_and_detection.feature_cache"):
            feature_cache[frame_idx] = (
                input_batch.img_batch.tensors[frame_idx],
                backbone_cache,
            )
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out, pos_pred_mask

    def run_tracker_propagation(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, np.ndarray],
    ):
        with torch.profiler.record_function("propagate_tracker_one_frame_local_gpu"):
            obj_ids_local, low_res_masks_local, obj_scores_local = (
                self._propogate_tracker_one_frame_local_gpu(
                    tracker_states_local, frame_idx=frame_idx, reverse=reverse
                )
            )

        assert np.all(
            obj_ids_local == tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        ), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        )

        with torch.profiler.record_function("all_gather_low_res_masks_local"):
            _, H_mask, W_mask = low_res_masks_local.shape
            if self.world_size > 1:
                low_res_masks_local = low_res_masks_local.float().contiguous()
                obj_scores_local = obj_scores_local.float().contiguous()
                num_obj_this_gpu = tracker_metadata_prev["num_obj_per_gpu"][self.rank]
                assert low_res_masks_local.size(0) == num_obj_this_gpu
                assert obj_scores_local.size(0) == num_obj_this_gpu
                low_res_masks_peers = [
                    low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                    for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
                ]
                obj_scores_peers = [
                    obj_scores_local.new_empty(num_obj)
                    for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
                ]
                dist.all_gather(low_res_masks_peers, low_res_masks_local)
                dist.all_gather(obj_scores_peers, obj_scores_local)
                low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
                obj_scores_global = torch.cat(obj_scores_peers, dim=0)
            else:
                low_res_masks_global = low_res_masks_local
                obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: Dict[str, Tensor],
        trk_id_to_max_iou_high_conf_det: Dict[int, int],
        tracker_states_local: List[Any],
        tracker_metadata: Dict[str, np.ndarray],
        tracker_obj_scores_global: Tensor,
        tracker_low_res_masks_global: Tensor,
    ):
        reconditioned_obj_ids = set()
        HIGH_CONF_THRESH = 0.8
        input_mask_res = self.tracker.input_mask_size

        if len(trk_id_to_max_iou_high_conf_det) == 0:
            return tracker_states_local, reconditioned_obj_ids

        trk_obj_ids = list(trk_id_to_max_iou_high_conf_det.keys())
        det_indices = list(trk_id_to_max_iou_high_conf_det.values())

        obj_ids_all_gpu_t = torch.from_numpy(tracker_metadata["obj_ids_all_gpu"]).to(
            device=tracker_obj_scores_global.device
        )
        trk_obj_ids_t = torch.tensor(
            trk_obj_ids, device=tracker_obj_scores_global.device
        )
        det_indices_t = torch.tensor(
            det_indices, device=tracker_obj_scores_global.device
        )

        matches = trk_obj_ids_t.unsqueeze(1) == obj_ids_all_gpu_t.unsqueeze(0)
        obj_indices_t = matches.int().argmax(dim=1)

        obj_scores_batch = tracker_obj_scores_global[obj_indices_t].sigmoid()
        high_conf_mask = obj_scores_batch > HIGH_CONF_THRESH

        high_conf_mask_cpu = high_conf_mask.cpu().numpy()

        valid_trk_obj_ids = [
            tid for tid, valid in zip(trk_obj_ids, high_conf_mask_cpu) if valid
        ]
        valid_det_indices = [
            did for did, valid in zip(det_indices, high_conf_mask_cpu) if valid
        ]
        valid_obj_indices = obj_indices_t[high_conf_mask]

        if len(valid_trk_obj_ids) == 0:
            return tracker_states_local, reconditioned_obj_ids

        valid_det_indices_t = torch.tensor(
            valid_det_indices, device=det_out["mask"].device
        )

        new_masks = det_out["mask"][valid_det_indices_t]
        new_masks_binary = (
            F.interpolate(
                new_masks.unsqueeze(1),
                size=(input_mask_res, input_mask_res),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            > 0
        )

        old_masks = tracker_low_res_masks_global[valid_obj_indices]
        binary_agreement = (new_masks > 0) == (old_masks > 0)
        updated_masks = torch.where(binary_agreement, old_masks, new_masks)

        updated_masks = fill_holes_in_mask_scores(
            updated_masks.unsqueeze(1),
            fill_hole_area=self.fill_hole_area,
            sprinkle_removal_area=self.sprinkle_removal_area,
            fill_holes=True,
            remove_sprinkles=True,
        ).squeeze(1)

        tracker_low_res_masks_global[valid_obj_indices] = updated_masks

        if self.is_multiplex:
            state_to_recondition_info = {}
            for i, trk_obj_id in enumerate(valid_trk_obj_ids):
                for state_idx, inference_state in enumerate(tracker_states_local):
                    if trk_obj_id in inference_state["obj_ids"]:
                        if state_idx not in state_to_recondition_info:
                            state_to_recondition_info[state_idx] = []
                        state_to_recondition_info[state_idx].append(
                            (trk_obj_id, new_masks_binary[i])
                        )
                        break

            for state_idx, recondition_list in state_to_recondition_info.items():
                inference_state = tracker_states_local[state_idx]
                obj_ids_to_recondition = [item[0] for item in recondition_list]
                masks_to_recondition = torch.stack(
                    [item[1] for item in recondition_list]
                )
                with torch.profiler.record_function(
                    "_recodition_masklets.add_new_masks"
                ):
                    self.tracker.add_new_masks(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_ids=obj_ids_to_recondition,
                        masks=masks_to_recondition,
                        reconditioning=True,
                    )
                reconditioned_obj_ids.update(inference_state["obj_idx_to_id"].values())
        else:
            for i, trk_obj_id in enumerate(valid_trk_obj_ids):
                for inference_state in tracker_states_local:
                    if trk_obj_id in inference_state["obj_ids"]:
                        self.tracker.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=trk_obj_id,
                            mask=new_masks_binary[i],
                        )
                        reconditioned_obj_ids.update(
                            inference_state["obj_idx_to_id"].values()
                        )
                        break

        return tracker_states_local, reconditioned_obj_ids

    def _deepcopy(self, x):
        if True:
            return x
        return deepcopy(x)

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        det_keep: Tensor,
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, np.ndarray],
        tracker_states_local: List[Any],
        is_image_only: bool = False,
    ):
        with torch.profiler.record_function("initialize_tracker_metadata_new"):
            tracker_metadata_new = self._create_planning_metadata(tracker_metadata_prev)

        reconditioned_obj_ids = set()

        det_mask_preds: Tensor = det_out["mask"]
        det_scores: Tensor = det_out["scores"].float()
        with torch.profiler.record_function("associate_det_trk"):
            adt_result = self._associate_det_trk(
                det_masks=det_mask_preds,
                det_scores=det_scores,
                det_keep=det_keep,
                trk_masks=tracker_low_res_masks_global,
                trk_obj_ids=tracker_metadata_prev["obj_ids_all_gpu"],
                default_det_thresh=(
                    self.image_only_det_thresh if is_image_only else None
                ),
            )

        rank0_metadata_new = self._deepcopy(tracker_metadata_prev["rank0_metadata"])
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            with torch.profiler.record_function("_process_hotstart_gpu"):
                to_remove_mask, to_suppress_mask, gpu_metadata_new = (
                    self._process_hotstart_gpu(
                        frame_idx=frame_idx,
                        reverse=reverse,
                        adt_result=adt_result,
                        tracker_metadata_prev=tracker_metadata_prev,
                        gpu_metadata_prev=tracker_metadata_prev["gpu_metadata"],
                    )
                )
            tracker_metadata_new["gpu_metadata"] = gpu_metadata_new
        else:
            N_obj = tracker_low_res_masks_global.size(0)
            to_remove_mask = torch.zeros(
                N_obj, dtype=torch.bool, device=tracker_low_res_masks_global.device
            )
            to_suppress_mask = torch.zeros(
                N_obj, dtype=torch.bool, device=tracker_low_res_masks_global.device
            )
        tracker_metadata_new["rank0_metadata"] = rank0_metadata_new

        should_recondition_iou = False

        if self.reconstruction_bbox_iou_thresh > 0:
            adt_result = realize_adt_result(
                adt_result, tracker_metadata_prev, det_mask_preds
            )
        if (
            self.reconstruction_bbox_iou_thresh > 0
            and len(adt_result.trk_id_to_max_iou_high_conf_det) > 0
        ):
            with torch.profiler.record_function(
                "evaluate_reconstruction_bbox_iou_thresh"
            ):
                trk_obj_ids = adt_result.trk_id_to_max_iou_high_conf_det.keys()
                sam2_obj_ids_all_gpu = list(tracker_metadata_prev["obj_ids_all_gpu"])
                trk_ids = [
                    sam2_obj_ids_all_gpu.index(trk_obj_id)
                    for trk_obj_id in trk_obj_ids
                    if trk_obj_id in sam2_obj_ids_all_gpu
                ]
                det_ids = list(adt_result.trk_id_to_max_iou_high_conf_det.values())

                det_boxes_bbox_iou = det_out["bbox"][det_ids]
                det_scores_bbox_iou = det_out["scores"][det_ids]
                sam2_mask = tracker_low_res_masks_global[trk_ids]
                mask_binary = sam2_mask > 0
                sam2_box_pixels = mask_to_box(mask_binary.unsqueeze(1)).squeeze(1)
                mask_height, mask_width = sam2_mask.shape[-2:]
                sam2_box_normalized = sam2_box_pixels / torch.tensor(
                    [mask_width, mask_height, mask_width, mask_height],
                    device=sam2_box_pixels.device,
                )
                iou = fast_diag_box_iou(det_boxes_bbox_iou, sam2_box_normalized)[0]
                if iou < self.reconstruction_bbox_iou_thresh and torch.any(
                    det_scores_bbox_iou >= self.reconstruction_bbox_det_score
                ):
                    should_recondition_iou = True

        if (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
        ):
            adt_result = realize_adt_result(
                adt_result, tracker_metadata_prev, det_mask_preds
            )

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(adt_result.trk_id_to_max_iou_high_conf_det) > 0
        )

        if should_recondition_periodic or should_recondition_iou:
            adt_result = realize_adt_result(
                adt_result, tracker_metadata_prev, det_mask_preds
            )
            with torch.profiler.record_function("_recondition_masklets"):
                tracker_states_local, reconditioned_obj_ids = (
                    self._recondition_masklets(
                        frame_idx,
                        det_out,
                        adt_result.trk_id_to_max_iou_high_conf_det,
                        tracker_states_local,
                        tracker_metadata_prev,
                        tracker_obj_scores_global,
                        tracker_low_res_masks_global,
                    )
                )

            for state in tracker_states_local:
                if any(
                    obj_id in reconditioned_obj_ids
                    for obj_id in state.get("obj_ids", [])
                ):
                    self.tracker.propagate_in_video_preflight(
                        state, run_mem_encoder=True
                    )

        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    with torch.profiler.record_function(
                        "_suppress_overlapping_based_on_recent_occlusion"
                    ):
                        tracker_low_res_masks_global = (
                            self._suppress_overlapping_based_on_recent_occlusion(
                                frame_idx,
                                tracker_low_res_masks_global,
                                tracker_metadata_prev,
                                tracker_metadata_new,
                                to_remove_mask,
                                reverse,
                            )
                        )
            with torch.profiler.record_function("_tracker_update_memories"):
                self._tracker_update_memories(
                    tracker_states_local,
                    frame_idx,
                    tracker_metadata=tracker_metadata_prev,
                    low_res_masks=tracker_low_res_masks_global,
                )

        adt_result = realize_adt_result(
            adt_result, tracker_metadata_prev, det_mask_preds
        )
        new_det_obj_ids, new_det_gpu_ids, num_obj_dropped_due_to_limit = (
            adt_result.get_new_det_gpu_ids(
                tracker_metadata_prev, is_image_only, det_scores, self
            )
        )

        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            obj_ids_all_gpu = tracker_metadata_prev["obj_ids_all_gpu"]
            to_remove_cpu = to_remove_mask.cpu().numpy()
            obj_ids_newly_removed = set(obj_ids_all_gpu[to_remove_cpu].tolist())
        else:
            obj_ids_newly_removed = set()

        for rank in range(self.world_size):
            new_det_obj_ids_this_gpu = new_det_obj_ids[new_det_gpu_ids == rank]
            updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids_per_gpu"][rank]
            if len(new_det_obj_ids_this_gpu) > 0:
                updated_obj_ids_this_gpu = np.concatenate(
                    [updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu]
                )
            if len(obj_ids_newly_removed) > 0:
                is_removed = np.isin(
                    updated_obj_ids_this_gpu, list(obj_ids_newly_removed)
                )
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
            tracker_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            tracker_metadata_new["num_obj_per_gpu"][rank] = len(
                updated_obj_ids_this_gpu
            )
        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata_new["obj_ids_per_gpu"]
        )
        if len(new_det_obj_ids) > 0:
            det_scores_np: np.ndarray = det_scores.cpu().numpy()
            tracker_metadata_new["obj_id_to_score"].update(
                zip(new_det_obj_ids, det_scores_np[adt_result.new_det_fa_inds])
            )
            new_det_scores_tensor = det_scores[adt_result.new_det_fa_inds]
            tracker_metadata_new["obj_id_to_sam2_score_frame_wise"][frame_idx].update(
                zip(new_det_obj_ids, new_det_scores_tensor)
            )
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"],
                np.max(new_det_obj_ids),
            )
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_sam2_score_frame_wise"][frame_idx][
                obj_id
            ] = torch.tensor(-1e4, dtype=torch.float32, device=det_scores.device)
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        assert "rank0_metadata" in tracker_metadata_new
        if self.masklet_confirmation_enable:
            with torch.profiler.record_function("update_masklet_confirmation_status"):
                rank0_metadata = self.update_masklet_confirmation_status(
                    rank0_metadata=tracker_metadata_new["rank0_metadata"],
                    obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids_all_gpu"],
                    obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids_all_gpu"],
                    det_to_matched_trk_obj_ids=adt_result.det_to_matched_trk_obj_ids,
                    new_det_obj_ids=new_det_obj_ids,
                )
                tracker_metadata_new["rank0_metadata"] = rank0_metadata

        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            if (
                "gpu_metadata" in tracker_metadata_new
                and tracker_metadata_new["gpu_metadata"].get("N_obj", 0) > 0
            ):
                with torch.profiler.record_function("compact_gpu_metadata"):
                    gpu_meta = tracker_metadata_new["gpu_metadata"]
                    removed_mask = gpu_meta[
                        "removed_mask"
                    ]
                    keep_indices = torch.nonzero(~removed_mask, as_tuple=True)[0]

                    gpu_meta["obj_first_frame"] = gpu_meta["obj_first_frame"][
                        keep_indices
                    ]
                    gpu_meta["consecutive_unmatch_count"] = gpu_meta[
                        "consecutive_unmatch_count"
                    ][keep_indices]
                    gpu_meta["trk_keep_alive"] = gpu_meta["trk_keep_alive"][
                        keep_indices
                    ]
                    gpu_meta["removed_mask"] = gpu_meta["removed_mask"][
                        keep_indices
                    ]
                    gpu_meta["last_occluded_tensor"] = gpu_meta["last_occluded_tensor"][
                        keep_indices
                    ]

                    overlap_counts = gpu_meta["overlap_pair_counts"]
                    overlap_counts = overlap_counts[keep_indices][:, keep_indices]
                    gpu_meta["overlap_pair_counts"] = overlap_counts

                    gpu_meta["N_obj"] = keep_indices.size(0)

            num_new = len(new_det_obj_ids)
            if num_new > 0:
                with torch.profiler.record_function(
                    "extend_gpu_metadata_for_new_objects"
                ):
                    gpu_meta = tracker_metadata_new["gpu_metadata"]
                    device = det_scores.device
                    NEVER_OCCLUDED = -1

                    gpu_meta["obj_first_frame"] = torch.cat(
                        [
                            gpu_meta.get(
                                "obj_first_frame",
                                torch.empty(0, dtype=torch.long, device=device),
                            ),
                            torch.full(
                                (num_new,), frame_idx, dtype=torch.long, device=device
                            ),
                        ]
                    )
                    gpu_meta["consecutive_unmatch_count"] = torch.cat(
                        [
                            gpu_meta.get(
                                "consecutive_unmatch_count",
                                torch.empty(0, dtype=torch.long, device=device),
                            ),
                            torch.zeros(num_new, dtype=torch.long, device=device),
                        ]
                    )
                    gpu_meta["trk_keep_alive"] = torch.cat(
                        [
                            gpu_meta.get(
                                "trk_keep_alive",
                                torch.empty(0, dtype=torch.long, device=device),
                            ),
                            torch.full(
                                (num_new,),
                                self.init_trk_keep_alive,
                                dtype=torch.long,
                                device=device,
                            ),
                        ]
                    )
                    gpu_meta["removed_mask"] = torch.cat(
                        [
                            gpu_meta.get(
                                "removed_mask",
                                torch.empty(0, dtype=torch.bool, device=device),
                            ),
                            torch.zeros(num_new, dtype=torch.bool, device=device),
                        ]
                    )
                    gpu_meta["last_occluded_tensor"] = torch.cat(
                        [
                            gpu_meta.get(
                                "last_occluded_tensor",
                                torch.empty(0, dtype=torch.long, device=device),
                            ),
                            torch.full(
                                (num_new,),
                                NEVER_OCCLUDED,
                                dtype=torch.long,
                                device=device,
                            ),
                        ]
                    )

                    old_N = gpu_meta.get("N_obj", 0)
                    new_N = old_N + num_new
                    old_overlap = gpu_meta.get(
                        "overlap_pair_counts",
                        torch.zeros((0, 0), dtype=torch.long, device=device),
                    )
                    new_overlap = torch.zeros(
                        (new_N, new_N), dtype=torch.long, device=device
                    )
                    if old_N > 0:
                        new_overlap[:old_N, :old_N] = old_overlap
                    gpu_meta["overlap_pair_counts"] = new_overlap

                    gpu_meta["N_obj"] = new_N

        sam2_update_plan = {
            "new_det_fa_inds": adt_result.new_det_fa_inds,
            "new_det_obj_ids": new_det_obj_ids,
            "new_det_gpu_ids": new_det_gpu_ids,
            "unmatched_trk_obj_ids": adt_result.unmatched_trk_obj_ids,
            "det_to_matched_trk_obj_ids": adt_result.det_to_matched_trk_obj_ids,
            "obj_ids_newly_removed": obj_ids_newly_removed,
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,
            "trk_id_to_max_iou_high_conf_det": adt_result.trk_id_to_max_iou_high_conf_det,
            "reconditioned_obj_ids": reconditioned_obj_ids,
        }
        return sam2_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_prev: Dict[str, Any],
        tracker_metadata_new: Dict[str, Any],
        to_remove_mask: Tensor,
        reverse: bool = False,
    ):
        """
        Suppress overlapping masks based on the most recent occlusion information. If an object is removed by hotstart, we always suppress it if it overlaps with any other object.
        Args:
            frame_idx (int): The current frame index.
            tracker_low_res_masks_global (Tensor): The low-resolution masks for the current frame.
            tracker_metadata_prev (Dict[str, Any]): The metadata from the previous frame.
            tracker_metadata_new (Dict[str, Any]): The metadata for the current frame (with updated gpu_metadata from _process_hotstart_gpu).
            to_remove_mask (Tensor): GPU boolean mask (N_obj,) indicating which objects are removed.
        Return:
            Tensor: The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = tracker_metadata_prev["obj_ids_all_gpu"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        num_ids = len(obj_ids_global)

        assert batch_size == num_ids, (
            f"Mask/metadata count mismatch in _suppress_overlapping: "
            f"batch_size={batch_size}, num_ids={num_ids}, frame_idx={frame_idx}"
        )

        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        if batch_size > 0:
            assert len(obj_ids_global) == batch_size, (
                f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            )
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000

            device = binary_tracker_low_res_masks_global.device

            gpu_metadata_new = tracker_metadata_new["gpu_metadata"]
            last_occluded_prev = gpu_metadata_new["last_occluded_tensor"]

            assert last_occluded_prev.size(0) == batch_size, (
                f"last_occluded_tensor size mismatch: {last_occluded_prev.size(0)} vs {batch_size}. "
                f"This indicates gpu_metadata tensors are out of sync."
            )

            last_occluded_prev = torch.where(
                to_remove_mask,
                torch.full_like(last_occluded_prev, ALWAYS_OCCLUDED),
                last_occluded_prev,
            )

            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx

            tracker_metadata_new["gpu_metadata"]["last_occluded_tensor"] = (
                last_occluded_new
            )

            tracker_metadata_new[
                "obj_id_to_last_occluded"
            ] = {}

            NO_OBJ_LOGIT = -10
            tracker_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def _create_planning_metadata(self, tracker_metadata_prev):
        """Extend planning metadata with multiplex-specific fields."""
        metadata = super()._create_planning_metadata(tracker_metadata_prev)
        if self.is_multiplex:
            metadata["num_buc_per_gpu"] = self._deepcopy(
                tracker_metadata_prev["num_buc_per_gpu"]
            )
        metadata["gpu_metadata"] = tracker_metadata_prev["gpu_metadata"]
        return metadata

    def _post_execution_phase_hook(self, tracker_states_local, tracker_metadata_new):
        """Update bucket count after execution phase (multiplex-specific)."""
        if self.is_multiplex and tracker_metadata_new is not None:
            actual_bucket_count = self._count_buckets_in_states(tracker_states_local)
            tracker_metadata_new["num_buc_per_gpu"][self.rank] = actual_bucket_count

    def _count_buckets_in_states(self, tracker_states_local: List[Any]) -> int:
        """Count the total number of buckets across all states."""
        if not self.is_multiplex:
            return 0
        total_buckets = 0
        for state in tracker_states_local:
            if "multiplex_state" in state:
                total_buckets += state["multiplex_state"].num_buckets
        return total_buckets

    def build_outputs(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[
            str, Tensor
        ],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, np.ndarray],
        sam2_update_plan: Dict[str, np.ndarray],
        orig_vid_height: int,
        orig_vid_width: int,
        reconditioned_obj_ids: set = None,
        det_to_matched_trk_obj_ids: dict = None,
    ):
        new_det_fa_inds: np.ndarray = sam2_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = sam2_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}

        existing_masklet_obj_ids_all = tracker_metadata_prev["obj_ids_all_gpu"]
        existing_masklet_obj_ids_per_gpu = np.concatenate(
            tracker_metadata_prev["obj_ids_per_gpu"]
        )
        use_per_gpu_ids = len(existing_masklet_obj_ids_per_gpu) != len(
            existing_masklet_obj_ids_all
        ) or not np.array_equal(
            existing_masklet_obj_ids_per_gpu, existing_masklet_obj_ids_all
        )
        existing_masklet_obj_ids = (
            existing_masklet_obj_ids_per_gpu
            if use_per_gpu_ids
            else existing_masklet_obj_ids_all
        )
        existing_masklet_video_res_masks = F.interpolate(
            tracker_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )
        num_masks = existing_masklet_video_res_masks.size(0)
        num_ids = len(existing_masklet_obj_ids)
        if num_masks != num_ids:
            if num_masks < num_ids:
                pad = existing_masklet_video_res_masks.new_zeros(
                    (num_ids - num_masks, 1, orig_vid_height, orig_vid_width)
                )
                existing_masklet_video_res_masks = torch.cat(
                    [existing_masklet_video_res_masks, pad], dim=0
                )
            else:
                existing_masklet_video_res_masks = existing_masklet_video_res_masks[
                    :num_ids
                ]
        existing_masklet_binary = existing_masklet_video_res_masks > 0
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask

        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        new_det_low_res_masks = fill_holes_in_mask_scores(
            new_det_low_res_masks,
            fill_hole_area=self.fill_hole_area,
            sprinkle_removal_area=self.sprinkle_removal_area,
            fill_holes=True,
            remove_sprinkles=True,
        )
        new_masklet_video_res_masks = F.interpolate(
            new_det_low_res_masks,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )

        new_masklet_binary = new_masklet_video_res_masks > 0
        assert len(new_det_obj_ids) == len(new_masklet_video_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_masklet_binary):
            obj_id_to_mask[obj_id] = mask

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: Tensor,
        obj_ids: np.ndarray,
        frame_idx: int = None,
        reverse: bool = False,
    ):
        assert binary_low_res_masks.dtype == torch.bool, (
            f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        )
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)

        mask_iou_thresh = (
            iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)

        last_occ_expanded_i = last_occluded.unsqueeze(1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)
        cmp_op = torch.gt if not reverse else torch.lt

        if self.allow_unoccluded_to_suppress:
            suppress_i_mask = overlapping_pairs & cmp_op(
                last_occ_expanded_i, last_occ_expanded_j
            )

            suppress_j_mask = overlapping_pairs & cmp_op(
                last_occ_expanded_j, last_occ_expanded_i
            )
        else:
            suppress_i_mask = (
                overlapping_pairs
                & cmp_op(
                    last_occ_expanded_i, last_occ_expanded_j
                )
                & (last_occ_expanded_j > -1)
            )

            suppress_j_mask = (
                overlapping_pairs
                & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
                & (
                    last_occ_expanded_i > -1
                )
            )

        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        if (
            self.rank == 0
            and logger.isEnabledFor(logging.DEBUG)
            and frame_idx is not None
        ):
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            batch_size = suppress_i_mask.shape[0]

            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(
        self,
        inference_states: List[Any],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = False,
        filter_obj_ids: Optional[List[int]] = None,
    ):
        """
        inference_states: List of inference states, each state corresponds to a different set of objects.
        """
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue

            num_frames_propagated = 0
            with torch.profiler.record_function("sam2_predictor.propagate_in_video"):
                for out in self.tracker.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=0,
                    reverse=reverse,
                    tqdm_disable=True,
                    run_mem_encoder=run_mem_encoder,
                ):
                    out_frame_idx, out_obj_ids, out_low_res_masks, _, out_obj_scores = (
                        out
                    )
                    num_frames_propagated += 1

            assert num_frames_propagated == 1 and out_frame_idx == frame_idx, (
                f"num_frames_propagated: {num_frames_propagated}, out_frame_idx: {out_frame_idx}, frame_idx: {frame_idx}"
            )
            assert isinstance(out_obj_ids, list)
            if filter_obj_ids is not None:
                if len(out_obj_ids) > 0:
                    max_mask_rows = out_low_res_masks.shape[0]
                    max_score_rows = out_obj_scores.shape[0]
                    if (
                        len(filter_obj_ids) == 1
                        and max_mask_rows == 1
                        and max_score_rows == 1
                    ):
                        out_obj_ids = [filter_obj_ids[0]]
                        keep_indices = [0]
                    else:
                        keep_indices = [
                            i
                            for i, oid in enumerate(out_obj_ids)
                            if oid in filter_obj_ids
                            and i < max_mask_rows
                            and i < max_score_rows
                        ]
                else:
                    keep_indices = []
                if len(keep_indices) > 0:
                    idx_tensor = torch.as_tensor(
                        keep_indices, device=out_low_res_masks.device, dtype=torch.long
                    )
                    out_low_res_masks = out_low_res_masks.index_select(
                        dim=0, index=idx_tensor
                    )
                    out_obj_scores = out_obj_scores.index_select(
                        dim=0, index=idx_tensor
                    )
                    out_obj_ids = [out_obj_ids[i] for i in keep_indices]
                else:
                    out_obj_ids = []

            if len(out_obj_ids) > 0:
                obj_ids_local.extend(out_obj_ids)
                low_res_masks_list.append(out_low_res_masks.squeeze(1))
                obj_scores_list.append(out_obj_scores.squeeze(1))

        with torch.profiler.record_function(
            "sam2_predictor.propagate_in_video.fill_holes"
        ):
            H_mask = W_mask = self.tracker.low_res_mask_size
            if len(low_res_masks_list) > 0:
                low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
                obj_scores_local = torch.cat(obj_scores_list, dim=0)
                assert low_res_masks_local.shape[1:] == (H_mask, W_mask)

                low_res_masks_local = fill_holes_in_mask_scores(
                    low_res_masks_local.unsqueeze(1),
                    fill_hole_area=self.fill_hole_area,
                    sprinkle_removal_area=self.sprinkle_removal_area,
                    fill_holes=True,
                    remove_sprinkles=True,
                )
                low_res_masks_local = low_res_masks_local.squeeze(1)
            else:
                low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)
                obj_scores_local = torch.zeros(0, device=self.device)

        if self.is_multiplex and self.tracker.is_multiplex_dynamic:
            if obj_ids_local != sorted(obj_ids_local):
                sort_indices = sorted(
                    range(len(obj_ids_local)), key=lambda i: obj_ids_local[i]
                )
                obj_ids_local = [obj_ids_local[i] for i in sort_indices]
                low_res_masks_local = low_res_masks_local[sort_indices]
                obj_scores_local = obj_scores_local[sort_indices]

        if self.is_multiplex and self.tracker.is_multiplex_dynamic:
            if obj_ids_local != sorted(obj_ids_local):
                sort_indices = sorted(
                    range(len(obj_ids_local)), key=lambda i: obj_ids_local[i]
                )
                obj_ids_local = [obj_ids_local[i] for i in sort_indices]
                if low_res_masks_local.shape[0] == len(sort_indices):
                    low_res_masks_local = low_res_masks_local[sort_indices]
                    obj_scores_local = obj_scores_local[sort_indices]

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores: Tensor,
        det_keep: Tensor,
        trk_masks: Tensor,
        trk_obj_ids: np.ndarray,
        default_det_thresh: Optional[float] = None,
    ):
        """
        Match detections on the current frame with the existing masklets.

        Args:
          - det_masks: (N, H, W) tensor of predicted masks
          - det_scores: (N,) array of detection scores
          - trk_masks: (M, H, W) tensor of track masks
          - trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
          - new_det_fa_inds: array of new object indices among in FA detection outputs
          - unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched
            to any detections on this frame (for unmatched, we only count masklets with >0 area)
          - det_to_matched_trk_obj_ids: dict[int, np.ndarray]: mapping from FA detection indices
            to the list of matched tracklet object IDs
          - empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
        """
        HIGH_CONF_THRESH = 0.8

        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = (
            self.new_det_thresh if default_det_thresh is None else default_det_thresh
        )

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(trk_obj_ids), (
            f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        )
        if trk_masks.size(0) == 0:
            with torch.profiler.record_function("No tracklets"):
                num_trk = 0
                is_new_det = det_scores >= new_det_thresh
                trk_is_unmatched = torch.zeros(
                    num_trk, dtype=torch.bool, device=det_scores.device
                )
                trk_is_nonempty = torch.zeros(
                    num_trk, dtype=torch.bool, device=det_scores.device
                )
                num_det = det_scores.shape[0]
                det_to_max_iou_trk_idx = torch.full(
                    (num_det,), -1, dtype=torch.long, device=det_scores.device
                )
                det_is_high_conf = det_scores >= HIGH_CONF_THRESH
                det_is_high_iou = torch.zeros(
                    num_det, dtype=torch.bool, device=det_scores.device
                )
                im_mask = torch.zeros(
                    num_det, num_trk, dtype=torch.bool, device=det_scores.device
                )
                return LazyAssociateDetTrkResult(
                    trk_is_unmatched,
                    trk_is_nonempty,
                    is_new_det,
                    det_to_max_iou_trk_idx,
                    det_is_high_conf,
                    det_is_high_iou,
                    det_keep,
                    im_mask,
                )
        elif det_masks.size(0) == 0:
            with torch.profiler.record_function("No detections"):
                assert det_keep.size(0) == 0
                trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2))
                num_det = 0
                num_trk = trk_masks.shape[0]
                trk_is_unmatched = torch.ones(
                    num_trk, dtype=torch.bool, device=trk_masks.device
                )
                trk_is_nonempty_tensor = trk_is_nonempty.to(trk_masks.device)
                is_new_det = torch.zeros(
                    num_det, dtype=torch.bool, device=trk_masks.device
                )
                det_to_max_iou_trk_idx = torch.full(
                    (num_det,), -1, dtype=torch.long, device=trk_masks.device
                )
                det_is_high_conf = torch.zeros(
                    num_det, dtype=torch.bool, device=trk_masks.device
                )
                det_is_high_iou = torch.zeros(
                    num_det, dtype=torch.bool, device=trk_masks.device
                )
                im_mask = torch.zeros(
                    num_det, num_trk, dtype=torch.bool, device=trk_masks.device
                )
                return LazyAssociateDetTrkResult(
                    trk_is_unmatched,
                    trk_is_nonempty_tensor,
                    is_new_det,
                    det_to_max_iou_trk_idx,
                    det_is_high_conf,
                    det_is_high_iou,
                    det_keep,
                    im_mask,
                )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        with torch.profiler.record_function("associate_det_trk_compilable"):
            if trk_masks.shape[0] < self.max_num_objects:
                padding_size = self.max_num_objects - trk_masks.shape[0]
                trk_masks_padded = torch.cat(
                    [
                        trk_masks,
                        torch.zeros(
                            padding_size,
                            *trk_masks.shape[1:],
                            device=trk_masks.device,
                            dtype=trk_masks.dtype,
                        ),
                    ],
                    dim=0,
                )
            else:
                trk_masks_padded = trk_masks
            result = _associate_det_trk_compilable(
                det_masks,
                det_scores,
                det_keep,
                trk_masks_padded,
                new_det_thresh,
                iou_threshold_trk,
                iou_threshold,
                HIGH_CONF_THRESH,
                self.use_iom_recondition,
                self.o2o_matching_masklets_enable,
                self.iom_thresh_recondition,
                self.iou_thresh_recondition,
            )
            (
                trk_is_unmatched,
                trk_is_nonempty,
                is_new_det,
                det_to_max_iou_trk_idx,
                det_is_high_conf,
                det_is_high_iou,
                det_keep,
                im_mask,
            ) = result
            trk_is_unmatched = trk_is_unmatched[: trk_masks.shape[0]]
            trk_is_nonempty = trk_is_nonempty[: trk_masks.shape[0]]
            im_mask = im_mask[:, : trk_masks.shape[0]]

        return LazyAssociateDetTrkResult(
            trk_is_unmatched,
            trk_is_nonempty,
            is_new_det,
            det_to_max_iou_trk_idx,
            det_is_high_conf,
            det_is_high_iou,
            det_keep,
            im_mask,
        )

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        """Distribute the new objects to the GPUs with the least workload."""
        workload_per_gpu: np.ndarray = prev_workload_per_gpu.copy()
        new_det_gpu_ids = np.zeros(new_det_num, np.int64)

        if self.is_multiplex:
            for i in range(0, new_det_num, self.bucket_capacity):
                min_gpu = np.argmin(workload_per_gpu)
                new_det_gpu_ids[i : i + self.bucket_capacity] = min_gpu
                workload_per_gpu[min_gpu] += 1
        else:
            for i in range(len(new_det_gpu_ids)):
                min_gpu = np.argmin(workload_per_gpu)
                new_det_gpu_ids[i] = min_gpu
                workload_per_gpu[min_gpu] += 1
        return new_det_gpu_ids

    def _process_hotstart_gpu(
        self,
        frame_idx: int,
        reverse: bool,
        adt_result,
        tracker_metadata_prev: Dict[str, Any],
        gpu_metadata_prev: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Compute removal/suppression masks entirely on GPU without ANY syncs or branches.

        Uses position-indexed metadata (indexed 0 to N_obj-1) instead of obj_id-indexed
        to avoid needing obj_ids as GPU tensor.

        Returns:
            to_remove: boolean tensor (N_obj,) - objects to remove this frame
            to_suppress: boolean tensor (N_obj,) - objec    ts to suppress (overlap suppression)
            gpu_metadata_new: updated GPU metadata for next frame
        """
        if isinstance(adt_result, RealizedAssociateDetTrkresult):
            empty_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            return empty_mask, empty_mask, {"N_obj": 0}

        device = adt_result.trk_is_unmatched.device
        N_obj = adt_result.trk_is_unmatched.size(0)

        prev_N_obj = gpu_metadata_prev.get("N_obj", 0)

        if prev_N_obj > 0:
            obj_first_frame_prev = gpu_metadata_prev["obj_first_frame"]
            consecutive_unmatch_count_prev = gpu_metadata_prev[
                "consecutive_unmatch_count"
            ]
            trk_keep_alive_prev = gpu_metadata_prev["trk_keep_alive"]
            removed_mask_prev = gpu_metadata_prev[
                "removed_mask"
            ]
            overlap_pair_counts_prev = gpu_metadata_prev[
                "overlap_pair_counts"
            ]
            last_occluded_prev = gpu_metadata_prev[
                "last_occluded_tensor"
            ]
        else:
            obj_first_frame_prev = None
            consecutive_unmatch_count_prev = None
            trk_keep_alive_prev = None
            removed_mask_prev = None
            overlap_pair_counts_prev = None
            last_occluded_prev = None

        assert N_obj == prev_N_obj, (
            f"N_obj ({N_obj}) should equal prev_N_obj ({prev_N_obj}); new objects handled after compaction"
        )

        NEVER_OCCLUDED = -1
        obj_first_frame = (
            obj_first_frame_prev
            if obj_first_frame_prev is not None
            else torch.full((N_obj,), frame_idx, dtype=torch.long, device=device)
        )
        consecutive_unmatch_count = (
            consecutive_unmatch_count_prev
            if consecutive_unmatch_count_prev is not None
            else torch.zeros(N_obj, dtype=torch.long, device=device)
        )
        trk_keep_alive = (
            trk_keep_alive_prev
            if trk_keep_alive_prev is not None
            else torch.zeros(N_obj, dtype=torch.long, device=device)
        )
        removed_mask = (
            removed_mask_prev
            if removed_mask_prev is not None
            else torch.zeros(N_obj, dtype=torch.bool, device=device)
        )
        overlap_pair_counts = (
            overlap_pair_counts_prev
            if overlap_pair_counts_prev is not None
            else torch.zeros((N_obj, N_obj), dtype=torch.long, device=device)
        )
        last_occluded = (
            last_occluded_prev
            if last_occluded_prev is not None
            else torch.full((N_obj,), NEVER_OCCLUDED, dtype=torch.long, device=device)
        )

        trk_is_matched = adt_result.im_mask.any(dim=0)

        trk_keep_alive = torch.where(
            trk_is_matched, trk_keep_alive + 1, trk_keep_alive - 1
        )
        trk_keep_alive = torch.clamp(
            trk_keep_alive, min=self.min_trk_keep_alive, max=self.max_trk_keep_alive
        )

        if self.decrease_trk_keep_alive_for_empty_masklets:
            trk_keep_alive = torch.where(
                ~adt_result.trk_is_nonempty,
                torch.clamp(trk_keep_alive - 1, min=self.min_trk_keep_alive),
                trk_keep_alive,
            )

        consecutive_unmatch_count = torch.where(
            adt_result.trk_is_unmatched,
            consecutive_unmatch_count + 1,
            consecutive_unmatch_count,
        )

        tracks_per_det = adt_result.im_mask.sum(dim=1)
        multi_match_mask = tracks_per_det > 1

        multi_match_tracks = adt_result.im_mask & multi_match_mask.unsqueeze(
            1
        )

        pairwise_overlap_this_frame = torch.einsum(
            "di,dj->dij", multi_match_tracks.float(), multi_match_tracks.float()
        )

        overlap_increment = pairwise_overlap_this_frame.sum(dim=0)
        overlap_increment.fill_diagonal_(0)
        overlap_increment = torch.triu(
            overlap_increment, diagonal=1
        )

        overlap_pair_counts = overlap_pair_counts + overlap_increment.long()

        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )

        is_within_hotstart = (
            (obj_first_frame > hotstart_diff)
            if not reverse
            else (obj_first_frame < hotstart_diff)
        )

        remove_by_unmatch = (
            is_within_hotstart
            & (consecutive_unmatch_count >= self.hotstart_unmatch_thresh)
            & ~removed_mask
        )

        suppress_by_unmatch = (
            (trk_keep_alive <= 0)
            & torch.tensor(not self.suppress_unmatched_only_within_hotstart)
            .pin_memory()
            .to(device=device, non_blocking=True)
            & ~removed_mask
            & ~remove_by_unmatch
        )

        first_frames_i = obj_first_frame.unsqueeze(1)
        first_frames_j = obj_first_frame.unsqueeze(0)

        if not reverse:
            is_earlier_matrix = first_frames_i < first_frames_j
        else:
            is_earlier_matrix = first_frames_i > first_frames_j

        if N_obj == 0:
            to_remove = remove_by_unmatch
        else:
            overlap_with_earlier = torch.where(
                is_earlier_matrix,
                overlap_pair_counts,
                torch.zeros_like(overlap_pair_counts),
            )

            max_overlap_with_earlier, _ = overlap_with_earlier.max(dim=0)

            remove_by_overlap = (
                is_within_hotstart
                & (max_overlap_with_earlier >= self.hotstart_dup_thresh)
                & ~removed_mask
            )

            to_remove = remove_by_unmatch | remove_by_overlap

        to_suppress = suppress_by_unmatch

        removed_mask = removed_mask | to_remove

        gpu_metadata_new = {
            "N_obj": N_obj,
            "obj_first_frame": obj_first_frame,
            "consecutive_unmatch_count": consecutive_unmatch_count,
            "trk_keep_alive": trk_keep_alive,
            "removed_mask": removed_mask,
            "overlap_pair_counts": overlap_pair_counts,
            "last_occluded_tensor": last_occluded,
        }

        return to_remove, to_suppress, gpu_metadata_new

    def _process_hotstart(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: Dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
        empty_trk_obj_ids: np.ndarray,
        unmatched_trk_obj_ids: np.ndarray,
        rank0_metadata: Dict[str, Any],
        tracker_metadata: Dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        obj_first_frame_idx = rank0_metadata["obj_first_frame_idx"]
        unmatched_frame_inds = rank0_metadata["unmatched_frame_inds"]
        trk_keep_alive = rank0_metadata["trk_keep_alive"]
        overlap_pair_to_frame_inds = rank0_metadata["overlap_pair_to_frame_inds"]
        removed_obj_ids = rank0_metadata["removed_obj_ids"]
        suppressed_obj_ids = rank0_metadata["suppressed_obj_ids"][frame_idx]

        obj_ids_newly_removed = set()
        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )

        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            trk_keep_alive[obj_id] = min(
                self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1
            )
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            trk_keep_alive[obj_id] = max(
                self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
            )
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                trk_keep_alive[obj_id] = max(
                    self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
                )

        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id] > hotstart_diff and not reverse
                ) or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    logger.info(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                logger.debug(
                    f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched"
                )
                suppressed_obj_ids.add(obj_id)

        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    logger.info(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, rank0_metadata

    def _tracker_update_memories(
        self,
        sam2_inference_states: List[Any],
        frame_idx: int,
        tracker_metadata: Dict[str, Any],
        low_res_masks: Tensor,
    ):
        """
        Run Sam2 memory encoder, enforcing non-overlapping constraints globally.
        """
        if len(sam2_inference_states) == 0:
            return
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(high_res_H, high_res_W),
            mode="bilinear",
            align_corners=False,
        )
        with torch.profiler.record_function(
            "sam2_predictor.propagate_in_video.apply_non_overlapping_constraints"
        ):
            high_res_masks = self.tracker._suppress_object_pw_area_shrinkage(
                high_res_masks
            )
        object_score_logits = torch.where(
            (high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0
        )

        if self.is_multiplex and self.tracker.is_multiplex_dynamic:
            object_idx_assignment: dict[int, list[int]] = {}
            all_object_ids: list[int] = []
            object_id_to_state_i: dict[int, int] = {}
            for state_i, sam2_state in enumerate(sam2_inference_states):
                obj_ids = sam2_state["obj_ids"]
                all_object_ids.extend(obj_ids)
                for obj_id in obj_ids:
                    object_id_to_state_i[obj_id] = state_i
                object_idx_assignment[state_i] = []
            sorted_indices = sorted(
                range(len(all_object_ids)), key=lambda i: all_object_ids[i]
            )
            for global_idx, local_idx in enumerate(sorted_indices):
                obj_id = all_object_ids[local_idx]
                object_idx_assignment[object_id_to_state_i[obj_id]].append(global_idx)

        start_idx_gpu = sum(tracker_metadata["num_obj_per_gpu"][: self.rank])
        start_idx_state = start_idx_gpu
        for state_i, sam2_state in enumerate(sam2_inference_states):
            num_obj_per_state = len(sam2_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            if self.is_multiplex and self.tracker.is_multiplex_dynamic:
                local_idx = (
                    torch.tensor(object_idx_assignment[state_i])
                    .pin_memory()
                    .to(device=high_res_masks.device, non_blocking=True)
                )
                local_high_res_masks = high_res_masks[local_idx]
                local_object_score_logits = object_score_logits[local_idx]
            else:
                end_idx_state = start_idx_state + num_obj_per_state
                local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
                local_object_score_logits = object_score_logits[
                    start_idx_state:end_idx_state
                ]
            local_batch_size = local_high_res_masks.size(0)
            encoded_mem = self.tracker._run_memory_encoder(
                sam2_state,
                frame_idx,
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
            )
            if self.is_multiplex:
                (
                    local_maskmem_features,
                    local_maskmem_pos_enc,
                    local_image_features,
                    local_image_pos_enc,
                ) = encoded_mem
            else:
                local_maskmem_features, local_maskmem_pos_enc = encoded_mem

            output_dict = sam2_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = (
                    local_maskmem_features
                )
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [
                    pos for pos in local_maskmem_pos_enc
                ]
                if self.is_multiplex:
                    output_dict[storage_key][frame_idx]["image_features"] = (
                        local_image_features
                    )
                    output_dict[storage_key][frame_idx]["image_pos_enc"] = (
                        local_image_pos_enc
                    )

                    if self.reapply_no_object_pointer:
                        newly_suppressed_objects = (
                            output_dict[storage_key][frame_idx]["object_score_logits"]
                            > self.tracker.object_score_logit_threshold
                        ) & (local_object_score_logits < 0)
                        if torch.any(newly_suppressed_objects):
                            existing_pointers = output_dict[storage_key][frame_idx][
                                "obj_ptr"
                            ]

                            multiplex_state = sam2_state["multiplex_state"]
                            existing_pointers = multiplex_state.demux(existing_pointers)

                            newly_suppressed_objects = newly_suppressed_objects.float()
                            new_pointers = (
                                newly_suppressed_objects
                                * self.tracker.no_obj_ptr_linear(existing_pointers)
                                + (1 - newly_suppressed_objects) * existing_pointers
                            )

                            output_dict[storage_key][frame_idx]["obj_ptr"] = (
                                multiplex_state.mux(new_pointers)
                            )
                elif self.reapply_no_object_pointer:
                    raise NotImplementedError(
                        "reapply_no_object_pointer is not implemented for non-multiplex"
                    )

                self.tracker.add_output_per_object(
                    inference_state=sam2_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: List[int],
        new_obj_masks: Tensor,
        tracker_states_local: List[Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        """Add new objects to SAM2 inference states."""

        prev_sam2_state = (
            tracker_states_local[0] if len(tracker_states_local) > 0 else None
        )
        if self.tracker.is_multiplex_dynamic:
            num_new_objects = len(new_obj_ids)

            best_state = None
            best_available_slots = float("inf")

            for state in tracker_states_local:
                available_slots = state["multiplex_state"].available_slots
                if (
                    available_slots >= num_new_objects
                    and available_slots < best_available_slots
                ):
                    best_state = state
                    best_available_slots = available_slots

            if best_state is not None:
                new_sam2_state = best_state
            else:
                new_sam2_state = self.tracker.init_state(
                    cached_features=feature_cache,
                    video_height=orig_vid_height,
                    video_width=orig_vid_width,
                    num_frames=num_frames,
                )
                new_sam2_state["backbone_out"] = (
                    prev_sam2_state.get("backbone_out", None)
                    if prev_sam2_state is not None
                    else None
                )
                tracker_states_local.append(new_sam2_state)
        else:
            if self.tracker.per_obj_inference:
                if prev_sam2_state is not None:
                    new_sam2_state = prev_sam2_state
                else:
                    new_sam2_state = self.tracker.init_state(
                        cached_features=feature_cache,
                        video_height=orig_vid_height,
                        video_width=orig_vid_width,
                        num_frames=num_frames,
                    )
                    new_sam2_state["backbone_out"] = None
                    tracker_states_local = [new_sam2_state]
            else:
                new_sam2_state = self.tracker.init_state(
                    cached_features=feature_cache,
                    video_height=orig_vid_height,
                    video_width=orig_vid_width,
                    num_frames=num_frames,
                )
                new_sam2_state["backbone_out"] = (
                    prev_sam2_state.get("backbone_out", None)
                    if prev_sam2_state is not None
                    else None
                )
                tracker_states_local.append(new_sam2_state)

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        input_mask_res = self.tracker.input_mask_size
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(1),
            size=(input_mask_res, input_mask_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        new_obj_masks = new_obj_masks > 0

        if self.is_multiplex:
            self.tracker.add_new_masks(
                inference_state=new_sam2_state,
                frame_idx=frame_idx,
                obj_ids=new_obj_ids,
                masks=new_obj_masks,
                add_mask_to_memory=True,
            )
        else:
            for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
                self.tracker.add_new_mask(
                    inference_state=new_sam2_state,
                    frame_idx=frame_idx,
                    obj_id=new_obj_id,
                    mask=new_mask,
                    add_mask_to_memory=True,
                )
        self.tracker.propagate_in_video_preflight(new_sam2_state, run_mem_encoder=True)

        return tracker_states_local

    def _tracker_remove_objects(
        self, tracker_states_local: List[Any], obj_ids: list[int]
    ):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        if self.is_multiplex:
            tracker_states_local_before_removal = tracker_states_local.copy()
            tracker_states_local.clear()
            for sam2_inference_state in tracker_states_local_before_removal:
                new_obj_ids, _ = self.tracker.remove_objects(
                    sam2_inference_state, obj_ids, strict=False, need_output=False
                )
                if len(new_obj_ids) > 0:
                    tracker_states_local.append(sam2_inference_state)
        else:
            for obj_id in obj_ids:
                self._tracker_remove_object(tracker_states_local, obj_id)

    def update_masklet_confirmation_status(
        self,
        rank0_metadata: Dict[str, Any],
        obj_ids_all_gpu_prev: np.ndarray,
        obj_ids_all_gpu_updated: np.ndarray,
        det_to_matched_trk_obj_ids: Dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
    ):
        """
        Update masklet confirmation status.
        """
        confirmation_data = rank0_metadata["masklet_confirmation"]
        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]

        N_prev = len(obj_ids_all_gpu_prev)
        N_updated = len(obj_ids_all_gpu_updated)

        unconfirmed_val = MaskletConfirmationStatus.UNCONFIRMED.value
        status = np.full(N_updated, unconfirmed_val, dtype=np.int64)
        consecutive_det_num = np.zeros(N_updated, dtype=np.int64)

        if N_prev > 0 and N_updated > 0:
            obj_id_to_new_idx = {
                obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)
            }

            for old_idx, obj_id in enumerate(obj_ids_all_gpu_prev):
                new_idx = obj_id_to_new_idx.get(obj_id)
                if new_idx is not None:
                    status[new_idx] = status_prev[old_idx]
                    consecutive_det_num[new_idx] = consecutive_det_num_prev[old_idx]

        matched_obj_ids = set(new_det_obj_ids)
        for matched_trk_ids in det_to_matched_trk_obj_ids.values():
            matched_obj_ids.update(matched_trk_ids)

        for idx, obj_id in enumerate(obj_ids_all_gpu_updated):
            if obj_id in matched_obj_ids:
                consecutive_det_num[idx] += 1
            else:
                consecutive_det_num[idx] = 0

            if (
                consecutive_det_num[idx]
                >= self.masklet_confirmation_consecutive_det_thresh
            ):
                status[idx] = MaskletConfirmationStatus.CONFIRMED.value

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return rank0_metadata

class Sam3MultiplexPredictorWrapper(Sam3MultiplexTrackerPredictor):
    """
    Wraps a pre-built multiplex tracker model with the same interface as the
    onevision Sam3MultiplexTrackerPredictor class. Inherits from Sam3MultiplexTrackerPredictor to pass
    isinstance checks, but skips Sam3MultiplexTrackerPredictor.__init__ (which requires Hydra).

    Provides bf16 autocast, attribute proxying, and configuration flags
    needed by Sam3MultiplexTracking.

    The onevision Sam3MultiplexTrackerPredictor builds the tracker from Hydra config and applies
    extensive hydra_overrides. This version skips Hydra entirely — the caller
    is responsible for building the tracker via model_builder.py with the
    correct parameters.

    Key parameters that the onevision Sam3MultiplexTrackerPredictor sets via hydra_overrides
    (documented here for reference — these must be set in model_builder.py):
      - image_size=1008, backbone_stride=14
      - maskmem_backbone.mask_downsampler.interpol_size=[1152,1152]
      - always_start_from_first_ann_frame=false
      - non_overlap_masks_for_mem_enc=false, non_overlap_masks_for_output=false
      - max_cond_frames_in_attn=4
      - offload_output_to_cpu_for_eval=false, trim_past_non_cond_mem_for_eval=false
      - sam_mask_decoder_extra_args: dynamic_multimask_via_stability=true, etc.
      - binarize_mask_from_pts_for_mem_enc=true (SAM2 tracker default)
      - only_obj_ptrs_in_the_past_for_eval=true
      - clear_non_cond_mem_around_input=true
      - transformer.encoder.layer.self_attention.feat_sizes=[72,72]
      - transformer.encoder.layer.cross_attention.feat_sizes=[72,72]
      - fill_hole_area=<fill_hole_area>
      - use_fa3, use_rope_real on self_attention, cross_attention,
        self_attention_rope, cross_attention_rope
      - use_memory_selection
    """

    def __init__(
        self,
        model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
        num_obj_for_compile=1,
        max_num_objects = 1,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.per_obj_inference = per_obj_inference
        self.fill_hole_area = fill_hole_area
        self.is_multiplex = is_multiplex
        self.is_multiplex_dynamic = is_multiplex_dynamic
        self.num_obj_for_compile = num_obj_for_compile
        self.max_num_objects = max_num_objects

        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.bf16_context.__enter__()
