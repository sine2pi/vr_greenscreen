import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

def _norm(path_value: str) -> str:
    return str(Path(path_value).expanduser().resolve())

def _assert_exists(path_value: str, what: str) -> None:
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{what} not found: {path_value}")

def _check_dataset_layout(dataset_root: str, supercategory: str) -> None:
    base = Path(dataset_root) / supercategory
    required = [
        base / "train",
        base / "test",
        base / "train" / "_annotations.coco.json",
        base / "test" / "_annotations.coco.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset layout check failed. Expected Roboflow-style COCO layout:\n"
            f"  {base}/train/_annotations.coco.json\n"
            f"  {base}/test/_annotations.coco.json\n"
            f"Missing:\n- " + "\n- ".join(missing)
        )

def _yaml_path(path_value: str) -> str:
    return Path(path_value).expanduser().resolve().as_posix()

def _run_preflight(repo_root: Path, dataset_root: str, supercategory: str, strict: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/check_vr_sbs_coco.py",
        "--dataset-root",
        _norm(dataset_root),
        "--supercategory",
        supercategory,
    ]
    if strict:
        cmd.append("--strict")

    print("Running dataset preflight...")
    rc = subprocess.call(cmd, cwd=str(repo_root))
    if rc != 0:
        raise RuntimeError(
            f"Dataset preflight failed with exit code {rc}. "
            "Fix dataset issues or re-run with --skip-preflight."
        )

def _write_runtime_config(repo_root: Path, args: argparse.Namespace) -> Path:
    cfg_dir = repo_root / "sam3" / "train" / "configs" / "custom"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    fname = f"_tmp_vr_sbs_runtime_{os.getpid()}_{int(time.time())}.yaml"
    cfg_path = cfg_dir / fname

    distributed_lines = []
    if platform.system().lower().startswith("win"):
        distributed_lines = [
            "  distributed:",
            "    backend: gloo",
        ]

    collate_override_lines = []
    if args.grad_accum > 1:
        collate_override_lines = [
            "  collate_fn:",
            "    _target_: sam3.train.data.collator.collate_fn_api_with_chunking",
            "    _partial_: true",
            f"    num_chunks: {args.grad_accum}",
            "    repeats: ${scratch.hybrid_repeats}",
            "    dict_key: all",
            "    with_seg_masks: ${scratch.enable_segmentation}",
        ]

    content_lines = [
        "# @package _global_",
        "defaults:",
        "  - /configs/custom/vr_sbs_local_ft.yaml",
        "  - _self_",
        "",
        "paths:",
        f"  roboflow_vl_100_root: {_yaml_path(args.dataset_root)}",
        f"  experiment_log_dir: {_yaml_path(args.log_dir)}",
        f"  bpe_path: {_yaml_path(args.bpe_path)}",
        "",
        "roboflow_train:",
        f"  supercategory: {args.supercategory}",
        "  num_images: null",
        "",
        "submitit:",
        "  use_cluster: false",
        "  job_array:",
        "    num_tasks: 0",
        "    task_index: 0",
        "",
        "launcher:",
        "  num_nodes: 1",
        f"  gpus_per_node: {args.num_gpus}",
        "",
        "trainer:",
        f"  max_epochs: {args.max_epochs}",
        f"  val_epoch_freq: {args.val_epoch_freq}",
        "  skip_saving_ckpts: false",
        *distributed_lines,
        "",
        "checkpoint:",
        f"  save_freq: {args.save_freq}",
        *(([f"  resume_from: {_yaml_path(args.resume_from)}"] if args.resume_from else [])),
        "",
        "scratch:",
        f"  resolution: {args.resolution}",
        f"  train_batch_size: {args.train_batch_size}",
        f"  val_batch_size: {args.val_batch_size}",
        f"  gradient_accumulation_steps: {args.grad_accum}",
        f"  num_train_workers: {args.num_workers}",
        f"  num_val_workers: {args.num_val_workers}",
        "  enable_segmentation: true",
        *collate_override_lines,
        *(([f"  lr_scale: {args.override_lr_scale}"] if args.override_lr_scale is not None else [])),
    ]

    cfg_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
    return cfg_path

def build_command(runtime_config_relpath: str, args: argparse.Namespace) -> list[str]:
    python_exe = sys.executable
    return [
        python_exe,
        "-m",
        "sam3.train.train",
        "-c",
        runtime_config_relpath,
        "--use-cluster",
        "0",
        "--num-gpus",
        str(args.num_gpus),
        "--num-nodes",
        "1",
    ]

def main() -> int:
    parser = argparse.ArgumentParser(description="Train SAM3 or 3.1 for VR-SBS masking (local, non-cluster).")
    parser.add_argument("--dataset-root", default="out", help="Root containing <supercategory>/train|test with COCO jsons")
    parser.add_argument("--supercategory", default="vr_sbs", help="Dataset subfolder name under dataset root")
    parser.add_argument("--log-dir", default="logs", help="Training output directory")
    parser.add_argument("--bpe-path", default="bpe_simple_vocab_16e6.txt.gz", help="Path to SAM3 BPE vocab file")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--val-epoch-freq", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-val-workers", type=int, default=0)
    parser.add_argument("--override-lr-scale", type=float, default=None, help="Optional override for scratch.lr_scale")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path to resume")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip dataset preflight checks")
    parser.add_argument("--strict-preflight", action="store_true", help="Fail when preflight warnings are present")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    _assert_exists(args.bpe_path, "BPE file")
    _check_dataset_layout(args.dataset_root, args.supercategory)
    if args.resume_from:
        _assert_exists(args.resume_from, "Resume checkpoint")

    repo_root = Path(__file__).resolve().parents[1]

    if not args.skip_preflight:
        _run_preflight(
            repo_root=repo_root,
            dataset_root=args.dataset_root,
            supercategory=args.supercategory,
            strict=args.strict_preflight,
        )

    runtime_cfg_path = _write_runtime_config(repo_root, args)
    runtime_cfg_rel = str(runtime_cfg_path.relative_to(repo_root / "sam3" / "train")).replace("\\", "/")
    cmd = build_command(runtime_cfg_rel, args)

    print("=" * 80)
    print("SAM3.1 VR-SBS training launcher")
    print("Working dir:", str(repo_root))
    print("Runtime config:", runtime_cfg_rel)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    if args.dry_run:
        return 0

    try:
        proc = subprocess.Popen(cmd, cwd=str(repo_root))
        proc.wait()
        return proc.returncode
    finally:
        if runtime_cfg_path.exists():
            runtime_cfg_path.unlink()

if __name__ == "__main__":
    raise SystemExit(main())
