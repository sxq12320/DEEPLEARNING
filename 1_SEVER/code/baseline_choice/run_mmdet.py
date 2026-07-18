"""One-click launcher for MMDetection citrus instance-segmentation baselines.

This covers RTMDet-Ins-tiny, SOLOv2-Light R18-FPN, and MMDetection Mask R-CNN
R50-FPN. Edit only the USER SETTINGS block, then run this file directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


# =============================================================================
# USER SETTINGS: edit this block only
# =============================================================================

# Server retraining default: train every selected model, then evaluate test split.
# Other choices: "prepare", "smoke", "train", "test".
RUN_MODE = "all"

# Run all MMDetection baselines by default. Keep one item here if you only want one model.
BASELINES = [
    "rtmdet_ins_tiny",
    "solov2_light",
    "mask_rcnn_r50",
]

# Windows paths
WINDOWS_SOURCE_DATASET = Path(r"E:\mastercode\data\orange_yolo")
WINDOWS_PREPARED_DATASET = Path(r"E:\mastercode\4_baseline_choice\datasets\citrus_prepared")
WINDOWS_MMDET_ROOT = Path(r"E:\mastercode\4_baseline_choice\third_party\mmdetection")
WINDOWS_OUTPUT_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\mmdet")
WINDOWS_EVALUATION_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\evaluation")

# Linux server paths. The source dataset is read-only; derived COCO data
# and outputs are written under results/002_retrain.
SERVER_SOURCE_DATASET = Path("/data/sxq/datasets/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/sxq/datasets/citrus_prepared")
SERVER_MMDET_ROOT = Path("/data/sxq/code/mmdetection")
SERVER_OUTPUT_ROOT = Path("/data/sxq/results/002_retrain/mmdet")
SERVER_EVALUATION_ROOT = Path("/data/sxq/results/002_retrain/evaluation")



# Formal experiment settings
FORMAL_EPOCHS = 300
SEED = 42
BATCH_SIZE = 16
WORKERS = 4
DEVICE = "cuda:1"
VAL_INTERVAL = 5
REQUIRE_CUDA = True

# Optional pretrained checkpoints. Leave None to use the official config default init.
CHECKPOINTS = {
    "rtmdet_ins_tiny": None,
    "solov2_light": None,
    "mask_rcnn_r50": None,
}

# =============================================================================
# END USER SETTINGS
# =============================================================================


SUITE_ROOT = Path(__file__).resolve().parent
SCRIPTS = SUITE_ROOT / "scripts"
VALID_MODES = {"prepare", "smoke", "train", "test", "all"}
MMDET_BASELINES = {"rtmdet_ins_tiny", "solov2_light", "mask_rcnn_r50"}


def platform_paths() -> dict[str, Path]:
    """Select Windows or Linux paths automatically."""
    if os.name == "nt":
        return {
            "source": WINDOWS_SOURCE_DATASET,
            "prepared": WINDOWS_PREPARED_DATASET,
            "mmdet_root": WINDOWS_MMDET_ROOT,
            "output": WINDOWS_OUTPUT_ROOT,
            "evaluation": WINDOWS_EVALUATION_ROOT,
        }
    return {
        "source": SERVER_SOURCE_DATASET,
        "prepared": SERVER_PREPARED_DATASET,
        "mmdet_root": SERVER_MMDET_ROOT,
        "output": SERVER_OUTPUT_ROOT,
        "evaluation": SERVER_EVALUATION_ROOT,
    }


def run_python(script_name: str, *arguments: object) -> None:
    """Run a worker script with the active Python environment."""
    command = [sys.executable, str(SCRIPTS / script_name)]
    command.extend(str(value) for value in arguments)
    print("\n" + "=" * 80)
    print("Running:", subprocess.list2cmdline(command))
    print("=" * 80)
    subprocess.run(command, cwd=SUITE_ROOT, check=True)


def registry_entry(baseline_id: str) -> dict[str, object]:
    """Load one MMDetection baseline from configs/baselines.yaml."""
    sys.path.insert(0, str(SCRIPTS))
    from baseline_common import get_baseline

    return get_baseline(baseline_id, family="mmdetection")


def check_environment(paths: dict[str, Path], baseline_ids: list[str]) -> None:
    """Check MMDetection imports, CUDA, and required official config files."""
    try:
        import mmcv
        import mmdet
        import mmengine
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The MMDetection environment is incomplete. Install torch, mmcv, "
            "mmengine, and mmdet before running this launcher."
        ) from exc

    print("\nEnvironment")
    print("-" * 80)
    print(f"Python executable : {sys.executable}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"MMCV              : {mmcv.__version__}")
    print(f"MMEngine          : {mmengine.__version__}")
    print(f"MMDetection       : {mmdet.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=True, but CUDA is unavailable.")
    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")

    for baseline_id in baseline_ids:
        config = paths["mmdet_root"] / str(registry_entry(baseline_id)["config"])
        if not config.is_file():
            raise FileNotFoundError(
                f"Missing official MMDetection config: {config}\n"
                "Clone MMDetection v3.x to MMDET_ROOT, or edit WINDOWS_MMDET_ROOT/SERVER_MMDET_ROOT."
            )


def prepared_dataset_exists(path: Path) -> bool:
    """Check whether all converted COCO splits exist."""
    annotation_dir = path / "coco" / "annotations"
    return all((annotation_dir / f"instances_{split}.json").is_file() for split in ("train", "val", "test"))


def prepare_dataset(paths: dict[str, Path]) -> None:
    """Convert YOLO polygons once and validate the prepared COCO dataset."""
    if not prepared_dataset_exists(paths["prepared"]):
        if not paths["source"].is_dir():
            raise FileNotFoundError(
                f"Source YOLO dataset not found: {paths['source']}\n"
                "Edit SOURCE_DATASET in the USER SETTINGS block."
            )
        run_python(
            "prepare_dataset.py",
            "--source",
            paths["source"],
            "--output",
            paths["prepared"],
            "--class-name",
            "orange_immature",
            "--mode",
            "auto",
        )
    else:
        print(f"Prepared dataset already exists: {paths['prepared']}")


def run_name(baseline_id: str, smoke: bool) -> str:
    """Create a stable experiment name."""
    prefix = "SMOKE_mmdet" if smoke else "E_mmdet"
    return f"{prefix}_{baseline_id}_seed{SEED}"


def find_checkpoint(run_dir: Path) -> Path | None:
    """Find the best or latest MMDetection checkpoint in one run directory."""
    candidates = sorted(run_dir.glob("best_*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    last_file = run_dir / "last_checkpoint"
    if last_file.is_file():
        value = last_file.read_text(encoding="utf-8").strip()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.is_file():
            return candidate
    candidates = sorted(run_dir.glob("epoch_*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def print_metrics(path: Path, title: str) -> None:
    """Print paper-table metrics when an evaluation file exists."""
    if not path.is_file():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print("\n" + title)
    print("-" * len(title))
    for key in ("mask_ap_50_95", "mask_ap_50", "mask_ap_75", "mask_ap_small", "mask_ap_medium", "mask_ap_large"):
        if key in metrics:
            print(f"{key:22s}: {float(metrics[key]) * 100:7.2f}%")
    for key in ("params_m", "latency_ms_per_image_end_to_end"):
        if key in metrics:
            print(f"{key:22s}: {float(metrics[key]):7.2f}")


def train_model(paths: dict[str, Path], baseline_id: str, smoke: bool) -> Path:
    """Train one MMDetection baseline, or reuse its existing checkpoint."""
    name = run_name(baseline_id, smoke)
    run_dir = paths["output"] / name
    checkpoint = find_checkpoint(run_dir)
    if checkpoint:
        print(f"Training already has a checkpoint: {checkpoint}")
        return checkpoint
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory exists but no checkpoint was found: {run_dir}")

    epochs = 2 if smoke else FORMAL_EPOCHS
    batch = 1 if smoke else BATCH_SIZE
    workers = 2 if smoke else WORKERS
    val_interval = 1 if smoke else VAL_INTERVAL
    arguments: list[object] = [
        "--baseline",
        baseline_id,
        "--dataset",
        paths["prepared"],
        "--name",
        name,
        "--mmdet-root",
        paths["mmdet_root"],
        "--output-root",
        paths["output"],
        "--epochs",
        epochs,
        "--batch",
        batch,
        "--workers",
        workers,
        "--seed",
        SEED,
        "--val-interval",
        val_interval,
    ]
    checkpoint_override = CHECKPOINTS.get(baseline_id)
    if checkpoint_override:
        arguments.extend(["--checkpoint", checkpoint_override])
    run_python("train_mmdet.py", *arguments)
    checkpoint = find_checkpoint(run_dir)
    if not checkpoint:
        raise FileNotFoundError(f"Training finished but no checkpoint was found in {run_dir}")
    return checkpoint


def evaluate_model(paths: dict[str, Path], baseline_id: str, weights: Path, smoke: bool, split: str) -> None:
    """Evaluate one trained MMDetection baseline with shared COCO mask metrics."""
    name = run_name(baseline_id, smoke)
    output = paths["evaluation"] / f"{name}_{split}"
    metrics = output / "metrics.json"
    if metrics.is_file():
        print(f"Evaluation already complete: {output}")
        print_metrics(metrics, f"{baseline_id} {split} metrics")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation directory exists but metrics.json is absent: {output}")
    run_python(
        "eval_mmdet.py",
        "--weights",
        weights,
        "--dataset",
        paths["prepared"],
        "--split",
        split,
        "--output",
        output,
        "--device",
        DEVICE,
    )
    print_metrics(metrics, f"{baseline_id} {split} metrics")


def normalize_baselines() -> list[str]:
    """Validate requested baseline IDs."""
    baseline_ids = [str(value) for value in BASELINES]
    unknown = sorted(set(baseline_ids) - MMDET_BASELINES)
    if unknown:
        raise ValueError(f"Unsupported MMDetection baselines: {unknown}. Choices: {sorted(MMDET_BASELINES)}")
    return baseline_ids


def main() -> None:
    """Run the selected one-click workflow."""
    if RUN_MODE not in VALID_MODES:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}. Choices: {sorted(VALID_MODES)}")
    baseline_ids = normalize_baselines()
    paths = platform_paths()
    print(f"Suite root: {SUITE_ROOT}")
    print(f"Mode      : {RUN_MODE}")
    print(f"Baselines : {', '.join(baseline_ids)}")

    prepare_dataset(paths)
    if RUN_MODE == "prepare":
        return
    check_environment(paths, baseline_ids)

    if RUN_MODE in {"smoke", "train", "all"}:
        smoke = RUN_MODE == "smoke"
        for baseline_id in baseline_ids:
            weights = train_model(paths, baseline_id, smoke=smoke)
            if smoke:
                evaluate_model(paths, baseline_id, weights, smoke=True, split="val")
            elif RUN_MODE == "all":
                evaluate_model(paths, baseline_id, weights, smoke=False, split="test")
        return

    if RUN_MODE == "test":
        for baseline_id in baseline_ids:
            name = run_name(baseline_id, smoke=False)
            run_dir = paths["output"] / name
            weights = find_checkpoint(run_dir)
            if not weights:
                raise FileNotFoundError(f"No formal checkpoint found for {baseline_id}: {run_dir}")
            evaluate_model(paths, baseline_id, weights, smoke=False, split="test")


if __name__ == "__main__":
    main()
