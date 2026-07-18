"""One-click launcher for the RF-DETR segmentation baseline.

Edit only the USER SETTINGS block, then run this file directly.
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

# Server retraining default: train, then evaluate the test split.
# Other choices: "prepare", "smoke", "train", "test".
RUN_MODE = "all"

BASELINE = "rfdetr_seg_nano"

# Windows paths
WINDOWS_SOURCE_DATASET = Path(r"E:\mastercode\data\orange_yolo")
WINDOWS_PREPARED_DATASET = Path(r"E:\mastercode\4_baseline_choice\datasets\citrus_prepared")
WINDOWS_OUTPUT_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\rfdetr")
WINDOWS_EVALUATION_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\evaluation")

# Linux server paths. The source dataset is read-only; derived COCO data
# and outputs are written under results/002_retrain.
SERVER_SOURCE_DATASET = Path("/data/sxq/datasets/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/sxq/results/002_retrain/_prepared/citrus_prepared")
SERVER_OUTPUT_ROOT = Path("/data/sxq/results/002_retrain/rfdetr")
SERVER_EVALUATION_ROOT = Path("/data/sxq/results/002_retrain/evaluation")

# Formal experiment settings
FORMAL_EPOCHS = 300
SEED = 42
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
WORKERS = 8
DEVICE = "cuda"
LEARNING_RATE = 0.0001
LEARNING_RATE_ENCODER = 0.00015
WEIGHT_DECAY = 0.0001
REQUIRE_CUDA = True

# =============================================================================
# END USER SETTINGS
# =============================================================================


SUITE_ROOT = Path(__file__).resolve().parent
SCRIPTS = SUITE_ROOT / "scripts"
VALID_MODES = {"prepare", "smoke", "train", "test", "all"}


def platform_paths() -> dict[str, Path]:
    """Select Windows or Linux paths automatically."""
    if os.name == "nt":
        return {
            "source": WINDOWS_SOURCE_DATASET,
            "prepared": WINDOWS_PREPARED_DATASET,
            "output": WINDOWS_OUTPUT_ROOT,
            "evaluation": WINDOWS_EVALUATION_ROOT,
        }
    return {
        "source": SERVER_SOURCE_DATASET,
        "prepared": SERVER_PREPARED_DATASET,
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


def check_environment() -> None:
    """Check RF-DETR, PyTorch, and CUDA before training."""
    try:
        import rfdetr
        import torch
    except ImportError as exc:
        raise RuntimeError("The RF-DETR environment is incomplete. Install rfdetr and torch first.") from exc
    print("\nEnvironment")
    print("-" * 80)
    print(f"Python executable : {sys.executable}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"RF-DETR           : {getattr(rfdetr, '__version__', 'unknown')}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=True, but CUDA is unavailable.")
    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")


def prepared_dataset_exists(path: Path) -> bool:
    """Check whether RF-DETR and COCO converted splits exist."""
    return all(
        (path / "rfdetr" / split_dir / "_annotations.coco.json").is_file()
        for split_dir in ("train", "valid", "test")
    ) and all((path / "coco" / "annotations" / f"instances_{split}.json").is_file() for split in ("train", "val", "test"))


def prepare_dataset(paths: dict[str, Path]) -> None:
    """Convert source YOLO polygons once if the prepared dataset is absent."""
    if prepared_dataset_exists(paths["prepared"]):
        print(f"Prepared dataset already exists: {paths['prepared']}")
        return
    if not paths["source"].is_dir():
        raise FileNotFoundError(f"Source YOLO dataset not found: {paths['source']}")
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


def run_name(smoke: bool) -> str:
    """Create a stable experiment name."""
    prefix = "SMOKE_rfdetr" if smoke else "E_rfdetr"
    return f"{prefix}_{BASELINE}_seed{SEED}"


def find_checkpoint(run_dir: Path) -> Path | None:
    """Find the preferred RF-DETR checkpoint."""
    for name in ("checkpoint_best_total.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
        path = run_dir / name
        if path.is_file():
            return path
    candidates = sorted(run_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
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


def train_model(paths: dict[str, Path], smoke: bool) -> Path:
    """Train RF-DETR, or reuse an existing checkpoint."""
    name = run_name(smoke)
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
    checkpoint_interval = 1 if smoke else 10
    run_python(
        "train_rfdetr.py",
        "--baseline",
        BASELINE,
        "--dataset",
        paths["prepared"],
        "--name",
        name,
        "--output-root",
        paths["output"],
        "--epochs",
        epochs,
        "--batch",
        batch,
        "--grad-accum-steps",
        GRAD_ACCUM_STEPS,
        "--workers",
        workers,
        "--device",
        DEVICE,
        "--seed",
        SEED,
        "--lr",
        LEARNING_RATE,
        "--lr-encoder",
        LEARNING_RATE_ENCODER,
        "--weight-decay",
        WEIGHT_DECAY,
        "--checkpoint-interval",
        checkpoint_interval,
    )
    checkpoint = find_checkpoint(run_dir)
    if not checkpoint:
        raise FileNotFoundError(f"Training finished but no checkpoint was found in {run_dir}")
    return checkpoint


def evaluate_model(paths: dict[str, Path], weights: Path, smoke: bool, split: str) -> None:
    """Evaluate RF-DETR with the shared COCO mask metrics."""
    name = run_name(smoke)
    output = paths["evaluation"] / f"{name}_{split}"
    metrics = output / "metrics.json"
    if metrics.is_file():
        print(f"Evaluation already complete: {output}")
        print_metrics(metrics, f"{BASELINE} {split} metrics")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation directory exists but metrics.json is absent: {output}")
    run_python(
        "eval_rfdetr.py",
        "--baseline",
        BASELINE,
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
        "--batch",
        1,
    )
    print_metrics(metrics, f"{BASELINE} {split} metrics")


def main() -> None:
    """Run the selected one-click workflow."""
    if RUN_MODE not in VALID_MODES:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}. Choices: {sorted(VALID_MODES)}")
    paths = platform_paths()
    print(f"Suite root: {SUITE_ROOT}")
    print(f"Mode      : {RUN_MODE}")
    print(f"Baseline  : {BASELINE}")

    prepare_dataset(paths)
    if RUN_MODE == "prepare":
        return
    check_environment()

    if RUN_MODE in {"smoke", "train", "all"}:
        smoke = RUN_MODE == "smoke"
        weights = train_model(paths, smoke=smoke)
        if smoke:
            evaluate_model(paths, weights, smoke=True, split="val")
        elif RUN_MODE == "all":
            evaluate_model(paths, weights, smoke=False, split="test")
        return

    if RUN_MODE == "test":
        weights = find_checkpoint(paths["output"] / run_name(smoke=False))
        if not weights:
            raise FileNotFoundError(f"No formal checkpoint found: {paths['output'] / run_name(smoke=False)}")
        evaluate_model(paths, weights, smoke=False, split="test")


if __name__ == "__main__":
    main()
