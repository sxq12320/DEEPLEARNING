"""One-click launcher for YOLO segmentation baselines.

Edit only the USER SETTINGS block, then run this file directly. It uses the
shared prepared dataset and shared COCO mask evaluator.
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

# Server retraining default: train every model, then evaluate the test split.
# Other choices: "prepare", "smoke", "train", "test".
RUN_MODE = "all"

BASELINES = [
    "yolov8n_seg",
    "yolo11n_seg",
    "yolo12n_seg",
    "yolo26n_seg",
]

# Windows paths
WINDOWS_SOURCE_DATASET = Path(r"E:\mastercode\data\orange_yolo")
WINDOWS_PREPARED_DATASET = Path(r"E:\mastercode\4_baseline_choice\datasets\citrus_prepared")
WINDOWS_OUTPUT_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\yolo")
WINDOWS_EVALUATION_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\evaluation")

# Linux server paths. The source dataset is read-only; derived formats and
# outputs are written under results/002_retrain.
SERVER_SOURCE_DATASET = Path("/data/sxq/datasets/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/sxq/results/002_retrain/_prepared/citrus_prepared")
SERVER_OUTPUT_ROOT = Path("/data/sxq/results/002_retrain/yolo")
SERVER_EVALUATION_ROOT = Path("/data/sxq/results/002_retrain/evaluation")

# Formal experiment settings
FORMAL_EPOCHS = 300
SEED = 42
BATCH_SIZE = 4
IMAGE_SIZE = 640
WORKERS = 8
DEVICE = "0"
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
REQUIRE_CUDA = True

# =============================================================================
# END USER SETTINGS
# =============================================================================


SUITE_ROOT = Path(__file__).resolve().parent
SCRIPTS = SUITE_ROOT / "scripts"
VALID_MODES = {"prepare", "smoke", "train", "test", "all"}
YOLO_BASELINES = {"yolov8n_seg", "yolo11n_seg", "yolo12n_seg", "yolo26n_seg", "yolo11s_seg"}


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
    """Check Ultralytics and CUDA before running YOLO baselines."""
    try:
        import torch
        import ultralytics
    except ImportError as exc:
        raise RuntimeError("The YOLO environment is incomplete. Install ultralytics and torch first.") from exc
    print("\nEnvironment")
    print("-" * 80)
    print(f"Python executable : {sys.executable}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"Ultralytics       : {ultralytics.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=True, but CUDA is unavailable.")
    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")


def prepared_dataset_exists(path: Path) -> bool:
    """Check whether YOLO and COCO converted splits exist."""
    return all(
        (path / "yolo" / "images" / split).is_dir()
        and (path / "yolo" / "labels" / split).is_dir()
        and (path / "coco" / "annotations" / f"instances_{split}.json").is_file()
        for split in ("train", "val", "test")
    )


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


def run_name(baseline_id: str, smoke: bool) -> str:
    """Create a stable experiment name."""
    prefix = "SMOKE_yolo" if smoke else "E_yolo"
    return f"{prefix}_{baseline_id}_seed{SEED}"


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
    if "latency_ms_per_image_end_to_end" in metrics:
        print(f"{'latency_ms_per_image':22s}: {float(metrics['latency_ms_per_image_end_to_end']):7.2f}")


def train_model(paths: dict[str, Path], baseline_id: str, smoke: bool) -> Path:
    """Train one YOLO baseline, or reuse an existing best.pt."""
    name = run_name(baseline_id, smoke)
    run_dir = paths["output"] / name
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.is_file():
        print(f"Training already has best weights: {best_weights}")
        return best_weights
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory exists but best.pt is absent: {run_dir}")

    epochs = 2 if smoke else FORMAL_EPOCHS
    batch = 2 if smoke else BATCH_SIZE
    workers = 2 if smoke else WORKERS
    run_python(
        "train_yolo.py",
        "--baseline",
        baseline_id,
        "--dataset",
        paths["prepared"],
        "--name",
        name,
        "--output-root",
        paths["output"],
        "--epochs",
        epochs,
        "--imgsz",
        IMAGE_SIZE,
        "--batch",
        batch,
        "--device",
        DEVICE,
        "--workers",
        workers,
        "--optimizer",
        OPTIMIZER,
        "--lr0",
        LEARNING_RATE,
        "--weight-decay",
        WEIGHT_DECAY,
        "--seed",
        SEED,
    )
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_weights}")
    return best_weights


def evaluate_model(paths: dict[str, Path], baseline_id: str, weights: Path, smoke: bool, split: str) -> None:
    """Evaluate one YOLO baseline with the shared COCO mask metrics."""
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
        "eval_yolo.py",
        "--weights",
        weights,
        "--dataset",
        paths["prepared"],
        "--split",
        split,
        "--output",
        output,
        "--imgsz",
        IMAGE_SIZE,
        "--device",
        DEVICE,
        "--batch",
        1,
        "--workers",
        WORKERS,
    )
    print_metrics(metrics, f"{baseline_id} {split} metrics")


def normalize_baselines() -> list[str]:
    """Validate requested YOLO baseline IDs."""
    baseline_ids = [str(value) for value in BASELINES]
    unknown = sorted(set(baseline_ids) - YOLO_BASELINES)
    if unknown:
        raise ValueError(f"Unsupported YOLO baselines: {unknown}. Choices: {sorted(YOLO_BASELINES)}")
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
    check_environment()

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
            weights = paths["output"] / run_name(baseline_id, smoke=False) / "weights" / "best.pt"
            if not weights.is_file():
                raise FileNotFoundError(f"No formal best.pt found for {baseline_id}: {weights}")
            evaluate_model(paths, baseline_id, weights, smoke=False, split="test")


if __name__ == "__main__":
    main()
