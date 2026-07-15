"""One-click launcher for the citrus U-Net + watershed instance baseline.

Edit only the USER SETTINGS block, then run this file directly from an IDE.
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

# Windows paths
WINDOWS_SOURCE_DATASET = Path(r"E:\mastercode\data\orange_yolo")
WINDOWS_PREPARED_DATASET = Path(
    r"E:\mastercode\4_baseline_choice\datasets\citrus_prepared"
)
WINDOWS_OUTPUT_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\unet_watershed")
WINDOWS_EVALUATION_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\evaluation")

# Linux server paths. The source dataset is read-only; converted semantic data
# and outputs are written under results/002_retrain.
SERVER_SOURCE_DATASET = Path("/data/sxq/datasets/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/sxq/results/002_retrain/_prepared/citrus_prepared")
SERVER_OUTPUT_ROOT = Path("/data/sxq/results/002_retrain/unet_watershed")
SERVER_EVALUATION_ROOT = Path("/data/sxq/results/002_retrain/evaluation")

# Formal experiment parameters
FORMAL_NAME = "E_unet_r18_watershed_seed42"
FORMAL_EPOCHS = 300
ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
IMAGE_SIZE = 640
WORKERS = 8
DEVICE = "cuda:0"
REQUIRE_CUDA = True
USE_AMP = True

# Semantic-to-instance parameters. Keep fixed across formal comparisons.
PROBABILITY_THRESHOLD = 0.50
WATERSHED_MIN_DISTANCE = 8
WATERSHED_MIN_AREA = 20
MAX_INSTANCES = 50

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
    """Print versions and stop early when U-Net dependencies are incomplete."""
    try:
        import pycocotools  # noqa: F401
        import scipy
        import segmentation_models_pytorch as smp
        import skimage
        import timm
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "The U-Net environment is incomplete. Run: "
            "pip install -r requirements-unet.txt"
        ) from exc
    print("\nEnvironment")
    print("-" * 80)
    print(f"Python executable            : {sys.executable}")
    print(f"PyTorch                     : {torch.__version__}")
    print(f"Torchvision                 : {torchvision.__version__}")
    print(f"segmentation_models_pytorch : {smp.__version__}")
    print(f"timm                        : {timm.__version__}")
    print(f"scikit-image                : {skimage.__version__}")
    print(f"SciPy                       : {scipy.__version__}")
    print(f"CUDA available              : {torch.cuda.is_available()}")
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "REQUIRE_CUDA=True, but CUDA is unavailable. Fix the server environment "
            "or temporarily set REQUIRE_CUDA=False for a CPU smoke test."
        )
    if torch.cuda.is_available():
        print(f"GPU                         : {torch.cuda.get_device_name(0)}")


def prepared_dataset_exists(path: Path) -> bool:
    """Check semantic pairs and COCO annotations for all splits."""
    return all(
        (path / "semantic" / "images" / split).is_dir()
        and (path / "semantic" / "masks" / split).is_dir()
        and (path / "coco" / "annotations" / f"instances_{split}.json").is_file()
        for split in ("train", "val", "test")
    )


def prepare_dataset(paths: dict[str, Path]) -> None:
    """Convert YOLO polygons once and validate U-Net inputs."""
    if not prepared_dataset_exists(paths["prepared"]):
        if not paths["source"].is_dir():
            raise FileNotFoundError(
                f"Source YOLO dataset not found: {paths['source']}\n"
                "Edit the matching SOURCE_DATASET path in USER SETTINGS."
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
        print(f"Prepared semantic/COCO dataset exists: {paths['prepared']}")
    sys.path.insert(0, str(SCRIPTS))
    from unet_common import validate_semantic_dataset

    reports = validate_semantic_dataset(paths["prepared"], ("train", "val", "test"))
    print("\nDataset")
    print("-" * 80)
    for split, report in reports.items():
        print(
            f"{split:5s}: images={report['images']:4d}, "
            f"instances={report['instances']:4d}"
        )


def completed_epochs(run_dir: Path) -> int:
    """Return the final epoch in an existing history file."""
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return 0
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return int(history[-1]["epoch"]) if history else 0


def print_settings(mode: str, paths: dict[str, Path], smoke: bool = False) -> None:
    """Print one-click workflow paths and effective hyperparameters."""
    settings = {
        "mode": mode,
        **paths,
        "experiment": "UNET_R18_WATERSHED_smoke" if smoke else FORMAL_NAME,
        "epochs": 2 if smoke else FORMAL_EPOCHS,
        "encoder": ENCODER,
        "encoder_weights": ENCODER_WEIGHTS,
        "image_size": IMAGE_SIZE,
        "batch_size": 2 if smoke else BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "watershed_min_distance": WATERSHED_MIN_DISTANCE,
        "watershed_min_area": WATERSHED_MIN_AREA,
        "max_instances": MAX_INSTANCES,
        "seed": SEED,
        "device": DEVICE,
        "amp": USE_AMP,
    }
    print("\n" + "=" * 80)
    print("U-Net + Watershed one-click settings")
    print("=" * 80)
    for key, value in settings.items():
        print(f"{key:28s}: {value}")
    print("=" * 80)


def print_metrics(path: Path, title: str) -> None:
    """Print semantic, instance, efficiency, and resource metrics."""
    if not path.is_file():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print("\n" + title)
    print("-" * 80)
    percentage_metrics = (
        "semantic_dice",
        "semantic_iou",
        "mask_ap_50_95",
        "mask_ap_50",
        "mask_ap_75",
        "mask_ap_small",
        "mask_ap_medium",
        "mask_ap_large",
        "mask_precision",
        "mask_recall",
        "mask_f1",
    )
    for key in percentage_metrics:
        if key in metrics:
            print(f"{key:28s}: {float(metrics[key]) * 100:8.2f}%")
    for key in (
        "prediction_count",
        "params_m",
        "model_latency_ms_per_image",
        "peak_vram_mb",
    ):
        if key in metrics:
            print(f"{key:28s}: {float(metrics[key]):8.3f}")


def train_model(paths: dict[str, Path], smoke: bool) -> Path:
    """Train with automatic resume and return the best checkpoint."""
    name = "UNET_R18_WATERSHED_smoke" if smoke else FORMAL_NAME
    epochs = 2 if smoke else FORMAL_EPOCHS
    batch_size = 2 if smoke else BATCH_SIZE
    workers = min(WORKERS, 2) if smoke else WORKERS
    val_interval = 1 if smoke else 5
    save_interval = 1 if smoke else 25
    run_dir = paths["output"] / name
    best_weights = run_dir / "model_best.pth"
    print_settings(RUN_MODE, paths, smoke)
    if completed_epochs(run_dir) >= epochs and best_weights.is_file():
        print(f"Training already complete: {run_dir}")
        print_metrics(
            run_dir / "validation" / "best_metrics.json",
            "Best validation metrics",
        )
        return best_weights

    arguments: list[object] = [
        "--dataset",
        paths["prepared"],
        "--output-root",
        paths["output"],
        "--name",
        name,
        "--encoder",
        ENCODER,
        "--encoder-weights",
        ENCODER_WEIGHTS,
        "--epochs",
        epochs,
        "--batch",
        batch_size,
        "--workers",
        workers,
        "--lr",
        LEARNING_RATE,
        "--weight-decay",
        WEIGHT_DECAY,
        "--imgsz",
        IMAGE_SIZE,
        "--prob-threshold",
        PROBABILITY_THRESHOLD,
        "--watershed-min-distance",
        WATERSHED_MIN_DISTANCE,
        "--watershed-min-area",
        WATERSHED_MIN_AREA,
        "--max-instances",
        MAX_INSTANCES,
        "--val-interval",
        val_interval,
        "--save-interval",
        save_interval,
        "--seed",
        SEED,
        "--device",
        DEVICE,
    ]
    if not USE_AMP:
        arguments.append("--no-amp")
    last_weights = run_dir / "model_last.pth"
    if last_weights.is_file():
        print(f"Interrupted run found; resuming: {last_weights}")
        arguments.extend(["--resume", last_weights])
    run_python("train_unet.py", *arguments)
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Best U-Net checkpoint was not created: {best_weights}"
        )
    print_metrics(
        run_dir / "validation" / "best_metrics.json",
        "Best validation metrics",
    )
    return best_weights


def test_model(paths: dict[str, Path], weights: Path | None = None) -> None:
    """Evaluate a best checkpoint on the untouched test split."""
    weights = weights or paths["output"] / FORMAL_NAME / "model_best.pth"
    if not weights.is_file():
        raise FileNotFoundError(
            f"Formal checkpoint not found: {weights}\n"
            "Run with RUN_MODE='train' first."
        )
    output = paths["evaluation"] / f"{FORMAL_NAME}_test"
    metrics_path = output / "metrics.json"
    print_settings(RUN_MODE, paths)
    if not metrics_path.is_file():
        arguments: list[object] = [
            "--weights",
            weights,
            "--dataset",
            paths["prepared"],
            "--split",
            "test",
            "--output",
            output,
            "--batch",
            min(BATCH_SIZE, 4),
            "--workers",
            WORKERS,
            "--device",
            DEVICE,
            "--prob-threshold",
            PROBABILITY_THRESHOLD,
            "--watershed-min-distance",
            WATERSHED_MIN_DISTANCE,
            "--watershed-min-area",
            WATERSHED_MIN_AREA,
            "--max-instances",
            MAX_INSTANCES,
        ]
        run_python("eval_unet.py", *arguments)
    else:
        print(f"Test metrics already exist: {metrics_path}")
    print_metrics(metrics_path, "Final test metrics")


def main() -> None:
    """Run the selected one-click workflow."""
    mode = RUN_MODE.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"RUN_MODE must be one of {sorted(VALID_MODES)}")
    paths = platform_paths()
    check_environment()
    prepare_dataset(paths)
    if mode == "prepare":
        print_settings(mode, paths)
        return
    if mode == "smoke":
        train_model(paths, smoke=True)
        return
    if mode == "train":
        train_model(paths, smoke=False)
        return
    if mode == "test":
        test_model(paths)
        return
    if mode == "all":
        best_weights = train_model(paths, smoke=False)
        test_model(paths, best_weights)


if __name__ == "__main__":
    main()
