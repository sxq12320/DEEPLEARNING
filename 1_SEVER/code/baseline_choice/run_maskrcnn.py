"""One-click launcher for the citrus Torchvision Mask R-CNN baseline.

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
WINDOWS_OUTPUT_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\torchvision_maskrcnn")
WINDOWS_EVALUATION_ROOT = Path(r"E:\mastercode\4_baseline_choice\runs\evaluation")

# Linux server paths. The source dataset is read-only; converted COCO data
# and outputs are written under results/002_retrain.
SERVER_SOURCE_DATASET = Path("/data/sxq/datasets/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/sxq/datasets/citrus_prepared")
SERVER_OUTPUT_ROOT = Path("/data/sxq/results/002_retrain/maskrcnn")
SERVER_EVALUATION_ROOT = Path("/data/sxq/results/002_retrain/evaluation")

# Formal experiment settings
FORMAL_NAME = "001_6_maskrcnn_r50_fpn_seed42"
FORMAL_EPOCHS = 300
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.005
IMAGE_SIZE = 640
WORKERS = 4
DEVICE = "cuda:0"
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
    """Run one existing baseline script with the active Python environment."""
    command = [sys.executable, str(SCRIPTS / script_name)]
    command.extend(str(value) for value in arguments)
    print("\n" + "=" * 80)
    print("Running:", subprocess.list2cmdline(command))
    print("=" * 80)
    subprocess.run(command, cwd=SUITE_ROOT, check=True)


def check_environment() -> None:
    """Check Torchvision operators, pycocotools, and CUDA before starting."""
    try:
        import pycocotools  # noqa: F401
        import torch
        import torchvision
        from torchvision.ops import nms
    except ImportError as exc:
        raise RuntimeError(
            "The active Python environment is incomplete. Install torch, torchvision, "
            "pycocotools, Pillow, PyYAML, and numpy first."
        ) from exc

    nms(
        torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        torch.tensor([1.0]),
        0.5,
    )
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "REQUIRE_CUDA=True, but the current PyTorch is CPU-only. "
            "Use the server CUDA environment, or temporarily set REQUIRE_CUDA=False."
        )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def prepared_dataset_exists(path: Path) -> bool:
    """Check whether all converted COCO splits exist."""
    annotation_dir = path / "coco" / "annotations"
    return all(
        (annotation_dir / f"instances_{split}.json").is_file()
        for split in ("train", "val", "test")
    )


def prepare_dataset(paths: dict[str, Path]) -> None:
    """Convert the YOLO polygons when COCO files are absent, then validate them."""
    if not prepared_dataset_exists(paths["prepared"]):
        if not paths["source"].is_dir():
            raise FileNotFoundError(
                f"Source dataset not found: {paths['source']}\n"
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
        print(f"Prepared COCO dataset already exists: {paths['prepared']}")

    run_python(
        "validate_torchvision_maskrcnn_dataset.py",
        "--dataset",
        paths["prepared"],
    )


def completed_epochs(run_dir: Path) -> int:
    """Return the latest epoch recorded in history.json."""
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return 0
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return int(history[-1]["epoch"]) if history else 0


def print_metrics(path: Path, title: str) -> None:
    """Print the metrics needed for the paper table."""
    if not path.is_file():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print("\n" + title)
    print("-" * len(title))
    for key in (
        "mask_ap_50_95",
        "mask_ap_50",
        "mask_ap_75",
        "mask_precision",
        "mask_recall",
        "mask_f1",
    ):
        if key in metrics:
            print(f"{key:22s}: {float(metrics[key]) * 100:7.2f}%")
    for key in ("params_m", "model_latency_ms_per_image", "peak_vram_mb"):
        if key in metrics:
            print(f"{key:22s}: {float(metrics[key]):7.2f}")


def train_model(paths: dict[str, Path], smoke: bool) -> Path:
    """Run smoke/formal training and resume automatically after interruption."""
    name = "MRCNN_R50_FPN_smoke" if smoke else FORMAL_NAME
    epochs = 2 if smoke else FORMAL_EPOCHS
    batch_size = 1 if smoke else BATCH_SIZE
    learning_rate = 0.0025 if smoke else LEARNING_RATE
    detections_per_image = 20 if smoke else 50
    workers = 2 if smoke else WORKERS
    val_interval = 1 if smoke else 5
    save_interval = 1 if smoke else 25
    run_dir = paths["output"] / name
    best_weights = run_dir / "model_best.pth"

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
        "--epochs",
        epochs,
        "--batch",
        batch_size,
        "--workers",
        workers,
        "--lr",
        learning_rate,
        "--imgsz",
        IMAGE_SIZE,
        "--detections-per-image",
        detections_per_image,
        "--seed",
        SEED,
        "--val-interval",
        val_interval,
        "--save-interval",
        save_interval,
        "--device",
        DEVICE,
    ]
    last_weights = run_dir / "model_last.pth"
    if last_weights.is_file():
        print(f"Interrupted run found; resuming from: {last_weights}")
        arguments.extend(["--resume", last_weights])

    run_python("train_torchvision_maskrcnn.py", *arguments)
    if not best_weights.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_weights}")
    print_metrics(
        run_dir / "validation" / "best_metrics.json",
        "Best validation metrics",
    )
    return best_weights


def test_model(paths: dict[str, Path], weights: Path | None = None) -> None:
    """Evaluate the formal best checkpoint on the test split."""
    weights = weights or paths["output"] / FORMAL_NAME / "model_best.pth"
    if not weights.is_file():
        raise FileNotFoundError(
            f"Formal best checkpoint not found: {weights}\n"
            "Set RUN_MODE='train' first."
        )
    output = paths["evaluation"] / f"{FORMAL_NAME}_test"
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        run_python(
            "eval_torchvision_maskrcnn.py",
            "--weights",
            weights,
            "--dataset",
            paths["prepared"],
            "--split",
            "test",
            "--output",
            output,
            "--workers",
            WORKERS,
            "--device",
            DEVICE,
        )
    else:
        print(f"Test result already exists: {metrics_path}")
    print_metrics(metrics_path, "Final test metrics")


def main() -> None:
    """Run the selected workflow."""
    mode = RUN_MODE.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"RUN_MODE must be one of {sorted(VALID_MODES)}")
    paths = platform_paths()
    print(f"Mask R-CNN one-click mode: {mode}")
    for name, path in paths.items():
        print(f"{name:12s}: {path}")

    check_environment()
    prepare_dataset(paths)
    if mode == "prepare":
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
