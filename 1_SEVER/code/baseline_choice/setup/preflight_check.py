"""Per-environment preflight check for the citrus baseline suite.

Run inside EACH conda environment before launching run_comparison_batch.py.
The script only imports and inspects; it never trains or downloads anything.

Examples (one per environment, on the server):
    python setup/preflight_check.py --family common
    python setup/preflight_check.py --family yolo
    python setup/preflight_check.py --family mmdet --mmdet-root /data/sxq/code/mmdetection
    python setup/preflight_check.py --family torchvision
    python setup/preflight_check.py --family rfdetr
    python setup/preflight_check.py --family unet
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SUITE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

RESULTS: list[tuple[str, str]] = []


def record(status: str, message: str) -> None:
    """Append one PASS/WARN/FAIL line and print it immediately."""
    RESULTS.append((status, message))
    print(f"[{status:4s}] {message}")


def check_import(module: str, label: str | None = None, warn: bool = False) -> object | None:
    """Import a module and record the outcome; returns the module or None."""
    name = label or module
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        record("WARN" if warn else "FAIL", f"{name}: import failed ({exc})")
        return None
    version = getattr(imported, "__version__", "?")
    record("PASS", f"{name}: {version}")
    return imported


def check_torch() -> None:
    """Check torch plus CUDA visibility."""
    torch = check_import("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        record("PASS", f"torch.cuda: {torch.cuda.get_device_name(0)} (cuda {torch.version.cuda})")
    else:
        record("FAIL", "torch.cuda.is_available() is False")


def check_registry():
    """Load configs/baselines.yaml through baseline_common when PyYAML exists."""
    try:
        from baseline_common import get_baseline, load_registry
    except ImportError as exc:
        record("WARN", f"registry helpers unavailable ({exc}); skipping registry-driven checks")
        return None, None
    try:
        registry = load_registry()
    except Exception as exc:
        record("FAIL", f"configs/baselines.yaml failed to load: {exc}")
        return None, None
    record("PASS", f"configs/baselines.yaml: {len(registry.get('baselines', {}))} baselines")
    return registry, get_baseline


def check_common(args: argparse.Namespace) -> None:
    """Shared dependencies needed by every environment."""
    check_import("yaml", "PyYAML")
    check_import("PIL", "Pillow")
    check_import("numpy")
    check_import("tqdm")
    check_import("pycocotools")
    if args.source is not None:
        source = Path(args.source)
        for split in ("train", "val", "test"):
            layouts = (
                (source / "images" / split, source / "labels" / split),
                (source / split / "images", source / split / "labels"),
            )
            if any(images.is_dir() and labels.is_dir() for images, labels in layouts):
                record("PASS", f"source split '{split}': found under {source}")
            else:
                record("FAIL", f"source split '{split}': missing images/labels under {source}")
    if args.prepared is not None:
        prepared = Path(args.prepared)
        missing = [
            f"{layout}/{split}"
            for layout in ("yolo/images", "yolo/labels", "coco/images", "semantic/images", "semantic/masks")
            for split in ("train", "val", "test")
            if not (prepared / layout / split).is_dir()
        ]
        missing += [
            f"coco/annotations/instances_{split}.json"
            for split in ("train", "val", "test")
            if not (prepared / "coco" / "annotations" / f"instances_{split}.json").is_file()
        ]
        if missing:
            record("WARN", f"prepared dataset incomplete ({len(missing)} missing); "
                   "run_comparison_batch.py will create it on first run")
        else:
            record("PASS", f"prepared dataset: complete at {prepared}")


def check_yolo() -> None:
    """Ultralytics fork with the YOLO segmentation model zoo."""
    ultralytics = check_import("ultralytics")
    if ultralytics is not None:
        major_minor = tuple(int(part) for part in str(ultralytics.__version__).split(".")[:2])
        if major_minor >= (8, 4):
            record("PASS", "ultralytics >= 8.4 (yolo26n-seg available)")
        else:
            record("WARN", "ultralytics < 8.4; yolo26n-seg may be unavailable")
    check_torch()


def check_mmdet(args: argparse.Namespace) -> None:
    """MMDetection stack, official configs, and pretrained checkpoints."""
    check_import("mmengine")
    check_import("mmcv")
    check_import("mmdet")
    check_torch()
    mmdet_root = Path(args.mmdet_root) if args.mmdet_root else None
    if mmdet_root is None or not mmdet_root.is_dir():
        record("FAIL", f"mmdetection repo not found: {mmdet_root} "
               "(clone v3.3.0, see setup/install_server_envs.sh)")
        return
    registry, get_baseline = check_registry()
    if not registry:
        return
    for name, entry in registry["baselines"].items():
        if entry.get("family") != "mmdetection":
            continue
        config = mmdet_root / str(entry["config"])
        if config.is_file():
            record("PASS", f"{name}: config {entry['config']}")
        else:
            record("FAIL", f"{name}: missing config {config}")
        pattern = entry.get("checkpoint_glob")
        if pattern:
            if list(mmdet_root.glob(pattern)):
                record("PASS", f"{name}: pretrained checkpoint present ({pattern})")
            else:
                record("WARN", f"{name}: no checkpoint matching '{pattern}'; "
                       "run setup/fetch_mmdet_checkpoints.py (training falls back to config default init)")


def check_torchvision() -> None:
    """Torchvision detection stack."""
    check_torch()
    check_import("torchvision")
    check_import("torchvision.models.detection", "torchvision.models.detection")


def check_rfdetr() -> None:
    """RF-DETR package and the configured segmentation class."""
    check_torch()
    package = check_import("rfdetr")
    registry, get_baseline = check_registry()
    if package is None or not registry:
        return
    entry = get_baseline("rfdetr_seg_nano", family="rfdetr")
    class_name = str(entry["model_class"])
    if hasattr(package, class_name) or hasattr(package, "RFDETRSegPreview"):
        record("PASS", f"rfdetr provides {class_name} (or RFDETRSegPreview fallback)")
    else:
        record("FAIL", f"rfdetr lacks {class_name} and RFDETRSegPreview")


def check_unet() -> None:
    """Semantic-baseline stack: SMP, timm, watershed deps."""
    check_torch()
    check_import("segmentation_models_pytorch")
    check_import("timm")
    check_import("skimage")
    check_import("scipy")


CHECKS = {
    "common": check_common,
    "yolo": lambda args: check_yolo(),
    "mmdet": check_mmdet,
    "torchvision": lambda args: check_torchvision(),
    "rfdetr": lambda args: check_rfdetr(),
    "unet": lambda args: check_unet(),
}


def main() -> int:
    """Run the selected family checks and return a shell-friendly exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(CHECKS), help="Environment family to check.")
    parser.add_argument("--source", type=Path, default=None, help="Source YOLO dataset root (common).")
    parser.add_argument("--prepared", type=Path, default=None, help="Prepared dataset root (common).")
    parser.add_argument("--mmdet-root", type=Path, default=None, help="Cloned MMDetection repo root (mmdet).")
    args = parser.parse_args()
    print(f"Python: {sys.executable}")
    print(f"Family: {args.family}")
    CHECKS[args.family](args)
    failures = sum(1 for status, _ in RESULTS if status == "FAIL")
    warnings = sum(1 for status, _ in RESULTS if status == "WARN")
    print("-" * 60)
    print(f"FAIL={failures} WARN={warnings} PASS={len(RESULTS) - failures - warnings}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
