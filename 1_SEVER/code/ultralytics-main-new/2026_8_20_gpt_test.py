#!/usr/bin/env python3
"""Train the ten 2026-08-20 citrus instance-segmentation candidates in a fixed protocol."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "2026_8_20_gpt_test"
MODEL_YAMLS = tuple(sorted(YAML_DIR.glob("*.yaml")))


def parse_args() -> argparse.Namespace:
    """Parse batch-training arguments while keeping one protocol for all selected models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/data/sxq/datasets/orange_yolo/data.yaml", help="Dataset YAML path.")
    parser.add_argument("--project", default="/data/sxq/results/2026_8_20_gpt_test", help="Output project path.")
    parser.add_argument("--pretrained", default="yolo11n-seg.pt", help="Pretrained checkpoint, or 'none'.")
    parser.add_argument("--start", type=int, default=1, choices=range(1, 11), help="First model index (1-10).")
    parser.add_argument("--end", type=int, default=10, choices=range(1, 11), help="Last model index (1-10).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run 3 epochs at 320 px for integration testing.")
    parser.add_argument("--dry-run", action="store_true", help="Build selected models without training.")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip, rather than overwrite, existing run folders."
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to the next model after a failure.")
    return parser.parse_args()


def select_models(start: int, end: int) -> tuple[Path, ...]:
    """Return the inclusive, ordered model slice and reject incomplete YAML collections."""
    if len(MODEL_YAMLS) != 10:
        raise RuntimeError(f"Expected exactly 10 model YAMLs in {YAML_DIR}, found {len(MODEL_YAMLS)}")
    if start > end:
        raise ValueError(f"--start ({start}) must not exceed --end ({end})")
    return MODEL_YAMLS[start - 1 : end]


def training_options(args: argparse.Namespace, run_name: str) -> dict:
    """Create the literature-backed, shared screening protocol for one run."""
    epochs = 3 if args.smoke else args.epochs
    imgsz = 320 if args.smoke else args.imgsz
    return {
        "data": args.data,
        "project": args.project,
        "name": run_name,
        "epochs": epochs,
        "batch": min(args.batch, 2) if args.smoke else args.batch,
        "imgsz": imgsz,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 5.0,
        "dropout": 0.0,
        "amp": True,
        "cache": False,
        "patience": 100,
        "close_mosaic": 20,
        "mosaic": 0.5,
        "scale": 0.35,
        "copy_paste": 0.1,
        "iou_type": "CIoU",
        "nwd_ratio": 0.2,
        "tal_metric": "Mix",
        "tal_min_pos": True,
        "mask_dice": 0.5,
        "boundary_loss": 0.2,
        "freq_loss": 0.05,
        "freq_roi_size": 32,
        "plots": True,
        "save": True,
        "save_period": -1,
        "exist_ok": False,
        "verbose": True,
    }


def run_one(model_yaml: Path, args: argparse.Namespace) -> None:
    """Build, optionally load pretrained weights, and train one candidate."""
    # Keep result-folder names exactly identical to YAML stems for unambiguous model/result matching.
    run_name = model_yaml.stem
    output_dir = Path(args.project) / run_name
    if output_dir.exists() and not args.dry_run:
        if args.skip_existing:
            print(f"[skip] {run_name}: {output_dir} already exists")
            return
        raise FileExistsError(
            f"Refusing to overwrite or auto-renumber {output_dir}. Move it, or rerun with --skip-existing."
        )

    print(f"\n{'=' * 96}\n[model] {model_yaml.name}\n[result] {output_dir}\n{'=' * 96}")
    model = YOLO(str(model_yaml))
    if args.pretrained.lower() != "none":
        model.load(args.pretrained)
    if args.dry_run:
        model.info(detailed=False, verbose=True)
        return
    model.train(**training_options(args, run_name))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Train the selected contiguous range of the ten candidates."""
    args = parse_args()
    selected = select_models(args.start, args.end)
    failures: list[tuple[str, str]] = []
    for model_yaml in selected:
        try:
            run_one(model_yaml, args)
        except Exception as exc:
            failures.append((model_yaml.name, repr(exc)))
            print(f"[failed] {model_yaml.name}: {exc}")
            if not args.continue_on_error:
                raise

    if failures:
        details = "\n".join(f"  - {name}: {error}" for name, error in failures)
        raise RuntimeError(f"{len(failures)} model(s) failed:\n{details}")
    print(f"\nCompleted {len(selected)} model(s).")


if __name__ == "__main__":
    main()
