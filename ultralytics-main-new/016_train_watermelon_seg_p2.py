"""
Train the 016 first-stage YOLO11n-P2 segmentation model for watermelon flowers.

The model keeps YOLO11n as the lightweight base and adds a P2/4 segmentation
branch for smaller flower instances and finer mask boundaries.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent

DEFAULT_MODEL = ROOT / "mine_yaml" / "016_yolo11n_seg_p2.yaml"
DEFAULT_DATA = ROOT / "208_shr_watermelon_seg.yaml"
DEFAULT_PROJECT = ROOT / "results"
DEFAULT_PRETRAINED = WORKSPACE_ROOT / "yolo11n-seg.pt"


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11n-P2 segmentation for watermelon flowers.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="P2 segmentation model YAML.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML.")
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED), help="Pretrained YOLO11n-seg weights.")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from random initialization.")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--optimizer", default="AdamW", choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"])
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--cos-lr", type=str2bool, default=True)

    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="16_watermelon_seg_p2")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk according to Ultralytics defaults.")

    parser.add_argument("--mask-ratio", type=int, default=4, help="Mask downsample ratio. Try 2 for finer masks if VRAM allows.")
    parser.add_argument("--overlap-mask", type=str2bool, default=True)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--copy-paste", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.5)
    parser.add_argument("--hsv-v", type=float, default=0.4)

    parser.add_argument("--smoke", action="store_true", help="Run a tiny 1-epoch load/training check.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_pretrained(args):
    if args.no_pretrained:
        return ""

    pretrained = str(args.pretrained).strip()
    if not pretrained:
        return ""

    path = Path(pretrained)
    if path.exists():
        return str(path)

    # Fall back to the Ultralytics model name if the local absolute path is not available.
    if path.name == "yolo11n-seg.pt":
        return "yolo11n-seg.pt"

    raise FileNotFoundError(f"Pretrained weights not found: {pretrained}")


def main():
    args = parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch = 1
        args.imgsz = min(args.imgsz, 320)
        args.workers = 0
        if not args.name.endswith("_smoke"):
            args.name = f"{args.name}_smoke"
        args.exist_ok = True

    set_seed(args.seed)

    pretrained = resolve_pretrained(args)
    print("=" * 60)
    print("Train 016 YOLO11n-P2 segmentation")
    print("=" * 60)
    print(f"Model YAML: {args.model}")
    print(f"Dataset YAML: {args.data}")
    print(f"Pretrained: {pretrained or 'none'}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Device: {args.device}")

    model = YOLO(args.model)
    if pretrained:
        model.load(pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        cos_lr=args.cos_lr,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=args.deterministic,
        exist_ok=args.exist_ok,
        cache=args.cache,
        mask_ratio=args.mask_ratio,
        overlap_mask=args.overlap_mask,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        scale=args.scale,
        translate=args.translate,
        degrees=args.degrees,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Best weights will be saved at: {best_path}")


if __name__ == "__main__":
    main()
