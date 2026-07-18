"""Clean, reproducible training driver for the immature-citrus instance-seg experiments.

Every hyperparameter except the model architecture is FIXED here so that E0 (baseline)
and every improved variant (E1..E4) differ ONLY in architecture — the iron rule for a
clean ablation. Outputs land in 1_results/ORANGE_WUXI_SEG/<name>/.

Examples:
    # E0 baseline from COCO-pretrained weights
    python train_citrus_seg.py --model yolo11n-seg.pt --name E0_yolo11n_seg_baseline_941

    # an improved architecture from a YAML, transferring matching COCO weights
    python train_citrus_seg.py --model citrus_yaml/E1_yolo11n_seg_p2.yaml \
        --pretrained yolo11n-seg.pt --name E1_p2_head

    # quick 3-epoch smoke test to verify the data/protocol before the real 300-epoch run
    python train_citrus_seg.py --model yolo11n-seg.pt --name E0_smoke --epochs 3
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from ultralytics import YOLO

# ---- fixed experiment protocol (do NOT change between experiments) ----
SEED = 42
DATA = r"E:/mastercode/data/test/orange_wuxi_seg.yaml"
PROJECT = r"E:/mastercode/ultralytics-main-new/1_results/ORANGE_WUXI_SEG"
FIXED = dict(
    optimizer="AdamW",
    patience=100,
    lr0=0.01,
    workers=4,
    cache=False,
    amp=0,          # AMP off — matches the existing orange runs, avoids fp16 mask noise
    dropout=0.0,
    seed=SEED,
    deterministic=True,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an immature-citrus YOLO-seg experiment (fixed protocol).")
    p.add_argument("--model", required=True, help=".pt weights or a model .yaml (architecture under test).")
    p.add_argument("--name", required=True, help="Run name, e.g. E0_yolo11n_seg_baseline_941.")
    p.add_argument("--data", default=DATA, help="Dataset YAML. Defaults to the current single-split dataset.")
    p.add_argument("--project", default=PROJECT, help="Experiment output directory.")
    p.add_argument("--pretrained", default=None,
                   help="Optional .pt to transfer matching weights into a .yaml model (recommended for E1..E4).")
    # only knobs allowed to vary: epochs/batch/imgsz/device — everything else is FIXED above.
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)     # LOCKED at 640 for this study
    p.add_argument("--device", default="0")
    return p.parse_args()


def build_model(model: str, pretrained: str | None) -> YOLO:
    """YOLO(.pt) for the baseline, or YOLO(.yaml).load(.pt) to transfer matching weights."""
    if model.endswith(".yaml") and pretrained:
        return YOLO(model).load(pretrained)
    return YOLO(model)


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    model = build_model(args.model, args.pretrained)
    model.train(
        data=args.data,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        **FIXED,
    )


if __name__ == "__main__":
    main()
