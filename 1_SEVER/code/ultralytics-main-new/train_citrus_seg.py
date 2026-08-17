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
DATA = r"/data/sxq/datasets/orange_yolo/data.yaml"
PROJECT = r"/data/sxq/results/000_anyothers"
FIXED = dict(
    optimizer="AdamW",
    patience=100,
    lr0=0.001,
    workers=4,
    cache=True,
    amp=0,
    dropout=0.1,
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
    p.add_argument("--pretrained", default=None,
                   help="Optional .pt to transfer matching weights into a .yaml model (recommended for E1..E4).")
    # only knobs allowed to vary: epochs/batch/imgsz/device — everything else is FIXED above.
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)     # LOCKED at 640 for this study
    p.add_argument("--device", default="0")
    # ---- loss/optimizer ablation knobs (F-series experiments; defaults keep the E-protocol EXACTLY) ----
    p.add_argument("--iou-type", default="CIoU",
                   help="box loss: CIoU/EIoU/SIoU/MPDIoU/ShapeIoU/WIoU/NWDWise/FocalerCIoU/FocalerWIoU")
    p.add_argument("--inner-ratio", type=float, default=1.0, help="Inner-IoU ratio (arXiv:2311.02877), 1.0=off")
    p.add_argument("--nwd-ratio", type=float, default=0.0, help="NWD blend ratio (arXiv:2110.13389), 0.0=off")
    p.add_argument("--slide", action="store_true", help="enable Slide Loss cls weighting (arXiv:2208.02019)")
    p.add_argument("--optimizer", default="AdamW", help="AdamW (default) / Lion / SGD / PIDAO / SMCAO / MuSGD")
    p.add_argument("--tal-metric", default="CIoU", help="GA-TAL assigner metric: CIoU (stock) / NWD / Mix")
    p.add_argument("--freq-loss", type=float, default=0.0, help="FFL 频域掩码对齐损失比重 (0=off, 建议 0.05-0.2)")
    p.add_argument("--tal-min-pos", action="store_true", help="GA-TAL: guarantee >=1 positive per GT (tiny fruits)")
    p.add_argument("--aug-preset", default="none", choices=["none", "dark", "smallobj", "dark_smallobj"],
                   help="data-driven aug recipes (M-series): dark=hsv_v 0.6 逆曝光增强; smallobj=copy_paste 0.3+scale 0.7")
    p.add_argument("--lr0", type=float, default=None,
                   help="override lr0 ONLY for optimizer ablations (e.g. Lion needs 0.001-0.003); default keeps 0.01")
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
    overrides = dict(FIXED)
    overrides["optimizer"] = args.optimizer
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    # loss ablation knobs: only pass when non-default so E-series args.yaml stays byte-identical
    if args.iou_type != "CIoU":
        overrides["iou_type"] = args.iou_type
    if args.inner_ratio != 1.0:
        overrides["inner_ratio"] = args.inner_ratio
    if args.nwd_ratio != 0.0:
        overrides["nwd_ratio"] = args.nwd_ratio
    if args.slide:
        overrides["use_slide"] = True
    if args.tal_metric != "CIoU":
        overrides["tal_metric"] = args.tal_metric
    if args.tal_min_pos:
        overrides["tal_min_pos"] = True
    if args.freq_loss > 0:
        overrides["freq_loss"] = args.freq_loss
    # M 系列数据配方（依据 _dataset_analysis.md：小果显著更暗 → dark；47.9% <32px → smallobj）
    AUG_PRESETS = {
        "dark": dict(hsv_v=0.6),                                # 亮度扰动加强，模拟远处欠曝（数据: 小果 V=103 vs 大果 132）
        "smallobj": dict(copy_paste=0.3, scale=0.7),            # 分割 copy-paste 免费扩增小目标 (Kisantal 2019, arXiv:1902.07296)
        "dark_smallobj": dict(hsv_v=0.6, copy_paste=0.3, scale=0.7),
    }
    if args.aug_preset != "none":
        overrides.update(AUG_PRESETS[args.aug_preset])
    model.train(
        data=DATA,
        project=PROJECT,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        **overrides,
    )


if __name__ == "__main__":
    main()
