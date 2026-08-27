"""Training driver for the HVIEnhance low-light front-end citrus-seg variant.

Identical FIXED protocol to ``train_citrus_seg.py`` (imported, not copied), plus a
layer-index-shift-aware transfer loader. Inserting ``HVIEnhance`` at layer 0 shifts
every downstream layer by +1, so a plain ``YOLO(...).load(pt)`` would match ~0
tensors by key name (``model.<i>.*``). We remap pretrained ``model.<i>.*`` ->
``model.<i+1>.*`` before loading, so the whole backbone/neck/head transfers from
COCO and only the new front-end (layer 0) is randomly initialised — keeping the
ablation vs the 001 baseline fair.

Examples:
    # smoke test (random init, no transfer) — just proves the pipeline runs
    python train_citrus_seg_hvi.py --model 0_orange_yaml/010_yolo11-seg-hvi.yaml \
        --name E_hvi_smoke --epochs 3

    # real run with fair COCO transfer into the shifted layers
    python train_citrus_seg_hvi.py --model 0_orange_yaml/010_yolo11-seg-hvi.yaml \
        --pretrained yolo11n-seg.pt --name E_hvi
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import HVIEnhance  # noqa: F401 — ensures the module is imported/registered

from train_citrus_seg import DATA, FIXED, PROJECT, SEED, set_seed


def _safe_load(pretrained: str):
    """Load an Ultralytics .pt across torch versions (weights_only default changed)."""
    try:
        return torch.load(pretrained, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(pretrained, map_location="cpu")


def load_pretrained_shifted(model: YOLO, pretrained: str, shift: int = 1) -> YOLO:
    """Transfer COCO weights into an index-shifted model (HVIEnhance inserted at layer 0)."""
    ckpt = _safe_load(pretrained)
    src = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    src_sd = src.float().state_dict()

    def shift_key(k: str) -> str:
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit():
            parts[1] = str(int(parts[1]) + shift)
        return ".".join(parts)

    shifted = {shift_key(k): v for k, v in src_sd.items()}
    dst = model.model.state_dict()
    inter = {k: v for k, v in shifted.items() if k in dst and v.shape == dst[k].shape}
    model.model.load_state_dict(inter, strict=False)
    print(f"[HVI transfer] loaded {len(inter)}/{len(dst)} tensors from {pretrained} (+{shift} index shift)")
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the HVIEnhance citrus-seg variant (FIXED protocol).")
    p.add_argument("--model", required=True, help="HVI model .yaml, e.g. 0_orange_yaml/010_yolo11-seg-hvi.yaml.")
    p.add_argument("--name", required=True, help="Run name.")
    p.add_argument("--data", default=DATA, help="Dataset YAML (defaults to the baseline protocol dataset).")
    p.add_argument("--project", default=PROJECT, help="Experiment output directory.")
    p.add_argument("--pretrained", default=None,
                   help=".pt to transfer (shift-aware). Omit for from-scratch / smoke test.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)  # LOCKED at 640 for this study
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    model = YOLO(args.model)
    if args.pretrained:
        load_pretrained_shifted(model, args.pretrained, shift=1)

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
