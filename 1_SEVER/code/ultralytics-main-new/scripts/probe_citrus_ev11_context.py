"""Probe author-inspired GAP/stable aggregation on actual trained C5 features.

This is a multi-fruit visible-mask adaptation of patch-score localization, NOT
the paper's single-object ImageNet Point-in-Box score. It does not compute AP.
Use the SAME validation membership as the trained weights. Never tune on test.
"""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.modules.block import C2PSA
from ultralytics.nn.modules.citrus_e_v11 import EV11StablePool
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=32)
    args = p.parse_args()
    if args.limit < 1 or args.output.exists():
        raise ValueError("Use a positive limit and a new output filename")
    torch.set_num_threads(2)
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    hyp = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    dataset = YOLODataset(
        img_path=str(Path(cfg["path"]) / cfg["val"]),
        data=cfg,
        task="segment",
        imgsz=640,
        augment=False,
        hyp=hyp,
        batch_size=2,
        cache=False,
        rect=False,
    )
    model = YOLO(str(args.weights), verbose=False).model.float().eval()
    stages = [m for m in model.model if isinstance(m, C2PSA)]
    if len(stages) != 1:
        raise ValueError("Probe expects exactly one C5 C2PSA/EV11ContextStage")
    captured = []

    def capture(module, inputs):
        captured.append(inputs[0].detach())

    hook = stages[0].m[0].attn.register_forward_pre_hook(capture)
    gap, stable = EV11StablePool(stages[0].c, False), EV11StablePool(stages[0].c, True)
    rows = []
    try:
        with torch.inference_mode():
            indices = torch.linspace(0, len(dataset) - 1, min(args.limit, len(dataset))).long().tolist()
            for start in range(0, len(indices), 2):
                batch = dataset.collate_fn([dataset[i] for i in indices[start : start + 2]])
                captured.clear()
                model(batch["img"].float() / 255)
                features = captured[0]
                maps = [F.cosine_similarity(features, pool(features), dim=1, eps=1e-6) for pool in (gap, stable)]
                for i, path in enumerate(batch["im_file"]):
                    masks = batch["masks"][i] > 0
                    row = dict(image=Path(path).name, grid=list(features.shape[-2:]))
                    for name, scores in zip(("gap", "stable"), maps):
                        h, w = scores.shape[-2:]
                        peak = int(scores[i].flatten().argmax())
                        y, x = divmod(peak, w)
                        # Evaluate the centre against ORIGINAL mask grid, not max-pooled inflated foreground.
                        yy = min(masks.shape[0] - 1, int((y + 0.5) * masks.shape[0] / h))
                        xx = min(masks.shape[1] - 1, int((x + 0.5) * masks.shape[1] / w))
                        row[name + "_point_in_visible_mask"] = bool(masks[yy, xx])
                    rows.append(row)
    finally:
        hook.remove()
    result = dict(
        weights=str(args.weights.resolve()),
        data=str(args.data.resolve()),
        images=len(rows),
        rows=rows,
        gap_hit_fraction=sum(r["gap_point_in_visible_mask"] for r in rows) / max(1, len(rows)),
        stable_hit_fraction=sum(r["stable_point_in_visible_mask"] for r in rows) / max(1, len(rows)),
        limitations="Frozen C5 features; coarse cell centres; multi-object visible masks; not original PiB, AP or causal proof.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
