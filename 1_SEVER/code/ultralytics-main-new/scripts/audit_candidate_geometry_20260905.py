"""Read-only candidate eligibility audit; not a training assignment or recall measurement."""

# ruff: noqa: E402 -- use the user-scoped checkout
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.utils.tal import TaskAlignedAssigner


def main():
    torch.set_num_threads(2)
    dataset = Path("E:/mastercode/data/orange_yolo/val")
    out = ROOT / "reports/research_reset_20260905/candidate_geometry.json"
    strides = [8, 16, 32]
    grids = []
    for stride in strides:
        yy, xx = np.meshgrid(np.arange(640 // stride), np.arange(640 // stride), indexing="ij")
        grids.append(np.stack((xx, yy), -1).reshape(-1, 2) * stride + stride / 2)
    anchors = torch.tensor(np.concatenate(grids), dtype=torch.float32)
    assigner = TaskAlignedAssigner(stride=strides)
    records, errors = [], []
    images = sorted((dataset / "images").glob("*.jpg"))
    for image in images:
        with Image.open(image) as im:
            w, h = im.size
        scale = min(640 / w, 640 / h)
        pad = np.array([round((640 - round(w * scale)) / 2 - 0.1),
                        round((640 - round(h * scale)) / 2 - 0.1)])
        label = dataset / "labels" / (image.stem + ".txt")
        for index, line in enumerate(label.read_text(encoding="utf-8").splitlines()):
            try:
                fields = np.asarray([float(v) for v in line.split()])
                points = fields[1:].reshape(-1, 2)
                if len(points) < 3:
                    raise ValueError("Not a polygon")
                points = points * np.array([w, h]) * scale + pad
                box = torch.tensor(np.concatenate((points.min(0), points.max(0))), dtype=torch.float32)
                raw = ((anchors > box[:2]) & (anchors < box[2:])).all(1)
                expanded = assigner.select_candidates_in_gts(
                    anchors, box.reshape(1, 1, 4), torch.ones(1, 1, 1)
                ).reshape(-1)
                wh = (box[2:] - box[:2]).numpy()
                records.append(dict(image=image.name, polygon_line=index, bbox_width=float(wh[0]),
                                    bbox_height=float(wh[1]), raw_inside=int(raw.sum()),
                                    current_eligible=int(expanded.sum()),
                                    dimension_expanded=bool((wh < 8).any())))
            except ValueError as exc:
                errors.append(dict(image=image.name, polygon_line=index, error=str(exc)))
    payload = dict(
        protocol="Local JPG validation; fixed square letterbox640; real current TAL eligibility function; no augmentation",
        limitation="Eligibility is before top-k, alignment scores and competing GT assignment. Not actual positives. "
                   "Train mosaic/scale and other evaluator shapes can change the geometry. No data edits.",
        images=len(images), instances=len(records), errors=errors,
        zero_raw_inside=sum(r["raw_inside"] == 0 for r in records),
        zero_current_eligible=sum(r["current_eligible"] == 0 for r in records),
        expanded_dimensions=sum(r["dimension_expanded"] for r in records),
        min_current_eligible=min(r["current_eligible"] for r in records),
        median_current_eligible=float(np.median([r["current_eligible"] for r in records])),
        records=records,
    )
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
