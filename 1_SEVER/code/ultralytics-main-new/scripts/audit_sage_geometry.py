"""Measure train/val geometry and v4 target coverage; do not read test or alter labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import polygons2masks_overlap
from ultralytics.utils.sage_v4_loss import structure_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    payload = {
        "dataset": str(args.dataset),
        "input_size": 640,
        "mask_stride": 4,
        "protocol": "Square letterbox, no augmentation. Current local train/val only; not server manifest proof.",
        "splits": {},
    }
    for split in ("train", "val"):
        records, image_records = [], []
        for image_path in sorted((args.dataset / split / "images").glob("*.jpg")):
            with Image.open(image_path) as image:
                width, height = image.size
            ratio = min(640 / width, 640 / height)
            scaled_w, scaled_h = round(width * ratio), round(height * ratio)
            offset = np.array([round((640 - scaled_w) / 2 - 0.1), round((640 - scaled_h) / 2 - 0.1)])
            polygons, areas, solidities = [], [], []
            label_path = args.dataset / split / "labels" / (image_path.stem + ".txt")
            # Missing or malformed labels are reported by failure; never deleted or rewritten.
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                values = [float(v) for v in line.split()]
                if len(values) < 7 or (len(values) - 1) % 2:
                    raise ValueError(f"Unexpected polygon format: {label_path}")
                polygon = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
                polygon = (polygon * np.array([width, height]) * ratio + offset).astype(np.float32)
                area = cv2.contourArea(polygon)
                hull_area = cv2.contourArea(cv2.convexHull(polygon))
                areas.append(area)
                solidities.append(area / hull_area if hull_area else 0)
                polygons.append(polygon)
            if not polygons:
                image_records.append({"image": image_path.name, "instances": 0, "scale_ratio": None})
                continue
            ids, order = polygons2masks_overlap((640, 640), polygons, downsample_ratio=4)
            targets, _ = structure_targets(torch.from_numpy(ids)[None], torch.empty(0), 1, (160, 160))
            boundary = targets[0, 1].numpy().astype(bool)
            separator = targets[0, 2].numpy().astype(bool)
            for number, original in enumerate(order, 1):
                own = ids == number
                pixels = int(own.sum())
                records.append(
                    {
                        "image": image_path.name,
                        "label_index": int(original),
                        "polygon_area_at_640": areas[original],
                        "solidity": solidities[original],
                        "p2_pixels": pixels,
                        "boundary_coverage": float(boundary[own].mean()) if pixels else None,
                        "separator_coverage": float(separator[own].mean()) if pixels else None,
                    }
                )
            positive_areas = [a for a in areas if a > 0]
            span = float(np.sqrt(max(positive_areas) / min(positive_areas))) if len(positive_areas) >= 2 else None
            image_records.append({"image": image_path.name, "instances": len(polygons), "scale_ratio": span})
        spans = [r["scale_ratio"] for r in image_records if r["scale_ratio"] is not None]
        retained = [r for r in records if r["p2_pixels"]]
        small = [r for r in retained if r["polygon_area_at_640"] < 32**2]
        tiny = [r for r in retained if r["polygon_area_at_640"] < 16**2]
        summary = {
            "images": len(image_records),
            "instances": len(records),
            "solidity_lt_09": sum(r["solidity"] < 0.9 for r in records),
            "solidity_lt_08": sum(r["solidity"] < 0.8 for r in records),
            "area_lt_256": sum(r["polygon_area_at_640"] < 256 for r in records),
            "area_lt_1024": sum(r["polygon_area_at_640"] < 1024 for r in records),
            "lost_at_stride4_overlap": len(records) - len(retained),
            "multi_instance_images": len(spans),
            "scale_ratio_median": float(np.median(spans)),
            "scale_ratio_p90": float(np.quantile(spans, 0.9)),
            "scale_ratio_gt4": sum(s > 4 for s in spans),
            "retained_tiny": len(tiny),
            "retained_small": len(small),
            "tiny_boundary_coverage_median": float(np.median([r["boundary_coverage"] for r in tiny])),
            "tiny_fully_boundary": sum(r["boundary_coverage"] == 1 for r in tiny),
            "small_boundary_coverage_median": float(np.median([r["boundary_coverage"] for r in small])),
        }
        payload["splits"][split] = {"summary": summary, "instances": records, "images": image_records}
        print(split, json.dumps(summary, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
