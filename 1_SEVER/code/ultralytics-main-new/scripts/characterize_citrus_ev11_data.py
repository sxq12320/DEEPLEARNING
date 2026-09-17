"""Read cached polygon geometry; never repair images, filter labels or edit the split."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {}
    for split in ("train", "val", "test"):
        cache = args.dataset / split / "labels.cache"
        if cache.is_file():
            labels = np.load(cache, allow_pickle=True).item()["labels"]  # Trusted local workspace cache only.
        else:
            # Do NOT instantiate an auto-repairing YOLODataset just to characterize
            # an uncached split. Read the existing polygon text and image headers.
            labels = []
            for image in sorted((args.dataset / split / "images").iterdir()):
                if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                label_path = args.dataset / split / "labels" / (image.stem + ".txt")
                with Image.open(image) as im:
                    w, h = im.size
                segments = [
                    np.asarray([float(x) for x in line.split()[1:]], np.float32).reshape(-1, 2)
                    for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
                labels.append(dict(shape=(h, w), im_file=str(image), segments=segments))
        items, ratios = [], []
        for label in labels:
            h, w = label["shape"]
            image_areas = []
            for i, poly in enumerate(label["segments"]):
                contour = (np.asarray(poly, np.float32) * np.array([w, h], np.float32) * (640 / max(w, h))).reshape(
                    -1, 1, 2
                )
                area = cv2.contourArea(contour)
                hull = cv2.contourArea(cv2.convexHull(contour))
                image_areas.append(area)
                items.append(
                    dict(
                        image=Path(label["im_file"]).name,
                        instance=i,
                        area640_polygon=area,
                        solidity=area / hull if hull > 0 else None,
                    )
                )
            positive = [a for a in image_areas if a > 0]
            if len(positive) > 1:
                ratios.append((max(positive) / min(positive)) ** 0.5)
        area = np.asarray([r["area640_polygon"] for r in items])
        solid = np.asarray([r["solidity"] for r in items if r["solidity"] is not None])
        result[split] = dict(
            images=len(labels),
            instances=len(items),
            polygon_area_lt256=int((area < 256).sum()),
            solidity_lt08=int((solid < 0.8).sum()),
            solidity_lt09=int((solid < 0.9).sum()),
            median_solidity=float(np.median(solid)),
            image_diameter_ratio_p50=float(np.median(ratios)),
            image_diameter_ratio_p95=float(np.quantile(ratios, 0.95)),
            tiny_examples=sorted(items, key=lambda r: r["area640_polygon"])[:5],
            concave_examples=sorted([r for r in items if r["solidity"] is not None], key=lambda r: r["solidity"])[:5],
        )
    result["limits"] = (
        "Continuous visible-polygon proxies, NOT raster tiny bins/COCO APs, and not proof of occlusion cause. "
        "Shape priors must not convexify ground truth. Test labels describe dataset only; no test prediction/tuning."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
