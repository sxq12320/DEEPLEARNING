"""Global-view anti-aliasing control for the 2026-10-05 diagnosis (read-only, no training).

The existing ``global`` mode letterboxes the original image with INTER_LINEAR (the same interpolation as training).
``global_area`` feeds the INTER_AREA-downsampled 640 raster (re-expanded so window/coordinate mapping is unchanged;
LetterBox then lands back on the identical 640 grid). Everything else -- weights, NMS, EV6 decoding, matching and
GT -- reuses ``diagnose_citrus_20261005`` functions so the two modes are directly paired.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_citrus_20261005 import (  # noqa: E402
    classify, ground_truth, infer_view, metric_rows, properties, raster_shape, source_files, traced_merge,
    uniform_windows)
from ultralytics import YOLO  # noqa: E402

MODES = ("global_linear", "global_area")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    _, files = source_files(args.data, "val")
    files = files[:args.limit] if args.limit else files
    groups = {}
    if args.group_manifest:
        with args.group_manifest.open(encoding="utf-8-sig") as stream:
            groups = {r["stem"]: r["group_id"] for r in csv.DictReader(stream)}
    (args.output / "manifest.json").write_text(json.dumps(dict(
        weights=str(args.weights.resolve()), sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        data=str(args.data.resolve()), images=len(files),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2), encoding="utf-8")
    model = YOLO(str(args.weights)).model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rows, records = {m: [] for m in MODES}, []
    for number, file in enumerate(files):
        image = cv2.imread(str(file))
        h, w = image.shape[:2]
        shape = raster_shape(h, w, 640)
        window = uniform_windows(h, w, 0.4)[0]
        gt = ground_truth(file, (h, w), shape)
        props = properties(gt, image, shape)
        low_area = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        # Nearest re-expansion keeps the AREA samples exact; LetterBox's LINEAR resize back to 640 then
        # samples (almost exactly) the original AREA pixels, so the only change is the low-pass filter.
        area_full = cv2.resize(low_area, (w, h), interpolation=cv2.INTER_NEAREST)
        for mode, source in (("global_linear", image), ("global_area", area_full)):
            view = infer_view(model, source, window, (h, w), shape, 0)
            final, masks, removed = traced_merge(view["candidates"], shape, True)
            out, stats, *_ = classify(gt, props, [view], final, masks, removed, 0.25)
            for r in out:
                r.update(image=file.name, mode=mode, group=groups.get(file.stem, file.name))
            records.extend(out)
            rows[mode].append(stats)
        print(f"{number + 1}/{len(files)} {file.name}", flush=True)
    summary = dict(evidence="旧划分探索", images=len(files),
                   metrics={m: metric_rows(r) for m, r in rows.items()},
                   stages={m: dict(Counter(r["stage"] for r in records if r["mode"] == m)) for m in MODES})
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in records))))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
