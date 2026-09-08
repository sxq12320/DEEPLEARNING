"""Original-image paired evaluation: single pass vs full image + four RGB slices.

SAHI-inspired integration, not a claim of official SAHI reproduction. Predictions
are returned to one common image coordinate system, globally NMS'd and counted
once. We keep each selected mask, never union touching fruit masks. GT is used
only AFTER prediction. A fixed 640-long-side mask raster is shared by both arms;
its AP must not be mixed with historical stride4-validator AP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from citrus_slicing import read_polygons, source_files, view_windows
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths, polygons2masks_overlap
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import scale_masks


def raster_shape(h, w, long_side):
    scale = long_side / max(h, w)
    return max(1, round(h * scale)), max(1, round(w * scale))


def predict_view(model, image, window, eval_shape, device, imgsz=640, conf=0.001, iou=0.7, max_det=300):
    x0, y0, x1, y1 = window
    h, w = image.shape[:2]
    eh, ew = eval_shape
    gx0, gx1 = round(x0 * ew / w), round(x1 * ew / w)
    gy0, gy1 = round(y0 * eh / h), round(y1 * eh / h)
    result = model.predict(image[y0:y1, x0:x1], imgsz=imgsz, device=device, conf=conf, iou=iou,
                           max_det=max_det, verbose=False, save=False, retina_masks=False)[0]
    if result.masks is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
    # Strip the crop's letterbox, then map to the same full-image evaluation grid.
    masks = scale_masks(result.masks.data[None].float(), (gy1 - gy0, gx1 - gx0))[0].gt(0.5).cpu().numpy()
    candidates = []
    for box, score, cls, mask in zip(boxes, result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), masks):
        yy, xx = np.nonzero(mask)
        if len(xx):
            left, top, right, bottom = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
            patch = mask[top:bottom, left:right].copy()
            offset = (gx0 + int(left), gy0 + int(top))
        else:
            patch, offset = np.zeros((0, 0), dtype=bool), (0, 0)
        candidates.append(dict(box=box, score=score, cls=int(cls), patch=patch, offset=offset))
    return candidates


def merge_candidates(candidates, shape, iou=0.7, max_det=300):
    """Class-aware box NMS; no mask union or GT-dependent border filtering."""
    if not candidates:
        return dict(bboxes=torch.zeros(0, 4), conf=torch.zeros(0), cls=torch.zeros(0),
                    masks=np.zeros((0, *shape), dtype=bool))
    boxes = torch.tensor(np.asarray([c["box"] for c in candidates]), dtype=torch.float32)
    scores = torch.tensor([c["score"] for c in candidates])
    classes = torch.tensor([c["cls"] for c in candidates], dtype=torch.float32)
    span = float(boxes.max()) + 1
    keep = TorchNMS.nms(boxes + classes[:, None] * span, scores, iou)[:max_det]
    masks = np.zeros((len(keep), *shape), dtype=bool)
    for i, k in enumerate(keep.tolist()):
        item = candidates[k]
        x, y = item["offset"]
        ph, pw = item["patch"].shape
        masks[i, y:y + ph, x:x + pw] = item["patch"]
    return dict(bboxes=boxes[keep], conf=scores[keep], cls=classes[keep], masks=masks)


def ground_truth(file, original_shape, shape):
    rows = read_polygons(img2label_paths([str(file)])[0])
    h, w = original_shape
    eh, ew = shape
    if not rows:
        return dict(cls=torch.zeros(0), boxes=torch.zeros(0, 4), masks=np.zeros((0, eh, ew), dtype=bool))
    polygons = [p * [ew, eh] for _, p in rows]
    overlap, order = polygons2masks_overlap(shape, polygons, downsample_ratio=1)
    masks = overlap[None] == np.arange(1, len(rows) + 1)[:, None, None]
    boxes = np.asarray([np.r_[p.min(0), p.max(0)] * [w, h, w, h] for _, p in rows])[order]
    classes = np.asarray([c for c, _ in rows])[order]
    return dict(cls=torch.tensor(classes, dtype=torch.float32), boxes=torch.tensor(boxes, dtype=torch.float32),
                masks=masks)


def mask_overlaps(gt, pred, chunk=32):
    """Avoid converting all full-image prediction masks to FP32 simultaneously."""
    output = torch.zeros(len(gt), len(pred))
    if not len(gt) or not len(pred):
        return output
    target = torch.from_numpy(gt).flatten(1).float()
    for start in range(0, len(pred), chunk):
        output[:, start:start + chunk] = mask_iou(target, torch.from_numpy(pred[start:start + chunk]).flatten(1).float())
    return output


def _evaluate(weights, data, output, device="cpu", fraction=0.6, limit=0, mask_long_side=640, guide_checkpoint=None):
    if mask_long_side < 32 or limit < 0:
        raise ValueError("mask_long_side>=32 and limit>=0 required")
    cv2.setNumThreads(1)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Will not overwrite evaluation: {output}")
    output.mkdir(parents=True)
    config, files = source_files(data, "val")
    if limit:
        files = files[:limit]  # explicitly marked exploratory subset, never a full-result claim
    model = YOLO(str(weights))
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=device, imgsz=640, verbose=False)
    matcher = SegmentationValidator(args=dict(plots=False))
    guide = None
    if guide_checkpoint is not None:
        from citrus_crop_guide import CropGuide

        guide = CropGuide(guide_checkpoint)
        guide.heatmap(np.zeros((640, 640, 3), dtype=np.uint8))
    modes = ("global", "sliced") if guide is None else ("global", "sliced", "guided")
    metrics = {mode: SegmentMetrics(config["names"]) for mode in modes}
    timing = {mode: [] for mode in metrics}
    records = []
    guide_windows, guide_seconds = [], []
    for index, file in enumerate(files):
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError(f"Unreadable validation image: {file}")
        shape = raster_shape(*image.shape[:2], mask_long_side)
        windows = view_windows(*image.shape[:2], fraction)
        start = time.perf_counter()
        global_candidates = predict_view(model, image, windows[0], shape, device)
        global_seconds = time.perf_counter() - start
        start = time.perf_counter()
        candidates = list(global_candidates)
        for window in windows[1:]:
            candidates.extend(predict_view(model, image, window, shape, device))
        sliced_seconds = global_seconds + time.perf_counter() - start
        predictions_by_mode = [("global", global_candidates, global_seconds), ("sliced", candidates, sliced_seconds)]
        if guide is not None:
            start = time.perf_counter()
            selected = guide.windows(image, fraction)
            guide_seconds.append(time.perf_counter() - start)
            guided_candidates = list(global_candidates)
            for window in selected[1:]:
                guided_candidates.extend(predict_view(model, image, window, shape, device))
            predictions_by_mode.append(("guided", guided_candidates, global_seconds + time.perf_counter() - start))
            guide_windows.append(dict(image=str(file), windows=selected))
        gt = ground_truth(file, image.shape[:2], shape)
        area640 = gt["masks"].sum((1, 2)) * (640 / mask_long_side) ** 2
        for mode, predictions, seconds in predictions_by_mode:
            start = time.perf_counter()
            pred = merge_candidates(predictions, shape)
            timing[mode].append(seconds + time.perf_counter() - start)
            overlaps = mask_overlaps(gt["masks"], pred["masks"])
            correct_mask = matcher.match_predictions(pred["cls"], gt["cls"], overlaps).numpy()
            correct_box = matcher.match_predictions(pred["cls"], gt["cls"], box_iou(gt["boxes"], pred["bboxes"])).numpy()
            metrics[mode].update_stats(dict(
                tp=correct_box, tp_m=correct_mask, conf=pred["conf"].numpy(), pred_cls=pred["cls"].numpy(),
                target_cls=gt["cls"].numpy(), target_img=np.unique(gt["cls"].numpy()), im_name=file.name,
            ))
            # Fixed-threshold per-GT recall; explicitly NOT subset AP.
            from scripts.review_sage_v4r import greedy_pairs

            kept = pred["conf"] >= 0.25
            pairs = greedy_pairs(overlaps[:, kept].numpy(), 0.5)
            matched = set(pairs[:, 0].tolist())
            for i in range(len(area640)):
                records.append(dict(image=file.name, mode=mode, area640=float(area640[i]), matched25=i in matched))
        if index % 10 == 0 or index == len(files) - 1:
            print(f"PAIRED GLOBAL/SLICED: {index + 1}/{len(files)} original images", flush=True)
    summary = {}
    for mode, metric in metrics.items():
        metric.process(plot=False)
        rows = [r for r in records if r["mode"] == mode]
        tiny = [r for r in rows if r["area640"] < 256]
        summary[mode] = dict(
            **metric.results_dict, images=len(files), instances=len(rows),
            tiny_instances=len(tiny), tiny_recall25=sum(r["matched25"] for r in tiny) / len(tiny) if tiny else None,
            inference_fusion_median_ms=float(np.median(timing[mode]) * 1000),
            forward_passes_per_image=1 if mode == "global" else 5,
            additional_guide_passes_per_image=1 if mode == "guided" else 0,
        )
    payload = dict(
        weights=str(weights), data=str(data), summary=summary, records=records, timing_seconds=timing,
        guided_windows=guide_windows, guide_seconds=guide_seconds,
        protocol=dict(mask_long_side=mask_long_side, model_imgsz=640, conf=0.001, nms_iou=0.7, max_det=300,
                      tile_fraction=fraction, limited_subset=limit, device=str(device),
                      guide_checkpoint=str(guide_checkpoint) if guide else None,
                      guide_sha256=guide.sha256 if guide else None,
                      fusion="class-aware box NMS; selected masks, no union; no GT in tile selection"),
        limits="AP on one complete original-image coordinate system with a common mask raster. Not historical "
        "stride4-validator AP, not native-original-resolution COCO AP. Time excludes disk image reading, GT, metrics; "
        "includes model prediction, mask projection and final fusion. Tiling is extra input compute, not free GFLOPs.",
    )
    (output / "paired_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def evaluate(weights, data, output, device="cpu", fraction=0.6, limit=0, mask_long_side=640, guide_checkpoint=None):
    """Bound CPU mask-metric threads even when invoked after GPU training."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        return _evaluate(weights, data, output, device, fraction, limit, mask_long_side, guide_checkpoint)
    finally:
        torch.set_num_threads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mask-long-side", type=int, default=640)
    parser.add_argument("--guide-checkpoint", default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.weights, args.data, args.output, args.device, args.fraction, args.limit, args.mask_long_side,
             args.guide_checkpoint)


if __name__ == "__main__":
    main()
