"""Paired whole/fixed-slice/mask-aware fusion; one candidate cache, unchanged official AP.

SAHI motivates cross-view duplicate handling. Our mask suppression keeps one
visible instance (never unions touching fruits). Border confidence is a fixed
experimental prior, not a learned quality estimate or evidence of correctness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from citrus_slicing import source_files, view_windows
from eval_citrus_sliced import ground_truth, mask_overlaps, merge_candidates, predict_view, raster_shape
from scripts.review_sage_v4r import greedy_pairs
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics, box_iou


def add_view_metadata(candidates, window, original_shape, view):
    h, w = original_shape
    x0, y0, x1, y1 = window
    margin_x, margin_y = (x1 - x0) * 2 / 640, (y1 - y0) * 2 / 640
    for c in candidates:
        bx0, by0, bx1, by1 = c["box"]
        c["view"] = view
        c["internal_border"] = bool(
            (x0 > 0 and bx0 - x0 <= margin_x)
            or (y0 > 0 and by0 - y0 <= margin_y)
            or (x1 < w and x1 - bx1 <= margin_x)
            or (y1 < h and y1 - by1 <= margin_y)
        )
    return candidates


def patch_overlap(a, b):
    ax, ay = a["offset"]
    bx, by = b["offset"]
    ah, aw = a["patch"].shape
    bh, bw = b["patch"].shape
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if left >= right or top >= bottom:
        return 0.0, 0.0
    inter = np.count_nonzero(
        a["patch"][top - ay : bottom - ay, left - ax : right - ax]
        & b["patch"][top - by : bottom - by, left - bx : right - bx]
    )
    aa, bb = a["area"], b["area"]
    return inter / max(aa + bb - inter, 1), inter / max(min(aa, bb), 1)


def merge_views(candidates, shape, border_weight=1.0, mask_iou=0.5, fragment_ios=0.8, max_det=300):
    """Cross-view suppression; disjoint touching instances stay separate; GT-free."""
    ordered = []
    for c in candidates:
        score = c["score"] * (border_weight if c["internal_border"] else 1.0)
        ordered.append({**c, "score": score, "area": int(c["patch"].sum())})
    ordered.sort(key=lambda c: c["score"], reverse=True)
    keep = []
    for candidate in ordered:
        suppress = False
        for chosen in keep:
            if chosen["cls"] != candidate["cls"] or chosen["view"] == candidate["view"]:
                continue
            miou, ios = patch_overlap(chosen, candidate)
            if miou >= mask_iou or (
                ios >= fragment_ios and (chosen["internal_border"] or candidate["internal_border"])
            ):
                suppress = True
                break
        if not suppress:
            keep.append(candidate)
            if len(keep) >= max_det:
                break
    # The per-view detector already did box NMS. Avoid a second box-only NMS
    # that could discard disjoint touching masks after mask-aware selection.
    if not keep:
        return merge_candidates([], shape)
    masks = np.zeros((len(keep), *shape), bool)
    for i, c in enumerate(keep):
        x, y = c["offset"]
        h, w = c["patch"].shape
        masks[i, y : y + h, x : x + w] = c["patch"]
    return dict(
        bboxes=torch.tensor(np.asarray([c["box"] for c in keep]), dtype=torch.float32),
        conf=torch.tensor([c["score"] for c in keep]),
        cls=torch.tensor([c["cls"] for c in keep]),
        masks=masks,
    )


def evaluate(weights, data, output, device="cpu", limit=0, guide_checkpoint=None,
             quality_calibration=None, quality_floor=None):
    if quality_floor is not None and not 0 <= quality_floor <= 1:
        raise ValueError("quality_floor must be finite and in [0, 1]")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    previous = torch.get_num_threads()
    previous_cv_threads = cv2.getNumThreads()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    try:
        config, files = source_files(data, "val")
        files = files[:limit] if limit else files
        model = YOLO(str(weights))
        if quality_floor is not None:
            if not hasattr(model.model.model[-1], "quality_predictor"):
                raise ValueError("quality_floor requires an E V3/V4R mask-quality checkpoint")
            model.model.model[-1].quality_floor = float(quality_floor)
        if quality_calibration is not None:
            if not hasattr(model.model.model[-1], "quality_calibration"):
                raise ValueError("Quality toggle requires an E V3 quality-head checkpoint")
            model.model.model[-1].quality_calibration = bool(quality_calibration)
        model.predict(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)
        matcher = SegmentationValidator(args=dict(plots=False))
        guide = None
        modes = ["global", "fixed_box", "crossmask", "trustedmask"]
        if guide_checkpoint is not None:
            from citrus_crop_guide import CropGuide

            guide = CropGuide(guide_checkpoint)
            guide.heatmap(np.zeros((640, 640, 3), np.uint8))
            modes += ["guided_box", "guided_crossmask", "guided_trustedmask"]
        metrics = {m: SegmentMetrics(config["names"]) for m in modes}
        guide_windows = []
        records, timing, errors = (
            [],
            {m: [] for m in metrics},
            {m: dict(tp=0, duplicates=0, mask_or_localization=0, background=0) for m in metrics},
        )
        for i, file in enumerate(files):
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError(f"Unreadable validation image: {file}")
            shape = raster_shape(*image.shape[:2], 640)
            windows = view_windows(*image.shape[:2])
            start = time.perf_counter()
            full = add_view_metadata(
                predict_view(model, image, windows[0], shape, device), windows[0], image.shape[:2], 0
            )
            global_seconds = time.perf_counter() - start
            candidates = list(full)
            for view, window in enumerate(windows[1:], 1):
                candidates.extend(
                    add_view_metadata(predict_view(model, image, window, shape, device), window, image.shape[:2], view)
                )
            sliced_seconds = time.perf_counter() - start
            guided_candidates, guided_seconds = None, None
            if guide is not None:
                start = time.perf_counter()
                selected_windows = guide.windows(image, 0.6)
                guided_candidates = list(full)
                for view, window in enumerate(selected_windows[1:], 1):
                    guided_candidates.extend(
                        add_view_metadata(
                            predict_view(model, image, window, shape, device), window, image.shape[:2], view
                        )
                    )
                guided_seconds = global_seconds + time.perf_counter() - start
                guide_windows.append(dict(image=str(file), windows=selected_windows))
            gt = ground_truth(file, image.shape[:2], shape)
            for mode, metric in metrics.items():
                start = time.perf_counter()
                selected = full if mode == "global" else guided_candidates if mode.startswith("guided_") else candidates
                seconds = (
                    global_seconds
                    if mode == "global"
                    else guided_seconds
                    if mode.startswith("guided_")
                    else sliced_seconds
                )
                pred = (
                    merge_candidates(selected, shape)
                    if mode in {"global", "fixed_box", "guided_box"}
                    else merge_views(selected, shape, border_weight=0.5 if mode.endswith("trustedmask") else 1.0)
                )
                timing[mode].append(seconds + time.perf_counter() - start)
                overlap = mask_overlaps(gt["masks"], pred["masks"])
                boxes = box_iou(gt["boxes"], pred["bboxes"])
                tp_m = matcher.match_predictions(pred["cls"], gt["cls"], overlap).numpy()
                tp_b = matcher.match_predictions(pred["cls"], gt["cls"], boxes).numpy()
                metric.update_stats(
                    dict(
                        tp=tp_b,
                        tp_m=tp_m,
                        conf=pred["conf"].numpy(),
                        pred_cls=pred["cls"].numpy(),
                        target_cls=gt["cls"].numpy(),
                        target_img=np.unique(gt["cls"].numpy()),
                        im_name=file.name,
                    )
                )
                selected = pred["conf"] >= 0.25
                pairs = greedy_pairs(overlap[:, selected].numpy(), 0.5)
                matched = set(pairs[:, 0].tolist())
                pred_matched = set(pairs[:, 1].tolist())
                ov, bo = overlap[:, selected].numpy(), boxes[:, selected].numpy()
                for j in range(ov.shape[1]):
                    key = (
                        "tp"
                        if j in pred_matched
                        else "duplicates"
                        if ov[:, j].max(initial=0) >= 0.5
                        else "mask_or_localization"
                        if bo[:, j].max(initial=0) >= 0.5
                        else "background"
                    )
                    errors[mode][key] += 1
                for j, area in enumerate(gt["masks"].sum((1, 2))):
                    records.append(
                        dict(image=file.name, gt_index=j, mode=mode, area640=int(area), matched25=j in matched)
                    )
            if i % 20 == 0 or i == len(files) - 1:
                print(f"E V2 paired fusion diagnostic {i + 1}/{len(files)}", flush=True)
        summary = {}
        for mode, metric in metrics.items():
            metric.process(plot=False)
            rr = [r for r in records if r["mode"] == mode]
            tiny = [r for r in rr if r["area640"] < 256]
            summary[mode] = dict(
                **metric.results_dict,
                tiny_n=len(tiny),
                tiny_matched=sum(r["matched25"] for r in tiny),
                tiny_recall25=sum(r["matched25"] for r in tiny) / max(len(tiny), 1),
                median_ms=float(np.median(timing[mode]) * 1000),
                errors25=errors[mode],
            )
            counts = errors[mode]
            summary[mode]["all_precision25"] = counts["tp"] / max(sum(counts.values()), 1)
            summary[mode]["all_recall25"] = counts["tp"] / max(len(rr), 1)
            summary[mode]["mask_rmax_at_conf001"] = float(metric.seg.r_curve[:, 0].mean())
            # Keep the actual confidence sweep, not just a plotted zero-padded
            # PR envelope. This makes precision-constrained recall auditable.
            precision, recall = metric.seg.p_curve.mean(0), metric.seg.r_curve.mean(0)
            for target in (0.85, 0.90, 0.95):
                eligible = np.flatnonzero(precision >= target)
                index = eligible[np.argmax(recall[eligible])] if len(eligible) else None
                summary[mode][f"operating_p{int(target * 100)}"] = (
                    dict(
                        confidence=float(metric.seg.px[index]),
                        precision=float(precision[index]),
                        recall=float(recall[index]),
                    )
                    if index is not None
                    else None
                )
        payload = dict(
            weights=str(weights),
            data=str(data),
            summary=summary,
            records=records,
            guided_windows=guide_windows,
            confidence_curves={
                m: dict(
                    confidence=met.seg.px.tolist(), precision=met.seg.p_curve.tolist(), recall=met.seg.r_curve.tolist()
                )
                for m, met in metrics.items()
            },
            protocol=dict(
                imgsz=640,
                mask_raster=640,
                conf=0.001,
                iou=0.7,
                max_det=300,
                fraction=0.6,
                mask_iou=0.5,
                fragment_ios=0.8,
                trusted_border_weight=0.5,
                quality_calibration=getattr(model.model.model[-1], "quality_calibration", None),
                quality_floor=getattr(model.model.model[-1], "quality_floor", None),
                limit=limit,
                device=str(device),
                guide_checkpoint=str(guide_checkpoint) if guide else None,
                guide_sha256=guide.sha256 if guide else None,
            ),
            limits="Same frozen weights/inputs for all merge arms; exploratory validation ablation. Fixed-threshold error buckets are diagnostic, not semantic proof of leaf confusion. No GT in selection; no mask union. Common-raster AP is not official validator AP.",
        )
        (output / "paired_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        torch.set_num_threads(previous)
        cv2.setNumThreads(previous_cv_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("weights", "data", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--guide-checkpoint", default=None)
    parser.add_argument("--quality-floor", default=None, type=float)
    evaluate(**vars(parser.parse_args()))
