"""J V3 privileged teacher: mine candidate missing fruit on TRAINING sources from original pixels.

The teacher sees each training image as the global view plus the 3x3 fine (.4) windows cropped from ORIGINAL pixels
(E V5 uniform windows, mask-aware cross-view merge with border weight .5). This is information the 640 student never
receives at test time. A ``global`` teacher mode (single 640 pass, no extra pixels) is the matched control.

Instances that overlap existing annotation (mask IoU >= .5, or >= .5 of either area) are excluded: they are labelled
fruit, partial predictions or merges, not missing labels. The rest are tiered:

* high: score >= TAU_HIGH and corroborated by an independent second view (the global pass or the horizontally
  flipped pass for the fine teacher; the flipped pass for the global teacher);
* mid: any other instance with score >= TAU_LOW.

Thresholds are provisional engineering settings. The claimed human-review provenance
of the earlier 100-row audit is UNCONFIRMED; it cannot calibrate pseudo-label precision.
They are recorded with every output, never tuned on val/test here. Val/test sources are never read.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from citrus_e_v5_slicing import uniform_windows
from citrus_slicing import source_files
from eval_citrus_e_v5 import add_view_metadata, merge_views
from eval_citrus_sliced import ground_truth, merge_candidates, predict_view, raster_shape

PROTOCOL = "citrus_j3_teacher_mining_v1_20261007"
TEACHER_MODES = ("fine", "global")
TAU_HIGH = 0.6
TAU_LOW = 0.25
MATCH_IOU = 0.5
MATCH_COVER = 0.5
CONF_FLOOR = 0.1
RASTER = 800  # >= 2 raster px per stride-4 prototype px of a fine (.4) window on a 3072 px source


def teacher_pass(model, image, device, mode, raster=RASTER, conf=CONF_FLOOR):
    """Merged teacher instances on a common raster: dict(conf (n,), masks (n,eh,ew) bool, bboxes (n,4) source xyxy)."""
    h, w = image.shape[:2]
    shape = raster_shape(h, w, raster)
    windows = uniform_windows(h, w, 0.4)
    full = add_view_metadata(predict_view(model, image, windows[0], shape, device, conf=conf), windows[0], (h, w), 0)
    if mode == "global":
        pred = merge_candidates(full, shape)
    else:
        candidates = list(full)
        for view, window in enumerate(windows[1:], 1):
            candidates.extend(
                add_view_metadata(predict_view(model, image, window, shape, device, conf=conf), window, (h, w), view)
            )
        pred = merge_views(candidates, shape, border_weight=0.5)
    return dict(conf=np.asarray(pred["conf"], dtype=np.float64), masks=np.asarray(pred["masks"], dtype=bool),
                bboxes=np.asarray(pred["bboxes"], dtype=np.float64))


def _extents(masks):
    out = np.zeros((len(masks), 4), dtype=np.int64)  # y0, y1, x0, x1 (exclusive); empty -> zeros
    for i, m in enumerate(masks):
        ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        if len(ys):
            out[i] = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    return out


def overlap_matrix(a, b):
    """IoU, intersection/area(a), intersection/area(b) between two stacks of boolean masks (extent-pruned)."""
    iou, cover_a, cover_b = (np.zeros((len(a), len(b))) for _ in range(3))
    if not len(a) or not len(b):
        return iou, cover_a, cover_b
    ea, eb = _extents(a), _extents(b)
    area_a, area_b = a.reshape(len(a), -1).sum(1), b.reshape(len(b), -1).sum(1)
    for i in range(len(a)):
        for j in range(len(b)):
            y0, y1 = max(ea[i, 0], eb[j, 0]), min(ea[i, 1], eb[j, 1])
            x0, x1 = max(ea[i, 2], eb[j, 2]), min(ea[i, 3], eb[j, 3])
            if y0 >= y1 or x0 >= x1:
                continue
            inter = np.count_nonzero(a[i, y0:y1, x0:x1] & b[j, y0:y1, x0:x1])
            iou[i, j] = inter / max(area_a[i] + area_b[j] - inter, 1)
            cover_a[i, j], cover_b[i, j] = inter / max(area_a[i], 1), inter / max(area_b[j], 1)
    return iou, cover_a, cover_b


def assign_tiers(scores, masks, gt_masks, corroborating, tau_high=TAU_HIGH, tau_low=TAU_LOW):
    """Pure tiering rule (unit-tested).

    Args:
        scores (np.ndarray): (n,) merged teacher confidences.
        masks (np.ndarray): (n,h,w) teacher masks.
        gt_masks (np.ndarray): (g,h,w) existing annotation masks on the same raster.
        corroborating (list[np.ndarray]): independent passes' masks (each (k,h,w)); IoU >= .5 counts as one vote.

    Returns:
        (list[dict]): per teacher instance: tier ('high'|'mid'|None), excluded_by_gt, votes, best_gt_iou.
    """
    iou, cover_pred, cover_gt = overlap_matrix(masks, gt_masks)
    agreement = [overlap_matrix(masks, other)[0] for other in corroborating]
    results = []
    for i, score in enumerate(scores):
        best = float(iou[i].max()) if iou.shape[1] else 0.0
        excluded = bool(iou.shape[1]) and bool(
            best >= MATCH_IOU or cover_pred[i].max() >= MATCH_COVER or cover_gt[i].max() >= MATCH_COVER
        )
        votes = 1 + sum(int(m.shape[1] > 0 and m[i].max() >= MATCH_IOU) for m in agreement)
        if excluded or score < tau_low or not masks[i].any():
            tier = None
        elif score >= tau_high and votes >= 2:
            tier = "high"
        else:
            tier = "mid"
        results.append(dict(tier=tier, excluded_by_gt=excluded, votes=votes, best_gt_iou=best))
    return results


def mask_polygon(mask, shape):
    """Largest outer contour of a raster mask as a normalized polygon (None if degenerate)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.75, True).reshape(-1, 2).astype(np.float64)
    if len(contour) < 3:
        return None
    eh, ew = shape
    return ((contour + 0.5) / [ew, eh]).clip(0, 1).round(6).tolist()


def _flip_back(pred, width):
    masks = pred["masks"][:, :, ::-1].copy()
    boxes = pred["bboxes"].copy()
    if len(boxes):
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return dict(conf=pred["conf"], masks=masks, bboxes=boxes)


def mine_image(model, file, device, mode, flip=True, raster=RASTER):
    """Teacher instances for one training source (excluded/tiered), plus counters."""
    image = cv2.imread(str(file))
    if image is None:
        raise ValueError(f"Unreadable training image: {file}")
    h, w = image.shape[:2]
    main = teacher_pass(model, image, device, mode, raster)
    corroborating = []
    if mode == "fine":
        corroborating.append(teacher_pass(model, image, device, "global", raster)["masks"])
    if flip:
        corroborating.append(_flip_back(teacher_pass(model, image[:, ::-1].copy(), device, mode, raster), w)["masks"])
    shape = main["masks"].shape[1:]
    gt = ground_truth(file, (h, w), shape)
    tiers = assign_tiers(main["conf"], main["masks"], gt["masks"], corroborating)
    instances, counters = [], dict(teacher=len(tiers), excluded_by_gt=0, below_low=0, high=0, mid=0, degenerate=0)
    for k, info in enumerate(tiers):
        if info["excluded_by_gt"]:
            counters["excluded_by_gt"] += 1
            continue
        if info["tier"] is None:
            counters["below_low"] += 1
            continue
        poly = mask_polygon(main["masks"][k], shape)
        if poly is None:
            counters["degenerate"] += 1
            continue
        counters[info["tier"]] += 1
        size640 = math.sqrt(float(main["masks"][k].sum())) * 640 / max(shape)
        instances.append(dict(tier=info["tier"], score=round(float(main["conf"][k]), 5), votes=info["votes"],
                              size640=round(size640, 2), best_gt_iou=round(info["best_gt_iou"], 4), poly=poly))
    return dict(shape=[h, w], gt=int(len(gt["masks"])), instances=instances), counters


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mine(weights, data, output, device="cpu", mode="fine", flip=True, raster=RASTER, limit=0):
    """Write ``output/pseudo.json`` for every TRAINING source; reuse only an identical completed record."""
    from ultralytics import YOLO

    if mode not in TEACHER_MODES:
        raise ValueError(f"mode must be one of {TEACHER_MODES}")
    output = Path(output)
    _, files = source_files(data, "train")
    held_out = set()
    for split in ("val", "test"):
        try:
            held_out |= set(source_files(data, split)[1])
        except (KeyError, ValueError, TypeError):
            pass
    if set(files) & held_out:
        raise ValueError("Training sources overlap val/test; refusing to mine")
    files = files[:limit] if limit else files
    import torch

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    teacher_train_data = str((ckpt.get("train_args") or {}).get("data"))
    del ckpt
    record = dict(protocol=PROTOCOL, weights=str(Path(weights).resolve()), weights_sha256=_sha256(weights),
                  teacher_train_data=teacher_train_data, data=str(Path(data).resolve()), sources=len(files),
                  sources_sha256=hashlib.sha256("\n".join(map(str, files)).encode()).hexdigest(), mode=mode,
                  flip=bool(flip), raster=int(raster), conf_floor=CONF_FLOOR, tau_high=TAU_HIGH, tau_low=TAU_LOW,
                  match_iou=MATCH_IOU, match_cover=MATCH_COVER, code_sha256=_sha256(__file__))
    done = output / "mining_record.json"
    if done.is_file():
        if json.loads(done.read_text(encoding="utf-8")) == record and (output / "pseudo.json").is_file():
            print(f"J3 TEACHER reuse {output}", flush=True)
            return output / "pseudo.json"
        raise FileExistsError(f"Different teacher output already at {output}; use a new directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Partial teacher output at {output}; remove it yourself or choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    model.predict(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)
    images, totals, start = {}, {}, time.perf_counter()
    for i, file in enumerate(files):
        entry, counters = mine_image(model, file, device, mode, flip, raster)
        images[str(Path(file).resolve())] = entry
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + value
        if i % 25 == 0 or i + 1 == len(files):
            print(f"J3 TEACHER {mode}: {i + 1}/{len(files)} {totals} {time.perf_counter() - start:.0f}s", flush=True)
    sizes = [x["size640"] for e in images.values() for x in e["instances"] if x["tier"] == "high"]
    summary = dict(totals, gt=sum(e["gt"] for e in images.values()),
                   high_size640_quartiles=np.quantile(sizes, [0.25, 0.5, 0.75]).round(2).tolist() if sizes else None)
    (output / "pseudo.json").write_text(json.dumps(dict(record=record, summary=summary, images=images)),
                                        encoding="utf-8")
    (output / "mining_summary.json").write_text(json.dumps(dict(record=record, summary=summary), indent=2),
                                                encoding="utf-8")
    done.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("J3 TEACHER DONE:", json.dumps(summary), flush=True)
    return output / "pseudo.json"
