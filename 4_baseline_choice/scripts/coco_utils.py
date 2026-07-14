"""Shared COCO prediction serialization and evaluation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from baseline_common import save_json


def require_pycocotools():
    """Import pycocotools with an actionable error."""
    try:
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools is required. Install the requirements file for this environment.") from exc
    return mask_utils, COCO, COCOeval


def load_coco_image_index(annotation_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map COCO file names to image records."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    records = data.get("images", [])
    index = {str(record["file_name"]): record for record in records}
    if len(index) != len(records):
        raise ValueError(f"Duplicate file_name entries in {annotation_path}")
    return index


def resize_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor interpolation."""
    mask = np.asarray(mask)
    if mask.shape == (height, width):
        return mask.astype(bool)
    from PIL import Image

    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """Encode one binary mask as JSON-compatible COCO RLE."""
    mask_utils, _, _ = require_pycocotools()
    encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def mask_bbox(mask: np.ndarray) -> List[float]:
    """Return a COCO x/y/width/height box for a binary mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def prediction_from_mask(
    image_id: int,
    category_id: int,
    score: float,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Build one COCO instance prediction."""
    binary = np.asarray(mask, dtype=bool)
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "score": float(score),
        "bbox": mask_bbox(binary),
        "segmentation": encode_binary_mask(binary),
    }


def save_predictions(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    """Write COCO predictions."""
    save_json(path, list(predictions))


def _empty_metrics(prefix: str) -> Dict[str, float]:
    names = (
        "ap_50_95",
        "ap_50",
        "ap_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    return {f"{prefix}_{name}": 0.0 for name in names}


def _run_coco_eval(coco_gt: Any, coco_dt: Any, iou_type: str, prefix: str) -> Dict[str, float]:
    """Run one pycocotools evaluator and name its summary statistics."""
    _, _, COCOeval = require_pycocotools()
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "ap_50_95",
        "ap_50",
        "ap_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    return {f"{prefix}_{name}": float(value) for name, value in zip(names, evaluator.stats)}


def mask_prf(
    annotation_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> Dict[str, float]:
    """Compute greedy mask precision, recall, and F1 at fixed thresholds."""
    mask_utils, COCO, _ = require_pycocotools()
    coco_gt = COCO(str(annotation_path))
    ground_truth: MutableMapping[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    predicted: MutableMapping[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)

    for annotation in coco_gt.dataset.get("annotations", []):
        ground_truth[(int(annotation["image_id"]), int(annotation["category_id"]))].append(annotation)
    for prediction in predictions:
        if float(prediction["score"]) >= score_threshold:
            predicted[(int(prediction["image_id"]), int(prediction["category_id"]))].append(prediction)

    true_positive = false_positive = false_negative = 0
    for key in set(ground_truth) | set(predicted):
        gt_annotations = ground_truth.get(key, [])
        dt_annotations = sorted(predicted.get(key, []), key=lambda item: float(item["score"]), reverse=True)
        if not gt_annotations:
            false_positive += len(dt_annotations)
            continue
        if not dt_annotations:
            false_negative += len(gt_annotations)
            continue

        gt_rles = [coco_gt.annToRLE(annotation) for annotation in gt_annotations]
        dt_rles = [dict(annotation["segmentation"]) for annotation in dt_annotations]
        for rle in dt_rles:
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
        ious = mask_utils.iou(dt_rles, gt_rles, [int(annotation.get("iscrowd", 0)) for annotation in gt_annotations])
        matched_gt = set()
        for row in ious:
            candidates = np.argsort(row)[::-1]
            match = next(
                (int(index) for index in candidates if row[index] >= iou_threshold and int(index) not in matched_gt),
                None,
            )
            if match is None:
                false_positive += 1
            else:
                matched_gt.add(match)
                true_positive += 1
        false_negative += len(gt_annotations) - len(matched_gt)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_f1": f1,
        "pr_iou_threshold": iou_threshold,
        "pr_score_threshold": score_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def evaluate_predictions(
    annotation_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    output_path: Path | None = None,
    evaluate_bbox: bool = True,
    pr_iou_threshold: float = 0.5,
    pr_score_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Evaluate COCO predictions with a common protocol."""
    _, COCO, _ = require_pycocotools()
    coco_gt = COCO(str(annotation_path))
    metrics: Dict[str, Any] = {"prediction_count": len(predictions)}
    if predictions:
        coco_dt = coco_gt.loadRes(list(predictions))
        metrics.update(_run_coco_eval(coco_gt, coco_dt, "segm", "mask"))
        if evaluate_bbox:
            metrics.update(_run_coco_eval(coco_gt, coco_dt, "bbox", "box"))
    else:
        metrics.update(_empty_metrics("mask"))
        if evaluate_bbox:
            metrics.update(_empty_metrics("box"))
    metrics.update(mask_prf(annotation_path, predictions, pr_iou_threshold, pr_score_threshold))
    if output_path:
        save_json(output_path, metrics)
    return metrics
