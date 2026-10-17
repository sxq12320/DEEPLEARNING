"""Shared mask validation, separate monitoring grid, and original-image checkpoint selection."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from common import save_json
from coco_utils import PreparedCocoEvaluation, evaluate_predictions, prediction_from_mask
from registry import COMMON_VALIDATION

VALIDATION_PROTOCOL = COMMON_VALIDATION


def checked_protocol(value):
    """Reject partial/custom rules that would accidentally mix incomparable metrics."""
    if value != VALIDATION_PROTOCOL:
        raise ValueError("Unknown validation protocol. Change its version AND implement/tests before changing rules.")
    return copy.deepcopy(value)


def original_due(epoch, epochs, protocol):
    if not 1 <= epoch <= epochs:
        raise ValueError("Validation epoch must be one-based and within the training budget")
    return epoch in (1, epochs) or epoch % protocol["original_interval"] == 0


def prepare_quick_view(prepared, protocol):
    """Cache a validation-only letterbox view; retain every image, ID, and annotation."""
    from pycocotools import mask as mask_utils

    protocol = checked_protocol(protocol)
    prepared = Path(prepared)
    source = prepared / "coco/annotations/instances_val.json"
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    root = prepared / "validation_quick640"
    marker = root / "view.json"
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["annotation_sha256"] != fingerprint or saved["protocol"] != protocol:
            raise RuntimeError("Quick validation cache conflicts with the dataset/protocol; use a NEW PROJECT")
        if not (root / "instances_val.json").is_file() or any(
                not (root / "images" / name).is_file() for name in saved["files"]):
            raise RuntimeError("Incomplete quick validation cache; use a NEW PROJECT")
        return root
    coco = json.loads(source.read_text(encoding="utf-8"))
    size = protocol["quick_size"]
    transforms = {}
    (root / "images").mkdir(parents=True, exist_ok=True)
    for record in coco["images"]:
        w, h = record["width"], record["height"]
        ratio = min(size / w, size / h)
        nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
        left, top = (size - nw) // 2, (size - nh) // 2
        transforms[record["id"]] = (nw / w, nh / h, left, top)
        with Image.open(prepared / "coco/images/val" / record["file_name"]) as image:
            canvas = Image.new("RGB", (size, size), (114, 114, 114))
            canvas.paste(image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
            # Lossless PNG encoding even when the original filename has a JPEG extension;
            # PIL/OpenCV identify by the header, avoiding an extra compression confound.
            canvas.save(root / "images" / record["file_name"], format="PNG")
        record.update(width=size, height=size)
    zero_area = []
    for ann in coco["annotations"]:
        sx, sy, left, top = transforms[ann["image_id"]]
        if not isinstance(ann["segmentation"], list):
            raise ValueError("Prepared citrus COCO view must contain original polygon annotations")
        ann["segmentation"] = [[
            v * (sx if i % 2 == 0 else sy) + (left if i % 2 == 0 else top) for i, v in enumerate(polygon)
        ] for polygon in ann["segmentation"]]
        x, y, w, h = ann["bbox"]
        ann["bbox"] = [x * sx + left, y * sy + top, w * sx, h * sy]
        rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], size, size))
        ann["area"] = float(mask_utils.area(rle))
        if ann["area"] == 0:
            zero_area.append(ann["id"])
    save_json(root / "instances_val.json", coco)
    save_json(
        marker,
        dict(
            annotation_sha256=fingerprint,
            protocol=protocol,
            files=[r["file_name"] for r in coco["images"]],
            images=len(coco["images"]),
            instances=len(coco["annotations"]),
            zero_area_instance_ids=zero_area,
            note="Monitoring only. Tiny masks can disappear on the grid; original annotations are unchanged.",
        ))
    return root


def cap_predictions(predictions, limit=100, score_min=0.001):
    """Same score threshold and stable per-image/category cap for AP and P/R."""
    groups = {}
    for prediction in predictions:
        if not math.isfinite(float(prediction["score"])):
            raise ValueError("Non-finite prediction score")
        if float(prediction["score"]) >= score_min:
            key = (prediction["image_id"], prediction["category_id"])
            groups.setdefault(key, []).append(prediction)
    return [p for values in groups.values() for p in sorted(values, key=lambda p: -float(p["score"]))[:limit]]


@contextlib.contextmanager
def validation_state(model):
    """Do not consume training RNG, alter per-module modes, or enable autocast during validation."""
    import torch
    from inference_runtime import inference_runtime_scope

    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Common validation requires FP32 master weights; refusing to mutate training weights")
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        model.eval()
        with inference_runtime_scope(), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@lru_cache(maxsize=2)
def _cached_evaluator(annotation_path, modified_ns, size):
    """Keep only quick/original GT metadata and RLE, never images or predicted masks."""
    return PreparedCocoEvaluation(Path(annotation_path))


def prepared_evaluator(annotation):
    """Invalidate cached annotations on ordinary file replacement or modification."""
    path = Path(annotation).resolve()
    stat = path.stat()
    return _cached_evaluator(str(path), stat.st_mtime_ns, stat.st_size)


def _selected_masks(record, masks, scores, labels, categories, protocol):
    """Apply the existing stable per-category cap BEFORE expensive RLE encoding."""
    if not (len(masks) == len(scores) == len(labels)):
        raise ValueError("Masks/scores/category IDs have mismatched lengths")
    groups = {}
    for index, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Non-finite prediction score")
        if score < protocol["score_min"]:
            continue
        label = int(label)
        # Validate even candidates outside the cap, as the previous implementation did.
        if label not in categories or np.shape(mask) != (record["height"], record["width"]):
            raise ValueError("Mask coordinates or category ID violate the common validation contract")
        groups.setdefault(label, []).append((index, score))
    for label, values in groups.items():
        for index, score in sorted(values, key=lambda value: -value[1])[:protocol["max_dets"]]:
            yield prediction_from_mask(record["id"], label, score, masks[index])


def collect_metrics(annotation, image_root, predictor, protocol, profile_images=0):
    """Preserve common matching/AP while prefetching CPU input and reusing immutable GT.

    A positive profile_images times only the largest original images, with exactly
    the same predictor, score floor, cap and encoding. It never reports subset AP.
    Timings are wall times; input preparation can overlap inference and must not
    be added to the total a second time.
    """
    if not isinstance(profile_images, int) or profile_images < 0:
        raise ValueError("profile_images must be a nonnegative integer")
    begin = time.perf_counter()
    if profile_images:
        # A speed-only probe needs image metadata, not a full-val GT rasterization.
        metadata = json.loads(Path(annotation).read_text(encoding="utf-8"))
        records = metadata["images"]
        categories = {item["id"] for item in metadata["categories"]}
        evaluator = None
        records = sorted(records, key=lambda item: item["height"] * item["width"], reverse=True)[:profile_images]
    else:
        evaluator = prepared_evaluator(annotation)
        records, categories = list(evaluator.images), evaluator.category_ids
    setup_seconds = time.perf_counter() - begin
    paths = [Path(image_root) / record["file_name"] for record in records]
    results = iter(predictor.iter_predictions(paths)) if hasattr(predictor, "iter_predictions") else (
        predictor(path) for path in paths)
    predictions, per_image = [], []
    scope = "quick640" if "validation_quick640" in Path(annotation).parts else "original"
    print(f"COMMON VAL START scope={scope} images={len(records)} profile_only={bool(profile_images)}", flush=True)
    try:
        for index, record in enumerate(records, 1):
            started = time.perf_counter()
            try:
                masks, scores, labels = next(results)
            except StopIteration as exc:
                raise ValueError("Predictor returned fewer images than requested") from exc
            predict_seconds = time.perf_counter() - started
            started = time.perf_counter()
            encoded = list(_selected_masks(record, masks, scores, labels, categories, protocol))
            predictions.extend(encoded)
            encode_seconds = time.perf_counter() - started
            detail = dict(image_id=record["id"], height=record["height"], width=record["width"],
                          prediction_count=len(encoded), predict_seconds=predict_seconds, encode_seconds=encode_seconds)
            detail.update(getattr(predictor, "last_timing", {}))
            per_image.append(detail)
            # Do not keep the previous image's huge dense masks while requesting the next.
            del masks, scores, labels, encoded
            if profile_images or index == 1 or index % 25 == 0 or index == len(records):
                print(f"COMMON VAL {scope} {index}/{len(records)} "
                      f"predict={predict_seconds:.3f}s encode={encode_seconds:.3f}s", flush=True)
        try:
            next(results)
        except StopIteration:
            pass
        else:
            raise ValueError("Predictor returned more images than requested")
    finally:
        close = getattr(results, "close", None)
        if close is not None:
            close()
    started = time.perf_counter()
    if profile_images:
        metrics = dict(prediction_count=len(predictions), profile_only=True,
                       selection="largest original images; no AP/checkpoint selection")
    else:
        metrics = evaluate_predictions(
            Path(annotation), predictions, evaluate_bbox=False,
            pr_iou_threshold=protocol["pr_iou"], pr_score_threshold=protocol["pr_score"],
            prepared_evaluator=evaluator)
    evaluation_seconds = time.perf_counter() - started
    timing = dict(setup_seconds=setup_seconds,
                  predict_seconds=sum(item["predict_seconds"] for item in per_image),
                  encode_seconds=sum(item["encode_seconds"] for item in per_image),
                  evaluation_seconds=evaluation_seconds,
                  input_wait_seconds=sum(item.get("input_wait_seconds", 0.0) for item in per_image),
                  note="Predict includes input wait, H2D, inference, decoding and D2H; preparation overlaps it.")
    metrics.update(images=len(records), seconds=time.perf_counter() - begin, timing=timing, per_image_timing=per_image,
                   evaluation_protocol=protocol["id"], pr_max_dets_per_image_category=protocol["max_dets"],
                   validation_engine="prefetch_cached_coco_v2_20261007")
    print(f"COMMON VAL DONE {scope} total={metrics['seconds']:.3f}s "
          f"predict={timing['predict_seconds']:.3f}s encode={timing['encode_seconds']:.3f}s "
          f"COCO={evaluation_seconds:.3f}s", flush=True)
    return metrics


class CommonValidation:
    """Select checkpoints only on scheduled ORIGINAL-image Mask AP50-95; ties retain earlier epochs."""

    def __init__(self, job, prepared, run_dir):
        self.job, self.prepared, self.run_dir = job, Path(prepared), Path(run_dir)
        self.protocol = checked_protocol(job["validation"])
        self.quick_root = prepare_quick_view(prepared, self.protocol)
        self.root = self.run_dir / "validation_common"
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "best.json"
        self.best = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
        if self.best is not None and self.best["protocol"] != self.protocol:
            raise RuntimeError("Checkpoint selection protocol changed")

    def run(self, epoch, model, predictor, save_best, weight_policy, eligible=True):
        with validation_state(model):
            quick = collect_metrics(self.quick_root / "instances_val.json", self.quick_root / "images", predictor,
                                    self.protocol)
            original = None
            if eligible and original_due(epoch, self.job["epochs"], self.protocol):
                original = collect_metrics(self.prepared / "coco/annotations/instances_val.json",
                                           self.prepared / "coco/images/val", predictor, self.protocol)
        selected = False
        if original is not None:
            score = original["mask_ap_50_95"]
            if not math.isfinite(score) or score < 0:
                raise ValueError("Original validation Mask AP is undefined; checkpoint selection refused")
            if self.best is None or score > self.best["mask_ap_50_95"]:
                checkpoint = Path(save_best(epoch))
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Selected checkpoint was not saved: {checkpoint}")
                self.best = dict(epoch=epoch,
                                 mask_ap_50_95=score,
                                 checkpoint=str(checkpoint),
                                 weight_policy=weight_policy,
                                 protocol=self.protocol)
                save_json(self.root / "best.json", self.best)
                selected = True
        report = dict(epoch=epoch,
                      quick=quick,
                      original=original,
                      selected=selected,
                      weight_policy=weight_policy,
                      selection_eligible=eligible,
                      protocol=self.protocol)
        save_json(self.root / f"epoch_{epoch:04d}_{weight_policy}.json", report)
        row = dict(epoch=epoch,
                   weight_policy=weight_policy,
                   selected=int(selected),
                   quick_mask_AP=quick["mask_ap_50_95"],
                   quick_mask_AP50=quick["mask_ap_50"],
                   quick_R=quick["mask_recall"],
                   quick_seconds=quick["seconds"],
                   original_mask_AP=original["mask_ap_50_95"] if original else "",
                   original_mask_AP50=original["mask_ap_50"] if original else "",
                   original_R=original["mask_recall"] if original else "",
                   original_seconds=original["seconds"] if original else "")
        csv_path = self.root / "epochs.csv"
        exists = csv_path.is_file()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"COMMON VAL epoch={epoch} role={weight_policy} quick_AP={quick['mask_ap_50_95']:.4f} "
            f"original_AP={original['mask_ap_50_95'] if original else 'not scheduled'} best={selected}",
            flush=True)
        return report

    def checkpoint(self):
        if self.best is None or not Path(self.best["checkpoint"]).is_file():
            raise RuntimeError("No original-validation-selected common checkpoint was produced")
        return Path(self.best["checkpoint"])
