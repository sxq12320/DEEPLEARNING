"""Cached paired diagnostics and a separately labelled original-GT COCO report."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from citrus_e_v5_slicing import uniform_windows
from citrus_slicing import read_polygons, source_files
from eval_citrus_e_v5 import add_view_metadata, empirical_mask_pr, patch_overlap
from eval_citrus_sliced import ground_truth, mask_overlaps, merge_candidates, predict_view, raster_shape
from scripts.review_sage_v4r import greedy_pairs
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics, box_iou


def write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(str(temporary), str(path))


def save_candidates(path, candidates, timing):
    """No pickle/object arrays. Preserve candidate masks BEFORE GT is inspected."""
    metadata, arrays = [], []
    for item in candidates:
        row = {k: v for k, v in item.items() if k != "patch"}
        row["box"] = np.asarray(row["box"]).tolist()
        row["offset"] = list(row["offset"])
        row["patch_shape"] = list(item["patch"].shape)
        packed = np.packbits(item["patch"].ravel())
        row["packed_length"] = len(packed)
        metadata.append(row)
        arrays.append(packed)
    bits = np.concatenate(arrays) if arrays else np.zeros(0, np.uint8)
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temporary, bits=bits, metadata=np.array(json.dumps(metadata)), timing=np.array(timing))
    os.replace(str(temporary), str(path))


def load_candidates(path):
    with np.load(path, allow_pickle=False) as data:
        rows, bits, timing = json.loads(str(data["metadata"].item())), data["bits"], data["timing"].tolist()
        start = 0
        for row in rows:
            length, shape = row.pop("packed_length"), row.pop("patch_shape")
            size = int(np.prod(shape))
            row["patch"] = np.unpackbits(bits[start:start + length])[:size].astype(bool).reshape(shape)
            start += length
        if start != len(bits):
            raise ValueError("Malformed candidate cache")
    return rows, timing


def select_ids(candidates, border_weight=1.0, ios_rule="border", max_det=300):
    """Exactly one change per selection control; never use GT or mask unions."""
    if ios_rule not in {"none", "border", "always"}:
        raise ValueError(ios_rule)
    enriched = [dict(c, area=int(c["patch"].sum()), identity=i) for i, c in enumerate(candidates)]
    order = sorted(enriched, key=lambda c: c["score"] * (border_weight if c["internal_border"] else 1), reverse=True)
    keep = []
    for candidate in order:
        duplicate = False
        for chosen in keep:
            if chosen["cls"] != candidate["cls"] or chosen["view"] == candidate["view"]:
                continue
            iou, ios = patch_overlap(chosen, candidate)
            ios_enabled = ios_rule == "always" or (
                ios_rule == "border" and (chosen["internal_border"] or candidate["internal_border"])
            )
            if iou >= 0.5 or (ios_enabled and ios >= 0.8):
                duplicate = True
                break
        if not duplicate:
            keep.append(candidate)
            if len(keep) == max_det:
                break
    return [c["identity"] for c in keep]


def chosen_predictions(candidates, ids, shape, output_weight):
    masks = np.zeros((len(ids), *shape), bool)
    for position, index in enumerate(ids):
        item = candidates[index]
        x, y = item["offset"]
        h, w = item["patch"].shape
        masks[position, y:y + h, x:x + w] = item["patch"]
    return dict(
        bboxes=torch.tensor(np.asarray([candidates[i]["box"] for i in ids]).reshape(-1, 4), dtype=torch.float32),
        conf=torch.tensor([candidates[i]["score"] * (output_weight if candidates[i]["internal_border"] else 1)
                           for i in ids], dtype=torch.float32),
        cls=torch.tensor([candidates[i]["cls"] for i in ids], dtype=torch.float32), masks=masks,
    )


def iter_variants(candidates, shape, mechanisms=True):
    """Separate final-score changes from changes to the selected mask set."""
    start = time.perf_counter()
    yield "global", merge_candidates([c for c in candidates if c["view"] == 0], shape), start
    start = time.perf_counter()
    yield "fixed_box", merge_candidates(candidates, shape), start
    start = time.perf_counter()
    base_ids = select_ids(candidates)
    base_selection_seconds = time.perf_counter() - start
    yield "crossmask", chosen_predictions(candidates, base_ids, shape, 1.0), start
    start = time.perf_counter()
    trusted_ids = select_ids(candidates, 0.5)
    trusted_selection_seconds = time.perf_counter() - start
    yield "trustedmask", chosen_predictions(candidates, trusted_ids, shape, 0.5), start
    if mechanisms:
        for mode, weight, ids, seconds in (
            ("score_only", 0.5, base_ids, base_selection_seconds),
            ("selection_only", 1.0, trusted_ids, trusted_selection_seconds),
        ):
            start = time.perf_counter() - seconds
            yield mode, chosen_predictions(candidates, ids, shape, weight), start
        for mode, weight, rule in (
            ("iou_only", 1.0, "none"), ("unconditional_ios", 1.0, "always"),
            ("border025", 0.25, "border"), ("border075", 0.75, "border"),
        ):
            start = time.perf_counter()
            ids = select_ids(candidates, weight, rule)
            yield mode, chosen_predictions(candidates, ids, shape, weight), start


def variants(candidates, shape, mechanisms=True):
    """Small-case convenience for tests; production streams one mask array at a time."""
    return {mode: pred for mode, pred, _ in iter_variants(candidates, shape, mechanisms)}


def native_coco_gt(file, image, image_id, next_annotation):
    """Original-resolution polygon GT, no common-grid overlap-ID replacement."""
    from pycocotools import mask as mask_utils

    h, w = image.shape[:2]
    annotations = []
    label = img2label_paths([str(file)])[0]
    for cls, polygon in read_polygons(label):
        points = (polygon * [w, h]).ravel().tolist()
        rle = mask_utils.merge(mask_utils.frPyObjects([points], h, w))
        annotations.append(dict(
            id=next_annotation + len(annotations), image_id=image_id, category_id=int(cls) + 1,
            segmentation=[points], area=float(mask_utils.area(rle)),
            bbox=mask_utils.toBbox(rle).tolist(), iscrowd=0,
        ))
    return dict(id=image_id, file_name=str(file), height=h, width=w), annotations


def native_predictions(pred, image_shape, image_id):
    """Evaluate the implemented common-grid output, upsampled to original size."""
    from pycocotools import mask as mask_utils

    h, w = image_shape[:2]
    rows = []
    for mask, score, cls in zip(pred["masks"], pred["conf"].tolist(), pred["cls"].tolist()):
        native = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        rle = mask_utils.encode(np.asfortranarray(native))
        rle["counts"] = rle["counts"].decode("ascii")
        # Deliberately OMIT bbox: COCO segmentation area must come from mask, not box.
        rows.append(dict(image_id=image_id, category_id=int(cls) + 1, score=score, segmentation=rle))
    return rows


def coco_report(gt_data, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = gt_data
    gt.createIndex()
    if predictions:
        detections = gt.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = dict(images=gt_data["images"], categories=gt_data["categories"], annotations=[])
        detections.createIndex()
    evaluation = COCOeval(gt, detections, "segm")
    evaluation.params.maxDets = [1, 10, 100]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    keys = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    return {key: float(value) for key, value in zip(keys, evaluation.stats)}


def evaluate(weights, data, output, device="cpu", split="val", mechanisms=True, coco_secondary=True, limit=0):
    output, weights, data = Path(output), Path(weights), Path(data)
    if limit < 0 or split not in {"val", "test"}:
        raise ValueError("Invalid split/limit")
    config, files = source_files(data, split)
    files = files[:limit] if limit else files
    signature = {
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "data_yaml": data.read_text(encoding="utf-8"), "split": split, "mechanisms": mechanisms,
        "coco_secondary": coco_secondary, "fraction": 0.4, "limit": limit,
        "source_files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns,
                          hashlib.sha256(Path(img2label_paths([str(p)])[0]).read_bytes()).hexdigest()) for p in files],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": str(device), "torch": torch.__version__,
    }
    # JSON roundtrip makes tuples/lists identical on a restart.
    signature = json.loads(json.dumps(signature))
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "evaluation_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != signature:
        raise FileExistsError("Evaluation inputs changed; use a new project, never reuse stale predictions.")
    write_json(frozen, signature)
    report = output / "paired_metrics.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    cache = output / "candidate_cache"
    cache.mkdir(exist_ok=True)
    old_threads, old_cv = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = None
    try:
        model = YOLO(str(weights))
        head = model.model.model[-1]
        if hasattr(head, "quality_floor"):
            head.quality_floor = 0.5
        # Do NOT enable scoring in P10, whose quality supervision was removed.
        for _ in range(3):
            model.predict(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)
        matcher = SegmentationValidator(args=dict(plots=False))
        metrics, records, statistics, timers, errors, review = {}, [], [], {}, {}, []
        gt_data = dict(info=dict(description="Native polygon GT"), images=[], annotations=[],
                       categories=[dict(id=int(k) + 1, name=v) for k, v in config["names"].items()])
        coco_predictions = {m: [] for m in ("global", "fixed_box", "crossmask", "trustedmask")}
        annotation_id = 1

        def synchronize():
            if device != "cpu":
                torch.cuda.synchronize()

        for image_index, file in enumerate(files):
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError("Unreadable image: %s" % file)
            shape = raster_shape(*image.shape[:2], 640)
            cached = cache / ("%06d.npz" % image_index)
            if cached.exists():
                candidates, timing = load_candidates(cached)
            else:
                candidates = []
                synchronize()
                started = time.perf_counter()
                global_seconds = None
                for view, window in enumerate(uniform_windows(*image.shape[:2], 0.4)):
                    candidates.extend(add_view_metadata(
                        predict_view(model, image, window, shape, device), window, image.shape[:2], view
                    ))
                    if view == 0:
                        synchronize()
                        global_seconds = time.perf_counter() - started
                synchronize()
                timing = [global_seconds, time.perf_counter() - started]
                save_candidates(cached, candidates, timing)
            gt = ground_truth(file, image.shape[:2], shape)  # Only AFTER cached predictions.
            if coco_secondary:
                image_row, annotation_rows = native_coco_gt(file, image, image_index + 1, annotation_id)
                gt_data["images"].append(image_row)
                gt_data["annotations"].extend(annotation_rows)
                annotation_id += len(annotation_rows)
            # Each selection mode uses the same frozen candidate collection.
            for mode, pred, merge_started in iter_variants(candidates, shape, mechanisms):
                merge_seconds = time.perf_counter() - merge_started
                if mode not in metrics:
                    metrics[mode] = SegmentMetrics(config["names"])
                    timers[mode] = []
                    errors[mode] = dict(tp=0, duplicates=0, mask_or_localization=0, background=0)
                timers[mode].append((timing[0] if mode == "global" else timing[1]) + merge_seconds)
                overlap = mask_overlaps(gt["masks"], pred["masks"])
                boxes = box_iou(gt["boxes"], pred["bboxes"])
                flags = matcher.match_predictions(pred["cls"], gt["cls"], overlap).numpy()
                metrics[mode].update_stats(dict(
                    tp=matcher.match_predictions(pred["cls"], gt["cls"], boxes).numpy(), tp_m=flags,
                    conf=pred["conf"].numpy(), pred_cls=pred["cls"].numpy(),
                    target_cls=gt["cls"].numpy(), target_img=np.unique(gt["cls"].numpy()), im_name=str(file),
                ))
                threshold = pred["conf"].numpy() >= 0.25
                pairs = greedy_pairs(overlap[:, threshold].numpy(), 0.5)
                hit, matched_pred = set(pairs[:, 0].tolist()), set(pairs[:, 1].tolist())
                gt_area = gt["masks"].sum((1, 2))
                ov, bo = overlap[:, threshold].numpy(), boxes[:, threshold].numpy()
                for j in range(ov.shape[1]):
                    key = ("tp" if j in matched_pred else "duplicates" if ov[:, j].max(initial=0) >= 0.5
                           else "mask_or_localization" if bo[:, j].max(initial=0) >= 0.5 else "background")
                    errors[mode][key] += 1
                    if key != "tp" and mode in {"global", "trustedmask"}:
                        review.append(dict(image=str(file), mode=mode, event=key, instance=j, human_label=""))
                for j, area in enumerate(gt_area):
                    records.append(dict(image=str(file), mode=mode, gt_index_common=j,
                                        area640=int(area), matched25=j in hit))
                    if j not in hit and mode in {"global", "trustedmask"}:
                        review.append(dict(image=str(file), mode=mode, event="unmatched_GT",
                                           instance=j, human_label=""))
                statistics.append(dict(
                    image=str(file), mode=mode, group=None, targets=len(gt_area),
                    conf=pred["conf"].tolist(), tp_m=flags.tolist(), hits25=len(hit),
                    tiny_targets=int((gt_area < 256).sum()),
                    tiny_hits25=sum(j in hit for j in np.flatnonzero(gt_area < 256)),
                ))
                if coco_secondary and mode in coco_predictions:
                    coco_predictions[mode].extend(native_predictions(pred, image.shape, image_index + 1))
            print("PAPER paired %s %d/%d" % (split, image_index + 1, len(files)), flush=True)
        summary, raw_pr = {}, {}
        for mode, metric in metrics.items():
            stats = metric.process(plot=False)
            raw_pr[mode] = empirical_mask_pr(stats, max_points=None)
            rr = [r for r in records if r["mode"] == mode]
            tiny = [r for r in rr if r["area640"] < 256]
            counts = errors[mode]
            summary[mode] = dict(
                **metric.results_dict, tiny_n=len(tiny), tiny_matched=sum(r["matched25"] for r in tiny),
                errors25=counts, all_precision25=counts["tp"] / max(sum(counts.values()), 1),
                all_recall25=counts["tp"] / max(len(rr), 1),
                median_ms=float(np.median(timers[mode]) * 1000),
                p95_ms=float(np.percentile(timers[mode], 95) * 1000),
                latency_scope="synchronized view prediction+mask remapping+selection; excludes file IO/GT/metrics",
            )
            precision, recall = metric.seg.p_curve.mean(0), metric.seg.r_curve.mean(0)
            eligible = np.flatnonzero(precision >= 0.90)
            index = eligible[np.argmax(recall[eligible])] if len(eligible) else None
            summary[mode]["operating_p90"] = (
                dict(confidence=float(metric.seg.px[index]), precision=float(precision[index]),
                     recall=float(recall[index])) if index is not None else None
            )
        with (output / "prediction_stats.jsonl").open("w", encoding="utf-8") as handle:
            for row in statistics:
                handle.write(json.dumps(row) + "\n")
        with (output / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "mode", "event", "instance", "human_label"])
            writer.writeheader()
            writer.writerows(review)
        coco = {}
        if coco_secondary:
            write_json(output / "native_gt.json", gt_data)
            for mode, predictions in coco_predictions.items():
                write_json(output / ("native_predictions_" + mode + ".json"), predictions)
                coco[mode] = coco_report(gt_data, predictions)
        payload = dict(
            summary=summary, coco=coco, records=records, empirical_mask_pr=raw_pr, protocol=signature,
            limits="EV6 training remains unchanged. Diagnostic AP and native-GT COCO AP are separate. "
                   "COCO evaluates common640 predicted masks nearest-upsampled to original size, against native "
                   "polygon GT, maxDets100. Background buckets are not verified leaf errors. Single-seed "
                   "ablations are exploratory. No GT guides views or selection. Timings exclude file IO/GT/metrics; "
                   "cached inference times are reused only with the same evaluation signature.",
        )
        write_json(report, payload)
        return payload
    finally:
        del model
        torch.set_num_threads(old_threads)
        cv2.setNumThreads(old_cv)
