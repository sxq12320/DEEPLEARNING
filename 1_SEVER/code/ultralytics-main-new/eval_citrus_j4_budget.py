"""Live budgeted inference: only selected crops are forwarded, not all9 cached then counted.

This is an additional common640-raster diagnostic, NOT a replacement for J4's
I-series EV6 validation or baseline original-resolution COCO evaluation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from citrus_e_v5_slicing import uniform_windows
from citrus_j4_budget import select_tiles
from citrus_slicing import source_files
from eval_citrus_e_v5 import add_view_metadata, merge_views
from eval_citrus_j2_gaze import predict_view
from eval_citrus_sliced import ground_truth, mask_overlaps, merge_candidates, raster_shape
from scripts.review_sage_v4r import greedy_pairs
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics, box_iou


def synchronize(device):
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def infer_budget(model, image, device="cpu", budget=2, policy="coverage", seed=0):
    """Return merged predictions and actual view IDs; no annotation input accepted."""
    h, w = image.shape[:2]
    shape = raster_shape(h, w, 640)
    windows = uniform_windows(h, w, 0.4)
    glance = add_view_metadata(predict_view(model, image, windows[0], shape, device), windows[0], (h, w), 0)
    selected = select_tiles(glance, windows[1:], budget, policy, seed)
    candidates = list(glance)
    for k in selected:
        window = windows[k + 1]
        candidates += add_view_metadata(predict_view(model, image, window, shape, device), window, (h, w), k + 1)
    predictions = (merge_views(candidates, shape, border_weight=0.5) if selected
                   else merge_candidates(candidates, shape))
    return predictions, selected, shape


def evaluate(weights, data, output, device="cpu", budget=2, policy="coverage", seed=42, limit=0):
    if budget < 0 or limit < 0:
        raise ValueError("Nonnegative budget and limit required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new diagnostic output directory: {output}")
    config, files = source_files(data, "val")
    files = files[:limit] if limit else files
    if not files:
        raise ValueError("Empty validation set")
    model = YOLO(str(weights), task="segment")
    model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, device=device, verbose=False, save=False)
    matcher = SegmentationValidator(args=dict(plots=False))
    metrics = SegmentMetrics(config["names"])
    timings, records = [], []
    tp25 = predictions25 = tiny_hit = tiny_n = instances = views = 0
    with torch.inference_mode():
        for index, path in enumerate(files):
            synchronize(device)
            start = time.perf_counter()
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            pred, selected, shape = infer_budget(model, image, device, budget, policy, seed + index)
            synchronize(device)
            timings.append(time.perf_counter() - start)
            # Ground truth is accessed ONLY after selection and inference finish.
            gt = ground_truth(path, image.shape[:2], shape)
            overlap = mask_overlaps(gt["masks"], pred["masks"])
            boxes = box_iou(gt["boxes"], pred["bboxes"])
            metrics.update_stats(dict(tp=matcher.match_predictions(pred["cls"], gt["cls"], boxes).numpy(),
                tp_m=matcher.match_predictions(pred["cls"], gt["cls"], overlap).numpy(),
                conf=pred["conf"].numpy(), pred_cls=pred["cls"].numpy(), target_cls=gt["cls"].numpy(),
                target_img=np.unique(gt["cls"].numpy()), im_name=path.name))
            keep = pred["conf"] >= 0.25
            pairs = greedy_pairs(overlap[:, keep].numpy(), 0.5)
            matched = set(pairs[:, 0].tolist())
            area = gt["masks"].sum((1, 2))
            tiny_n += int((area < 256).sum())
            tiny_hit += sum(j in matched for j in range(len(area)) if area[j] < 256)
            tp25 += len(pairs)
            predictions25 += int(keep.sum())  # ALL unmatched predictions are FP, not background-only.
            instances += len(area)
            views += 1 + len(selected)
            records.append(dict(image=path.name, selected_tiles=selected, views=1 + len(selected),
                                elapsed_ms=timings[-1] * 1000, source=str(path),
                                image_sha256=file_sha256(path),
                                label_sha256=(file_sha256(img2label_paths([str(path)])[0])
                                    if Path(img2label_paths([str(path)])[0]).is_file() else None)))
            if (index + 1) % 20 == 0 or index + 1 == len(files):
                print(f"LIVE {policy} K={budget}: {index + 1}/{len(files)}", flush=True)
    metrics.process(plot=False)
    from citrus_j4_batch import implementation_hash, tracked_sources

    root = Path(__file__).resolve().parent
    diagnostic_sources = ("eval_citrus_j4_budget.py", "citrus_j4_budget.py",
                          "scripts/review_sage_v4r.py")
    diagnostic_digest = hashlib.sha256()
    for name in diagnostic_sources:
        diagnostic_digest.update(name.encode())
        diagnostic_digest.update((root / name).read_bytes())
    payload = dict(summary=dict(**metrics.results_dict, precision25=tp25 / max(predictions25, 1),
        recall25=tp25 / max(instances, 1), tiny_hit=tiny_hit, tiny_n=tiny_n,
        views_per_image=views / len(files), median_ms=float(np.median(timings) * 1000),
        p95_ms=float(np.percentile(timings, 95) * 1000)), images=len(files), records=records,
        protocol=dict(weights=str(weights), data=str(data), policy=policy, budget=budget, seed=seed,
                      mask_long_side=640, imgsz=640, conf=0.001, nms_iou=0.7, max_det=300, fraction=0.4,
                      selection_gt=False, device=str(device), selection_small_mask_area=1024,
                      selection_score_min=0.25, border_weight=0.5),
        provenance=dict(weights_sha256=file_sha256(weights), data_yaml_sha256=file_sha256(data),
                        model_code_sha256=implementation_hash(), model_code_files=tracked_sources(),
                        diagnostic_sha256=diagnostic_digest.hexdigest(), diagnostic_files=diagnostic_sources,
                        torch=torch.__version__, opencv=cv2.__version__,
                        note="Version logging only, not a dataset confirmation/identity gate. Input hashes excluded "
                             "from inference timing and accessed after prediction; no GT enters tile selection."),
        limits="Exploratory common-raster AP, not EV6/COCO. Measured time includes disk read, RGB crops, resize, "
               "forward, mask projection, selection and merge; excludes GT and metric calculation. "
               "Only selected tiles are forwarded. Mask AP improvement is not guaranteed.")
    output.mkdir(parents=True)
    (output / "live_budget_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "data", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--policy", choices=("coverage", "random", "global", "all"), default="coverage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    torch.set_num_threads(2)
    evaluate(**vars(parser.parse_args()))
