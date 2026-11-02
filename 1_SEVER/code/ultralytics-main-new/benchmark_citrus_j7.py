"""Benchmark trained .pt segmentation models through the real single-image YOLO prediction entry.

Example:
    python benchmark_citrus_j7.py --weights path/to/control/best_mask.pt path/to/J104/best_mask.pt \
        --source /data/sxq/datasets/orange/val/images --device 1 --output /data/sxq/results/j7_latency.json

No labels are needed. This measures preloaded-image prediction, NOT disk IO, dataset validation or accuracy.
"""

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from citrus_j7_batch import bind_device

PREDICT_ARGS = {
    "imgsz": 640, "batch": 1, "half": False, "conf": 0.001, "iou": 0.7, "max_det": 300,
    "rect": False, "augment": False, "agnostic_nms": False, "retina_masks": False,
    "save": False, "save_txt": False, "save_crop": False, "show": False, "plots": False,
    "verbose": False, "stream": False,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    """Parse without importing Torch/OpenCV; explicitly bound benchmark workloads."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", required=True, help="One or more existing trained .pt checkpoints")
    parser.add_argument("--source", required=True, help="Image file or image folder; labels are not read")
    parser.add_argument("--output", required=True, help="A NEW .json file; existing output is never overwritten")
    parser.add_argument("--device", default="1", help="One physical GPU index, or cpu; bound before Torch import")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=20, help="Maximum distinct images, 1..30 (default 20)")
    parser.add_argument("--repeats", type=int, default=2, help="Passes over those same images, 1..10")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured complete prediction calls per model, 1..10")
    args = parser.parse_args(argv)
    if not 1 <= args.samples <= 30 or not 1 <= args.repeats <= 10 or not 1 <= args.warmup <= 10:
        parser.error("samples must be 1..30; repeats and warmup must be 1..10")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    args.weights = [Path(path).expanduser().resolve() for path in args.weights]
    if len(set(args.weights)) != len(args.weights):
        parser.error("Weights must be distinct checkpoint paths")
    for path in args.weights:
        if path.suffix.lower() != ".pt" or not path.is_file():
            parser.error("Use an existing trained .pt file, not YAML/random models or remote downloads: {}".format(path))
    args.source = Path(args.source).expanduser().resolve()
    if not args.source.exists():
        parser.error("Source does not exist: {}".format(args.source))
    args.output = Path(args.output).expanduser().resolve()
    if args.output.suffix.lower() != ".json":
        parser.error("Output must have .json suffix")
    if args.output.exists():
        parser.error("Refusing to overwrite existing benchmark: {}".format(args.output))
    return args


def percentile(values, q):
    """Linear-interpolated percentile, matching NumPy's default without importing NumPy."""
    if not values or not 0 <= q <= 100:
        raise ValueError("Nonempty values and percentile within [0, 100] required")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = int(math.floor(position)), int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_rows(rows):
    """Keep wall time separate from predictor's narrower component profiles."""
    if not rows:
        raise ValueError("No measured prediction calls")
    result = {"prediction_calls": len(rows)}
    for key in ("wall_ms", "preprocess_ms", "inference_ms", "postprocess_ms"):
        values = [float(row[key]) for row in rows]
        result[key] = {
            "median": statistics.median(values), "p95": percentile(values, 95),
            "mean": statistics.mean(values), "min": min(values), "max": max(values),
        }
    for key in ("decoded_after_nms", "returned_nonempty_instances"):
        result[key] = {
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
            "mean": statistics.mean(row[key] for row in rows),
        }
    return result


def source_files(source, limit):
    """Read only the user-selected image directory, not labels or unrelated directories."""
    if source.is_file():
        files = [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    else:
        files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError("No supported images. Point --source directly at the images folder: {}".format(source))
    return files[:limit]


def checkpoint_evidence(model):
    """Reject checkpoints with no available evidence of training; this is not an integrity audit."""
    checkpoint = model.ckpt if isinstance(getattr(model, "ckpt", None), dict) else {}
    epoch = int(checkpoint.get("epoch", -1))
    history = checkpoint.get("train_results")
    metrics = checkpoint.get("train_metrics")
    has_history = isinstance(history, dict) and bool(history)
    has_metrics = isinstance(metrics, dict) and any(
        isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0
        for value in metrics.values()
    )
    if epoch < 0 and not has_history and not has_metrics:
        raise ValueError(
            "This checkpoint has no training epoch/history/metrics metadata. Refusing to compare a potentially "
            "untrained model as deployment evidence. Use the actual trained best_mask.pt."
        )
    return {"saved_epoch_zero_based": epoch, "training_history_present": has_history,
            "positive_training_metric_present": has_metrics, "integrity_or_quality_certified": False}


def benchmark(args):
    """Run synchronized complete prediction calls; no GFLOPs-based speed inference."""
    logical_device = bind_device(args.device, args.cpu_threads)
    import torch
    import cv2
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationPredictor

    class CountedSegmentationPredictor(SegmentationPredictor):
        """Observe counts only; retain the parent's exact mask decoding and empty-mask filtering."""

        def construct_result(self, pred, img, orig_img, img_path, proto):
            if not hasattr(self, "j7_decode_counts"):
                self.j7_decode_counts = []
            self.j7_decode_counts.append(int(pred.shape[0]))
            return super().construct_result(pred, img, orig_img, img_path, proto)

    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    if logical_device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested but CUDA is unavailable; no silent CPU fallback")
    device = torch.device("cpu" if logical_device == "cpu" else "cuda:0")
    files = source_files(args.source, args.samples)
    images = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not decode image: {}".format(path))
        images.append(image)  # Standard YOLO ndarray input is OpenCV BGR, not pre-converted RGB.
    if len(images) < args.samples:
        print("Only {} distinct images available; all will be used each repeat.".format(len(images)), flush=True)
    runtime = {
        "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "platform": platform.platform(),
        "physical_device_requested": str(args.device), "logical_device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    arms = []
    for weights in args.weights:
        print("Benchmarking {} (real checkpoint; {} images x {} repeats)".format(
            weights, len(images), args.repeats
        ), flush=True)
        model = YOLO(str(weights), task="segment", verbose=False)
        evidence = checkpoint_evidence(model)
        head = model.model.model[-1]
        if hasattr(head, "bind_high_view") or "citrus_j6" in type(head).__module__:
            raise ValueError("J6 requires its paired-view decoder; single-input YOLO timing would omit its deployment.")
        parameters = sum(parameter.numel() for parameter in model.model.parameters())
        prediction_args = dict(PREDICT_ARGS, device=logical_device)
        for index in range(args.warmup):
            model.predict(source=images[index % len(images)].copy(), predictor=CountedSegmentationPredictor,
                          **prediction_args)
        synchronize()
        if not isinstance(model.predictor, SegmentationPredictor):
            raise RuntimeError("Expected the official single-image SegmentationPredictor")
        if model.predictor.args.half or getattr(model.predictor.model, "fp16", False):
            raise RuntimeError("Predictor unexpectedly enabled FP16; comparison requires half=False")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        for repeat in range(args.repeats):
            for path, image in zip(files, images):
                synchronize()
                started = time.perf_counter()
                before_count = len(model.predictor.j7_decode_counts)
                results = model.predict(source=image.copy(), predictor=CountedSegmentationPredictor, **prediction_args)
                synchronize()  # Includes all outstanding mask composition/materialization on this GPU.
                elapsed = (time.perf_counter() - started) * 1000
                if len(results) != 1:
                    raise RuntimeError("Expected batch=1 and exactly one Results object")
                if len(model.predictor.j7_decode_counts) != before_count + 1:
                    raise RuntimeError("Expected one count observation before empty-mask filtering")
                result = results[0]
                masks = result.masks.data if result.masks is not None else None
                rows.append({
                    "repeat_zero_based": repeat, "image": str(path), "input_shape_hwc": list(image.shape),
                    "wall_ms": elapsed,
                    "preprocess_ms": float(result.speed["preprocess"]),
                    "inference_ms": float(result.speed["inference"]),
                    "postprocess_ms": float(result.speed["postprocess"]),
                    "decoded_after_nms": model.predictor.j7_decode_counts[-1],
                    "returned_nonempty_instances": len(result.boxes) if result.boxes is not None else 0,
                    "materialized_mask_shape": list(masks.shape) if masks is not None else None,
                    "materialized_mask_device": str(masks.device) if masks is not None else None,
                    "materialized_mask_bytes": masks.numel() * masks.element_size() if masks is not None else 0,
                })
                del result, results, masks
        summary = summarize_rows(rows)
        arms.append({
            "weights": str(weights), "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "checkpoint_training_evidence": evidence, "head_class": type(head).__name__,
            "unfused_parameters": parameters, "summary": summary, "per_image": rows,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        })
        print("{}: wall median {:.3f} / p95 {:.3f} ms; decoded {:.1f}; returned nonempty {:.1f}/image".format(
            weights.name, summary["wall_ms"]["median"], summary["wall_ms"]["p95"],
            summary["decoded_after_nms"]["mean"], summary["returned_nonempty_instances"]["mean"]
        ), flush=True)
        del model, head
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "scope": "batch-1 preloaded image -> official YOLO.predict -> materialized instance masks",
        "io_included": False, "labels_or_accuracy_evaluated": False, "input_array_copy_included": True,
        "output_cpu_transfer_included": False,
        "excluded": ["checkpoint loading/fusing", "warmup", "image disk IO/decoding", "image saving/plotting",
                     "evaluation matching/AP", "output mask transfer to CPU", "external views/paired decoders"],
        "prediction_args": dict(PREDICT_ARGS, device=logical_device), "amp_enabled": False,
        "warmup_calls_per_model": args.warmup, "requested_images": args.samples,
        "distinct_images": len(images), "repeats": args.repeats, "images": [str(path) for path in files],
        "mask_raster": "native SegmentationPredictor, not EV6 input/2 validation",
        "count_semantics": (
            "decoded_after_nms is observed before the unchanged parent's mask composition/empty-mask filtering; "
            "returned_nonempty_instances is len(Results.boxes) after that filtering. Zero returned instances "
            "does NOT mean mask composition was skipped. Count observer overhead is included in all wall times."
        ),
        "interpretation": (
            "Use trained checkpoints on identical images. No speed or AP claim follows from GFLOPs alone. "
            "Native prediction preserves each graph's prototype grid before its standard mask decoding. "
            "EV6 instead resizes prototype logits to input/2 BEFORE mask composition. This benchmark is "
            "not that exact EV6 execution chain; do not combine its latency with EV6 AP as an identical-path claim."
        ),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print("Saved new benchmark JSON: {}".format(args.output), flush=True)
    return record


if __name__ == "__main__":
    try:
        benchmark(parse_args())
    except KeyboardInterrupt:
        print("Benchmark interrupted; no next checkpoint will run.", flush=True)
        raise SystemExit(130)
