"""Export RF-DETR segmentation masks to COCO JSON and run the shared evaluator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from baseline_common import (
    environment_snapshot,
    framework_snapshot,
    get_baseline,
    require_new_directory,
    resolve_path,
    save_json,
)
from coco_utils import (
    evaluate_predictions,
    load_coco_image_index,
    prediction_from_mask,
    resize_binary_mask,
    save_predictions,
)
from rfdetr_common import model_kwargs, require_rfdetr_class


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="rfdetr_seg_nano")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=False)
    return parser


def batches(values: List[Any], size: int) -> List[List[Any]]:
    """Split a list into fixed-size batches."""
    return [values[index : index + size] for index in range(0, len(values), size)]


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")
    baseline = get_baseline(args.baseline, family="rfdetr")
    weights = resolve_path(args.weights)
    dataset_root = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)
    require_new_directory(output_dir)
    annotation_path = dataset_root / "coco" / "annotations" / f"instances_{args.split}.json"
    image_dir = dataset_root / "coco" / "images" / args.split
    image_index = load_coco_image_index(annotation_path)
    ordered = sorted(image_index.items(), key=lambda item: int(item[1]["id"]))

    model_class = require_rfdetr_class(baseline)
    model = model_class(**model_kwargs(baseline, args.device, str(weights)))
    if args.optimize:
        model.optimize_for_inference(compile=True, batch_size=args.batch)

    predictions: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for group in tqdm(batches(ordered, args.batch), desc=f"RF-DETR {args.split}", unit="batch", dynamic_ncols=True):
        images = []
        for file_name, _ in group:
            with Image.open(image_dir / file_name) as image:
                images.append(image.convert("RGB"))
        detections_batch = model.predict(images, threshold=args.score_threshold)
        if len(group) == 1:
            detections_batch = [detections_batch]
        for (_, record), detections in zip(group, detections_batch):
            if detections.mask is None:
                continue
            scores = detections.confidence
            labels = detections.class_id
            for mask, score, label in zip(detections.mask, scores, labels):
                binary = resize_binary_mask(mask > 0, int(record["height"]), int(record["width"]))
                if np.any(binary):
                    predictions.append(
                        prediction_from_mask(
                            image_id=int(record["id"]),
                            category_id=int(label) + 1,
                            score=float(score),
                            mask=binary,
                        )
                    )
    elapsed = time.perf_counter() - start

    prediction_path = output_dir / "predictions.coco.json"
    save_predictions(prediction_path, predictions)
    metrics = evaluate_predictions(annotation_path, predictions, output_dir / "metrics.json")
    torch_model = getattr(model.model, "model", model.model)
    metrics.update(
        {
            "images": len(ordered),
            "latency_ms_per_image_end_to_end": 1000.0 * elapsed / max(len(ordered), 1),
            "weights": str(weights),
            "baseline": baseline,
            "params_m": sum(parameter.numel() for parameter in torch_model.parameters()) / 1e6,
            "optimized": args.optimize,
            "environment": environment_snapshot(),
            "framework": framework_snapshot(),
        }
    )
    save_json(output_dir / "metrics.json", metrics)
    print(f"Mask AP50-95: {metrics['mask_ap_50_95']:.4f}")
    print(f"Predictions: {prediction_path}")


if __name__ == "__main__":
    main()
