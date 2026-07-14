"""Export YOLO masks to COCO JSON and run the shared evaluator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline_common import environment_snapshot, framework_snapshot, resolve_path, save_json
from coco_utils import (
    evaluate_predictions,
    load_coco_image_index,
    prediction_from_mask,
    resize_binary_mask,
    save_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset root.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True, help="New or empty evaluation directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold required for COCO AP integration.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU used by Ultralytics.")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    from baseline_common import require_new_directory

    dataset_root = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)
    require_new_directory(output_dir)
    weights = resolve_path(args.weights)
    annotation_path = dataset_root / "coco" / "annotations" / f"instances_{args.split}.json"
    image_dir = dataset_root / "coco" / "images" / args.split
    image_index = load_coco_image_index(annotation_path)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Activate the YOLO environment first.") from exc

    model = YOLO(str(weights))
    start = time.perf_counter()
    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        stream=True,
        retina_masks=True,
        verbose=False,
    )
    predictions: List[Dict[str, Any]] = []
    image_count = 0
    for result in results:
        image_count += 1
        file_name = Path(result.path).name
        record = image_index.get(file_name)
        if record is None:
            raise KeyError(f"Prediction image is absent from COCO annotations: {file_name}")
        if result.masks is None or result.boxes is None:
            continue
        masks = result.masks.data.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for mask, score, class_id in zip(masks, scores, classes):
            binary = resize_binary_mask(mask > 0.5, int(record["height"]), int(record["width"]))
            if np.any(binary):
                predictions.append(
                    prediction_from_mask(
                        image_id=int(record["id"]),
                        category_id=int(class_id) + 1,
                        score=float(score),
                        mask=binary,
                    )
                )
    elapsed = time.perf_counter() - start

    prediction_path = output_dir / "predictions.coco.json"
    save_predictions(prediction_path, predictions)
    metrics = evaluate_predictions(annotation_path, predictions, output_dir / "metrics.json")
    metrics["latency_ms_per_image_end_to_end"] = 1000.0 * elapsed / max(image_count, 1)
    metrics["images"] = image_count
    metrics["weights"] = str(weights)
    metrics["environment"] = environment_snapshot()
    metrics["framework"] = framework_snapshot()
    save_json(output_dir / "metrics.json", metrics)
    print(f"Mask AP50-95: {metrics['mask_ap_50_95']:.4f}")
    print(f"Predictions: {prediction_path}")


if __name__ == "__main__":
    main()
