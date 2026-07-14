"""Export MMDetection instance masks to COCO JSON and run the shared evaluator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline_common import (
    environment_snapshot,
    framework_snapshot,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Effective config; defaults to <run>/effective_config.py beside the weights directory.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    return parser


def masks_array(value: Any) -> np.ndarray:
    """Convert MMDetection mask containers or tensors to a NumPy array."""
    if hasattr(value, "to_ndarray"):
        value = value.to_ndarray()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def infer_config_path(weights: Path) -> Path:
    """Locate the frozen training config beside a checkpoint."""
    candidates = (weights.parent / "effective_config.py", weights.parent.parent / "effective_config.py")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Pass --config or keep effective_config.py in the run directory.")


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    dataset_root = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)
    require_new_directory(output_dir)
    weights = resolve_path(args.weights)
    config = resolve_path(args.config) if args.config else infer_config_path(weights)
    annotation_path = dataset_root / "coco" / "annotations" / f"instances_{args.split}.json"
    image_dir = dataset_root / "coco" / "images" / args.split
    image_index = load_coco_image_index(annotation_path)

    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError as exc:
        raise RuntimeError("MMDetection is unavailable. Activate the citrus_mmdet environment.") from exc

    model = init_detector(str(config), str(weights), device=args.device)
    predictions: List[Dict[str, Any]] = []
    start = time.perf_counter()
    image_count = 0
    for file_name, record in sorted(image_index.items(), key=lambda item: int(item[1]["id"])):
        result = inference_detector(model, str(image_dir / file_name))
        image_count += 1
        instances = result.pred_instances.cpu()
        if len(instances) == 0 or "masks" not in instances:
            continue
        masks = masks_array(instances.masks)
        scores = instances.scores.numpy()
        labels = instances.labels.numpy().astype(int)
        for mask, score, label in zip(masks, scores, labels):
            if float(score) < args.score_threshold:
                continue
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
    metrics.update(
        {
            "images": image_count,
            "latency_ms_per_image_end_to_end": 1000.0 * elapsed / max(image_count, 1),
            "weights": str(weights),
            "config": str(config),
            "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
            "environment": environment_snapshot(),
            "framework": framework_snapshot(),
        }
    )
    save_json(output_dir / "metrics.json", metrics)
    print(f"Mask AP50-95: {metrics['mask_ap_50_95']:.4f}")
    print(f"Predictions: {prediction_path}")


if __name__ == "__main__":
    main()
