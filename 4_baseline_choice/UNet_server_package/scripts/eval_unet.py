"""Evaluate a trained U-Net with semantic and watershed instance metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_common import resolve_path, save_json
from coco_utils import save_predictions
from train_unet import resolve_device
from unet_common import (
    CitrusSemanticDataset,
    build_unet,
    collate_semantic_batch,
    evaluate_model,
    semantic_split_paths,
    validate_semantic_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Build evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prob-threshold", type=float, default=None)
    parser.add_argument("--watershed-min-distance", type=int, default=None)
    parser.add_argument("--watershed-min-area", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    return parser


def main() -> None:
    """Run evaluation using training parameters stored in the checkpoint."""
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    weights = resolve_path(args.weights)
    dataset_root = resolve_path(args.dataset)
    output = resolve_path(args.output)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    validate_semantic_dataset(dataset_root, (args.split,))
    device = resolve_device(args.device)
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    encoder = str(config.get("encoder", "resnet18"))
    imgsz = int(config.get("image_size", 640))
    probability_threshold = (
        args.prob_threshold
        if args.prob_threshold is not None
        else float(config.get("probability_threshold", 0.5))
    )
    min_distance = (
        args.watershed_min_distance
        if args.watershed_min_distance is not None
        else int(config.get("watershed_min_distance", 8))
    )
    min_area = (
        args.watershed_min_area
        if args.watershed_min_area is not None
        else int(config.get("watershed_min_area", 20))
    )
    max_instances = (
        args.max_instances
        if args.max_instances is not None
        else int(config.get("max_instances", 50))
    )
    model = build_unet(encoder, None).to(device)
    model.load_state_dict(checkpoint["model"])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    dataset = CitrusSemanticDataset(dataset_root, args.split, imgsz, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_semantic_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    _, _, annotation_path = semantic_split_paths(dataset_root, args.split)

    print("\n" + "=" * 80)
    print("U-Net + Watershed evaluation parameters")
    print("=" * 80)
    parameters = {
        "weights": weights,
        "dataset": dataset_root,
        "split": args.split,
        "device": device,
        "encoder": encoder,
        "image_size": imgsz,
        "batch_size": args.batch,
        "workers": args.workers,
        "probability_threshold": probability_threshold,
        "watershed_min_distance": min_distance,
        "watershed_min_area": min_area,
        "max_instances": max_instances,
        "parameters_m": total_parameters / 1e6,
        "output": output,
    }
    for key, value in parameters.items():
        print(f"{key:28s}: {value}")
    print("=" * 80)

    metrics, predictions = evaluate_model(
        model,
        loader,
        device,
        annotation_path,
        probability_threshold=probability_threshold,
        min_distance=min_distance,
        min_area=min_area,
        max_instances=max_instances,
        description=f"Test {args.split}",
    )
    metrics["params_m"] = total_parameters / 1e6
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", metrics)
    save_predictions(output / "predictions.coco.json", predictions)
    print("\nFinal metrics")
    print("-" * 80)
    for key in (
        "semantic_dice",
        "semantic_iou",
        "mask_ap_50_95",
        "mask_ap_50",
        "mask_ap_75",
        "mask_precision",
        "mask_recall",
        "mask_f1",
        "model_latency_ms_per_image",
        "peak_vram_mb",
        "params_m",
    ):
        if key in metrics:
            print(f"{key:28s}: {float(metrics[key]):.6f}")
    print(f"Saved metrics: {output / 'metrics.json'}")
    print(f"Saved predictions: {output / 'predictions.coco.json'}")


if __name__ == "__main__":
    main()
