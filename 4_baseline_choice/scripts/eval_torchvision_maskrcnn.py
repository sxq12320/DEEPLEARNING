"""Evaluate a Torchvision Mask R-CNN checkpoint on validation or test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline_common import (
    environment_snapshot,
    framework_snapshot,
    require_new_directory,
    resolve_path,
    save_json,
)
from coco_utils import save_predictions
from torchvision_maskrcnn_common import (
    CocoMaskRCNNDataset,
    build_maskrcnn_model,
    collate_detection_batch,
    evaluate_model,
    prepared_split_paths,
    validate_prepared_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=None, help="Defaults to the value stored in the checkpoint.")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    return parser


def resolve_device(value: str):
    """Resolve an automatic or explicit torch device."""
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    weights_path = resolve_path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    dataset_root = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)
    require_new_directory(output_dir)
    class_names = ["orange_immature"]
    dataset_report = validate_prepared_dataset(dataset_root, (args.split,), class_names)
    annotation_path, image_dir = prepared_split_paths(dataset_root, args.split)

    device = resolve_device(args.device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    imgsz = args.imgsz or int(checkpoint_config.get("imgsz", 640))
    model = build_maskrcnn_model(
        num_foreground_classes=len(class_names),
        imgsz=imgsz,
        initialization="none",
    ).to(device)
    model.load_state_dict(state_dict)

    dataset = CocoMaskRCNNDataset(annotation_path, image_dir, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    metrics, predictions = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        annotation_path=annotation_path,
        score_threshold=args.score_threshold,
    )
    metrics.update(
        {
            "split": args.split,
            "weights": str(weights_path),
            "imgsz": imgsz,
            "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
            "dataset": dataset_report[args.split],
            "environment": environment_snapshot(),
            "framework": framework_snapshot(),
        }
    )
    save_predictions(output_dir / "predictions.coco.json", predictions)
    save_json(output_dir / "metrics.json", metrics)
    print(f"Mask AP50-95: {metrics['mask_ap_50_95']:.4f}")
    print(f"Mask AP50: {metrics['mask_ap_50']:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
