"""Train a U-Net semantic model and evaluate watershed-derived citrus instances."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from baseline_common import (
    environment_snapshot,
    framework_snapshot,
    require_new_directory,
    resolve_path,
    save_json,
    set_random_seed,
)
from coco_utils import save_predictions
from unet_common import (
    CitrusSemanticDataset,
    build_unet,
    collate_semantic_batch,
    evaluate_model,
    semantic_split_paths,
    train_one_epoch,
    validate_semantic_dataset,
)

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = SUITE_ROOT / "datasets" / "citrus_prepared"
DEFAULT_OUTPUT_ROOT = SUITE_ROOT / "runs" / "unet_watershed"


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments used by the one-click launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="E_unet_r18_watershed_seed42")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--watershed-min-distance", type=int, default=8)
    parser.add_argument("--watershed-min-area", type=int, default=20)
    parser.add_argument("--max-instances", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def resolve_device(value: str):
    """Resolve automatic or explicit torch device selection."""
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is unavailable.")
    return device


def seed_worker(worker_id: int) -> None:
    """Seed each data-loader worker."""
    del worker_id
    worker_seed = __import__("torch").initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_mask_ap: float,
    config: Dict[str, Any],
) -> None:
    """Save a fully resumable checkpoint."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_mask_ap": best_mask_ap,
            "config": config,
        },
        path,
    )


def print_configuration(config: Dict[str, Any], device, run_dir: Path) -> None:
    """Print all important experiment parameters before training."""
    print("\n" + "=" * 80)
    print("U-Net + Watershed training parameters")
    print("=" * 80)
    for key, value in config.items():
        print(f"{key:28s}: {value}")
    print(f"{'device':28s}: {device}")
    print(f"{'output_directory':28s}: {run_dir}")
    print("=" * 80)


def main() -> None:
    """Train and periodically evaluate the U-Net baseline."""
    args = build_parser().parse_args()
    positive_values = (
        args.epochs,
        args.batch,
        args.workers + 1,
        args.imgsz,
        args.watershed_min_distance,
        args.watershed_min_area,
        args.max_instances,
        args.val_interval,
        args.save_interval,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError(
            "Epoch, batch, image, watershed, and interval values are invalid"
        )
    if not 0.0 < args.prob_threshold < 1.0:
        raise ValueError("--prob-threshold must be in (0, 1)")

    import torch
    from torch.utils.data import DataLoader

    dataset_root = resolve_path(args.dataset)
    output_root = resolve_path(args.output_root)
    run_dir = output_root / args.name
    resume_path = resolve_path(args.resume) if args.resume else None
    if resume_path:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        require_new_directory(run_dir)

    dataset_report = validate_semantic_dataset(dataset_root, ("train", "val"))
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    train_dataset = CitrusSemanticDataset(dataset_root, "train", args.imgsz, train=True)
    val_dataset = CitrusSemanticDataset(dataset_root, "val", args.imgsz, train=False)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "collate_fn": collate_semantic_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(args.batch, 4)),
        shuffle=False,
        **loader_options,
    )

    checkpoint = (
        torch.load(resume_path, map_location=device, weights_only=False)
        if resume_path
        else None
    )
    encoder_weights = None if checkpoint else args.encoder_weights
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    model = build_unet(args.encoder, encoder_weights).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    config: Dict[str, Any] = {
        "model": "segmentation_models_pytorch.Unet",
        "encoder": args.encoder,
        "encoder_weights": args.encoder_weights,
        "dataset_root": str(dataset_root),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "workers": args.workers,
        "image_size": args.imgsz,
        "resize_protocol": "aspect-ratio-preserving square letterbox",
        "loss": f"BCEWithLogits + {args.dice_weight:g} * DiceLoss",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "probability_threshold": args.prob_threshold,
        "watershed_min_distance": args.watershed_min_distance,
        "watershed_min_area": args.watershed_min_area,
        "max_instances": args.max_instances,
        "validation_interval": args.val_interval,
        "seed": args.seed,
        "amp": amp_enabled,
        "parameters": total_parameters,
        "parameters_m": total_parameters / 1e6,
        "trainable_parameters": trainable_parameters,
    }
    print_configuration(config, device, run_dir)

    start_epoch = 1
    best_mask_ap = -math.inf
    history = []
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mask_ap = float(checkpoint.get("best_mask_ap", -math.inf))
        history_path = run_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resuming from epoch {start_epoch}: {resume_path}")

    save_json(
        run_dir / "run_metadata.json",
        {
            "config": config,
            "dataset": dataset_report,
            "resume": str(resume_path) if resume_path else None,
            "environment": environment_snapshot(),
            "framework": framework_snapshot(),
        },
    )
    _, _, val_annotation = semantic_split_paths(dataset_root, "val")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            args.prob_threshold,
            args.dice_weight,
        )
        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
        }
        should_validate = epoch % args.val_interval == 0 or epoch == args.epochs
        if should_validate:
            val_metrics, predictions = evaluate_model(
                model,
                val_loader,
                device,
                val_annotation,
                probability_threshold=args.prob_threshold,
                min_distance=args.watershed_min_distance,
                min_area=args.watershed_min_area,
                max_instances=args.max_instances,
                description=f"Validate epoch {epoch}",
            )
            val_metrics["params_m"] = total_parameters / 1e6
            epoch_record["val"] = val_metrics
            save_json(
                run_dir / "validation" / f"epoch_{epoch:03d}_metrics.json",
                val_metrics,
            )
            current_mask_ap = float(val_metrics["mask_ap_50_95"])
            print(
                f"epoch={epoch} loss={train_metrics['loss']:.4f} "
                f"semantic_dice={val_metrics['semantic_dice']:.4f} "
                f"semantic_iou={val_metrics['semantic_iou']:.4f} "
                f"mask_ap50_95={current_mask_ap:.4f} "
                f"mask_ap50={float(val_metrics['mask_ap_50']):.4f}"
            )
            if current_mask_ap > best_mask_ap:
                best_mask_ap = current_mask_ap
                save_predictions(
                    run_dir / "validation" / "best_predictions.coco.json",
                    predictions,
                )
                save_json(run_dir / "validation" / "best_metrics.json", val_metrics)
                save_checkpoint(
                    run_dir / "model_best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_mask_ap,
                    config,
                )

        scheduler.step()
        history.append(epoch_record)
        save_json(run_dir / "history.json", history)
        save_checkpoint(
            run_dir / "model_last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_mask_ap,
            config,
        )
        if epoch % args.save_interval == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_mask_ap,
                config,
            )

    print(f"Training complete: {run_dir}")
    print(f"Best validation Mask AP50-95: {best_mask_ap:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.path.insert(0, str(SUITE_ROOT))
        from run_unet import main as one_click_main

        one_click_main()
    else:
        main()
