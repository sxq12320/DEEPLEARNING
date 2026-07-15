"""Train the official Torchvision Mask R-CNN R50-FPN citrus baseline."""

from __future__ import annotations

import argparse
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
from torchvision_maskrcnn_common import (
    CocoMaskRCNNDataset,
    build_maskrcnn_model,
    collate_detection_batch,
    evaluate_model,
    prepared_split_paths,
    train_one_epoch,
    validate_prepared_dataset,
)

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = SUITE_ROOT / "datasets" / "citrus_prepared"
DEFAULT_OUTPUT_ROOT = SUITE_ROOT / "runs" / "torchvision_maskrcnn"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Prepared dataset root.",
    )
    parser.add_argument(
        "--name",
        default="E_maskrcnn_r50_fpn_seed42",
        help="Unique experiment name.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Maximum image side after aspect-ratio resize.",
    )
    parser.add_argument(
        "--detections-per-image",
        type=int,
        default=50,
        help="Maximum returned instances; 50 covers the dataset maximum of 35.",
    )
    parser.add_argument(
        "--val-interval", type=int, default=5, help="Validate every N epochs."
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="Keep an extra checkpoint every N epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--initialization", choices=("coco", "imagenet", "none"), default="coco"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable CUDA mixed precision."
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a saved training checkpoint.",
    )
    return parser


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside each data-loader worker."""
    del worker_id
    worker_seed = __import__("torch").initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(value: str):
    """Resolve an automatic or explicit torch device."""
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )
    return device


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
    """Save a resumable checkpoint."""
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


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if (
        args.epochs <= 0
        or args.batch <= 0
        or args.imgsz <= 0
        or args.detections_per_image <= 0
    ):
        raise ValueError(
            "epochs, batch, imgsz, and detections-per-image must be positive"
        )
    if args.val_interval <= 0 or args.save_interval <= 0:
        raise ValueError("val-interval and save-interval must be positive")

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

    class_names = ["orange_immature"]
    dataset_report = validate_prepared_dataset(
        dataset_root, ("train", "val"), class_names
    )
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(f"device={device} amp={amp_enabled} run={run_dir}")

    train_ann, train_images = prepared_split_paths(dataset_root, "train")
    val_ann, val_images = prepared_split_paths(dataset_root, "val")
    train_dataset = CocoMaskRCNNDataset(train_ann, train_images, train=True)
    val_dataset = CocoMaskRCNNDataset(val_ann, val_images, train=False)
    generator = torch.Generator().manual_seed(args.seed)
    loader_common = {
        "num_workers": args.workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        generator=generator,
        **loader_common,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_common)

    initialization = "none" if resume_path else args.initialization
    model = build_maskrcnn_model(
        num_foreground_classes=len(class_names),
        imgsz=args.imgsz,
        initialization=initialization,
        detections_per_image=args.detections_per_image,
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = sorted(
        {
            max(1, round(args.epochs * 0.8)),
            max(1, round(args.epochs * 0.9)),
        }
    )
    milestones = [value for value in milestones if value < args.epochs]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    config: Dict[str, Any] = {
        "model": "torchvision.models.detection.maskrcnn_resnet50_fpn",
        "class_names": class_names,
        "num_classes_including_background": len(class_names) + 1,
        "dataset_root": str(dataset_root),
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "imgsz": args.imgsz,
        "detections_per_image": args.detections_per_image,
        "resize_protocol": "aspect ratio preserved; longer side capped at imgsz",
        "initialization": args.initialization,
        "seed": args.seed,
        "amp": amp_enabled,
        "lr_milestones": milestones,
        "val_interval": args.val_interval,
    }
    start_epoch = 1
    best_mask_ap = -math.inf
    history = []
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mask_ap = float(checkpoint.get("best_mask_ap", -math.inf))
        history_path = run_dir / "history.json"
        if history_path.is_file():
            history = __import__("json").loads(history_path.read_text(encoding="utf-8"))

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

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
        }

        should_validate = epoch % args.val_interval == 0 or epoch == args.epochs
        if should_validate:
            val_metrics, predictions = evaluate_model(
                model=model,
                loader=val_loader,
                device=device,
                annotation_path=val_ann,
            )
            epoch_record["val"] = val_metrics
            save_json(
                run_dir / "validation" / f"epoch_{epoch:03d}_metrics.json", val_metrics
            )
            current_mask_ap = float(val_metrics["mask_ap_50_95"])
            print(
                f"epoch={epoch} val_mask_ap50_95={current_mask_ap:.4f} "
                f"val_mask_ap50={float(val_metrics['mask_ap_50']):.4f}"
            )
            if current_mask_ap > best_mask_ap:
                best_mask_ap = current_mask_ap
                save_predictions(
                    run_dir / "validation" / "best_predictions.coco.json", predictions
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
        from run_maskrcnn import main as one_click_main

        one_click_main()
    else:
        main()
