"""Train the official RF-DETR segmentation preview at the registered resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline_common import (
    environment_snapshot,
    framework_snapshot,
    get_baseline,
    load_registry,
    require_new_directory,
    resolve_path,
    save_json,
    set_random_seed,
)
from rfdetr_common import model_kwargs, require_rfdetr_class


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="rfdetr_seg_nano")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/rfdetr"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    registry = load_registry()
    baseline = get_baseline(args.baseline, family="rfdetr")
    dataset_root = resolve_path(args.dataset)
    rfdetr_dataset = dataset_root / "rfdetr"
    output_root = resolve_path(args.output_root)
    run_dir = output_root / args.name
    require_new_directory(run_dir)
    epochs = args.epochs or int(registry["project"]["default_epochs"])
    set_random_seed(args.seed)

    save_json(
        run_dir / "run_metadata.json",
        {
            "baseline": baseline,
            "dataset_root": str(dataset_root),
            "epochs": epochs,
            "batch": args.batch,
            "grad_accum_steps": args.grad_accum_steps,
            "seed": args.seed,
            "environment": environment_snapshot(),
            "framework": framework_snapshot(),
        },
    )
    model_class = require_rfdetr_class(baseline)
    model = model_class(**model_kwargs(baseline, args.device))
    model.train(
        dataset_dir=str(rfdetr_dataset),
        output_dir=str(run_dir),
        epochs=epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.workers,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=args.early_stopping,
        use_ema=args.use_ema,
        multi_scale=args.multi_scale,
        run_test=False,
        tensorboard=True,
        wandb=False,
    )
    print(f"Training complete: {run_dir}")
    print(f"Preferred checkpoint: {run_dir / 'checkpoint_best_total.pth'}")


if __name__ == "__main__":
    main()
