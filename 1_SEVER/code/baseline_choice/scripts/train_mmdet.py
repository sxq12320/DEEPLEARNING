"""Train RTMDet-Ins, Mask R-CNN, or SOLOv2 with MMDetection v3.3.0."""

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
from mmdet_common import build_training_config


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="MMDetection baseline ID.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset root.")
    parser.add_argument("--name", required=True, help="Unique experiment name.")
    parser.add_argument("--mmdet-root", type=Path, default=Path("third_party/mmdetection"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/mmdet"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional pretrained checkpoint.")
    parser.add_argument("--val-interval", type=int, default=5)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    registry = load_registry()
    baseline = get_baseline(args.baseline, family="mmdetection")
    dataset_root = resolve_path(args.dataset)
    output_root = resolve_path(args.output_root)
    mmdet_root = resolve_path(args.mmdet_root)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    run_dir = output_root / args.name
    require_new_directory(run_dir)

    epochs = args.epochs or int(registry["project"]["default_epochs"])
    class_names = registry["project"]["class_names"]
    set_random_seed(args.seed)
    cfg = build_training_config(
        baseline=baseline,
        mmdet_root=mmdet_root,
        dataset_root=dataset_root,
        run_dir=run_dir,
        class_names=class_names,
        epochs=epochs,
        batch_size=args.batch,
        workers=args.workers,
        seed=args.seed,
        checkpoint=checkpoint,
        val_interval=args.val_interval,
    )
    config_path = run_dir / "effective_config.py"
    cfg.dump(str(config_path))
    save_json(
        run_dir / "run_metadata.json",
        {
            "baseline": baseline,
            "dataset_root": str(dataset_root),
            "official_mmdet_root": str(mmdet_root),
            "effective_config": str(config_path),
            "epochs": epochs,
            "batch_per_gpu": args.batch,
            "seed": args.seed,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "environment": environment_snapshot(),
            "framework": framework_snapshot(),
        },
    )

    try:
        from mmengine.runner import Runner
    except ImportError as exc:
        raise RuntimeError("MMEngine is unavailable. Activate the citrus_mmdet environment.") from exc
    Runner.from_cfg(cfg).train()
    print(f"Training complete: {run_dir}")


if __name__ == "__main__":
    main()
