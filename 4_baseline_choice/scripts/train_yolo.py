"""Train one Ultralytics segmentation baseline without overwriting prior runs."""

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
    write_runtime_yolo_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="YOLO baseline ID from configs/baselines.yaml.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset root from prepare_dataset.py.")
    parser.add_argument("--name", required=True, help="Unique run name, for example B0_yolov8n_seed42.")
    parser.add_argument("--output-root", type=Path, default=Path("runs/yolo"), help="Parent directory for YOLO runs.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="Ultralytics device value, such as 0, 0,1, cpu, or mps.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    registry = load_registry()
    baseline = get_baseline(args.baseline, family="yolo")
    dataset_root = resolve_path(args.dataset)
    output_root = resolve_path(args.output_root)
    run_dir = output_root / args.name
    require_new_directory(run_dir)

    epochs = args.epochs or int(registry["project"]["default_epochs"])
    imgsz = args.imgsz or int(baseline.get("imgsz", registry["project"]["default_imgsz"]))
    class_names = registry["project"]["class_names"]
    runtime_yaml = write_runtime_yolo_yaml(dataset_root, run_dir / "dataset.runtime.yaml", class_names)
    set_random_seed(args.seed)

    metadata = {
        "baseline": baseline,
        "dataset_root": str(dataset_root),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "environment": environment_snapshot(),
        "framework": framework_snapshot(),
    }
    save_json(run_dir / "run_metadata.json", metadata)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Activate the YOLO environment first.") from exc

    model = YOLO(baseline["model"])
    model.train(
        data=str(runtime_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        amp=args.amp,
        cache=args.cache,
        pretrained=args.pretrained,
        project=str(output_root),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
    )
    print(f"Training complete: {run_dir}")


if __name__ == "__main__":
    main()
