"""Train any registered citrus segmentation YAML through the standard Ultralytics ``YOLO`` API.

The script is the single-model counterpart to the controlled batch runners. It keeps architecture selection in YAML,
loads matching COCO-pretrained YOLO11n-seg weights, rewrites only the runtime dataset root, and locks the protocol used
by the current grouped-deduplicated citrus experiments.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from citrus_protocol import fixed_train_args, validate_locked_runtime, write_protocol_snapshot


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DEFAULT_DATA = WORKSPACE / "data" / "orange_yolo_grouped_dedup_20260820" / "data.yaml"
DEFAULT_PROJECT = ROOT / "1_results" / "CITRUS_SINGLE_MODEL_300EP"
DEFAULT_PRETRAINED = ROOT / "yolo11n-seg.pt"
FIXED_TRAIN = fixed_train_args()


def parse_args() -> argparse.Namespace:
    """Parse the controlled single-model training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Model YAML or .pt checkpoint.")
    parser.add_argument("--name", required=True, help="Unique output run name.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=FIXED_TRAIN["batch"])
    parser.add_argument("--imgsz", type=int, default=FIXED_TRAIN["imgsz"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=FIXED_TRAIN["workers"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable AMP; leave disabled to match the current S series.")
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--citrus-quality", type=float, default=None)
    parser.add_argument("--citrus-boundary", type=float, default=None)
    parser.add_argument("--citrus-query", type=float, default=None)
    parser.add_argument("--citrus-contrast", type=float, default=None)
    parser.add_argument("--citrus-exclusive", type=float, default=None)
    parser.add_argument("--citrus-concavity", type=float, default=None)
    parser.add_argument("--citrus-topology", type=float, default=None)
    parser.add_argument("--citrus-vfl", type=float, default=None)
    parser.add_argument("--nwd-ratio", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set every relevant pseudorandom generator and deterministic backend option."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_runtime_data_yaml(source: Path, project: Path) -> Path:
    """Copy the dataset YAML while binding its root to the source YAML directory."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    config["path"] = str(source.parent.resolve())
    runtime_yaml = project / "_protocol" / "grouped_dedup_runtime.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_yaml


def custom_loss_overrides(args: argparse.Namespace, head_name: str, sdr_stage: int = 1) -> dict[str, float]:
    """Resolve explicit losses and apply the documented topology/quality head defaults."""
    names = (
        "citrus_quality",
        "citrus_boundary",
        "citrus_query",
        "citrus_contrast",
        "citrus_exclusive",
        "citrus_concavity",
        "citrus_topology",
        "citrus_vfl",
        "nwd_ratio",
    )
    overrides = {name: float(value) for name in names if (value := getattr(args, name, None)) is not None}
    if head_name in {"SegmentCitrusTopo", "SegmentCitrusBLite", "SegmentCitrusLiteBQ"}:
        overrides.setdefault("citrus_boundary", 0.25)
        overrides.setdefault("citrus_query", 0.05)
    if head_name == "SegmentCitrusBQuality" and "citrus_quality" not in overrides:
        overrides["citrus_quality"] = 0.20
    if head_name == "SegmentCitrusDualProto" and "citrus_topology" not in overrides:
        overrides["citrus_topology"] = 0.10
    if head_name == "SegmentCitrusSDR":
        if sdr_stage not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Unknown SegmentCitrusSDR stage: {sdr_stage}")
        # Pass one identical hyperparameter vector to every G_0839 stage. Heads
        # without a corresponding output naturally contribute zero loss.
        overrides.setdefault("citrus_query", 0.03)
        overrides.setdefault("citrus_contrast", 0.05)
        overrides.setdefault("citrus_boundary", 0.10)
        overrides.setdefault("citrus_topology", 0.05)
    if any(value < 0 for value in overrides.values()):
        raise ValueError(f"Custom loss gains must be non-negative: {overrides}")
    return overrides


def main() -> None:
    """Build or resume one YAML-defined model and train it with the locked citrus protocol."""
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    project = args.project.expanduser().resolve()
    cache: bool | str = False if args.cache == "false" else args.cache
    validate_locked_runtime(
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=cache,
        amp=args.amp,
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if model_path.suffix.lower() == ".yaml" and not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.imgsz != 640 and not args.dry_run:
        raise ValueError("Formal citrus experiments lock imgsz=640; use --dry-run for alternate build checks.")

    from ultralytics import YOLO

    run_dir = project / args.name
    last_path = run_dir / "weights" / "last.pt"
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume because last.pt is missing: {last_path}")
        model = YOLO(str(last_path), task="segment")
    else:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
        model = YOLO(str(model_path), task="segment")
        if model_path.suffix.lower() == ".yaml":
            model.load(str(pretrained))

    head = model.model.model[-1]
    head_name = head.__class__.__name__
    loss_overrides = custom_loss_overrides(args, head_name, int(getattr(head, "sdr_stage", 1)))

    if args.dry_run:
        model.info(detailed=False, verbose=True, imgsz=args.imgsz)
        print(f"Model build passed: {model_path}")
        return

    runtime_data = prepare_runtime_data_yaml(data_path, project)
    protocol_path, protocol_hash = write_protocol_snapshot(
        project,
        {"entrypoint": "train_citrus_yaml.py"},
    )
    protocol = fixed_train_args()
    protocol.update(
        data=str(runtime_data),
        project=str(project),
        name=args.name,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        exist_ok=False,
        **loss_overrides,
    )
    project.mkdir(parents=True, exist_ok=True)
    (project / "_protocol" / f"{args.name}.json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "pretrained": str(pretrained),
                "source_data": str(data_path),
                "formal_protocol": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "method_loss_gains": loss_overrides,
                "protocol": protocol,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    set_seed(args.seed)
    if args.resume:
        model.train(resume=True)
    else:
        model.train(**protocol)


if __name__ == "__main__":
    main()
