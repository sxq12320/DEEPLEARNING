"""Sequential batch runner for the latency-aware Light citrus series.

Examples:
    python 20260830_citrus_light_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite smoke --epochs 3
    python 20260830_citrus_light_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite screen --epochs 50
    python 20260830_citrus_light_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite pareto --epochs 50
    python 20260830_citrus_light_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite pr --epochs 50
    python 20260830_citrus_light_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite final --only Light03_deploy_lite --epochs 300
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from citrus_protocol import fixed_train_args, validate_locked_runtime, write_protocol_snapshot


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "Light_series"
FIXED_TRAIN = fixed_train_args()


@dataclass(frozen=True)
class Experiment:
    """One controlled Light architecture experiment."""

    name: str
    yaml: str
    hypothesis: str
    train_overrides: tuple[tuple[str, float], ...] = ()


STRUCTURE_EXPERIMENTS = (
    Experiment("Light00_backbone_only", "Light00_backbone_only.yaml", "deep-only partial-convolution compression"),
    Experiment("Light01_neck_only", "Light01_neck_only.yaml", "near-identity progressive neck"),
    Experiment("Light02_joint_core", "Light02_joint_core.yaml", "PConv-by-AFPN factorial interaction"),
    Experiment(
        "Light05_repmixer_backbone_only",
        "Light05_repmixer_backbone_only.yaml",
        "clean G04-supported deep RepMixer isolation on official PAN",
    ),
    Experiment(
        "Light06_repmixer_afpn",
        "Light06_repmixer_afpn.yaml",
        "RepMixer-by-AFPN factorial interaction",
    ),
)

DEPLOY_EXPERIMENTS = (
    Experiment(
        "Light03_deploy_lite",
        "Light03_deploy_lite.yaml",
        "aggressive PConv plus AFPN lightweight Pareto candidate",
    ),
    Experiment(
        "Light04_quality_rank",
        "Light04_quality_rank.yaml",
        "Mask Scoring-style hypothesis ranking on the Light03 graph",
        (("citrus_quality", 0.25),),
    ),
    Experiment(
        "Light07_repmixer_pan_lite",
        "Light07_repmixer_pan_lite.yaml",
        "conservative G04-supported RepMixer plus official PAN deployment candidate",
    ),
)

EXPERIMENTS = (*STRUCTURE_EXPERIMENTS, *DEPLOY_EXPERIMENTS)

PR_EXPERIMENTS = (
    Experiment("LightP00_lite_bce", "Light03_deploy_lite.yaml", "stock BCE ranking control"),
    Experiment(
        "LightP01_lite_vfl025",
        "Light03_deploy_lite.yaml",
        "Varifocal ranking at a conservative blend",
        (("citrus_vfl", 0.25),),
    ),
    Experiment(
        "LightP02_lite_nwd025",
        "Light03_deploy_lite.yaml",
        "tiny-box NWD localization without ranking changes",
        (("nwd_ratio", 0.25),),
    ),
    Experiment(
        "LightP03_lite_nwd_vfl",
        "Light03_deploy_lite.yaml",
        "factorial NWD plus Varifocal interaction",
        (("nwd_ratio", 0.25), ("citrus_vfl", 0.25)),
    ),
    Experiment(
        "LightP04_mask_quality",
        "Light04_quality_rank.yaml",
        "explicit mask-IoU calibration",
        (("citrus_quality", 0.25),),
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the locked sequential training interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset data.yaml; the path is supplied by you.")
    parser.add_argument(
        "--suite",
        choices=("smoke", "screen", "pareto", "pr", "all", "final"),
        default="screen",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=FIXED_TRAIN["batch"])
    parser.add_argument("--imgsz", type=int, default=FIXED_TRAIN["imgsz"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=FIXED_TRAIN["workers"])
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds; final paper runs use 42,43,44.")
    parser.add_argument("--only", default="", help="Comma-separated exact experiment names.")
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable AMP only for an explicitly paired audit; formal comparisons are AMP-off.",
    )
    amp_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Explicitly disable AMP (the formal protocol default).",
    )
    parser.set_defaults(amp=FIXED_TRAIN["amp"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def choose_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Return a deterministic experiment queue."""
    if args.suite == "pr":
        selected = list(PR_EXPERIMENTS)
    elif args.suite == "screen":
        selected = list(STRUCTURE_EXPERIMENTS)
    elif args.suite == "pareto":
        selected = list(DEPLOY_EXPERIMENTS)
    elif args.suite == "final":
        selected = list(DEPLOY_EXPERIMENTS)
    else:
        selected = list(EXPERIMENTS)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        all_experiments = (*EXPERIMENTS, *PR_EXPERIMENTS)
        known = {experiment.name for experiment in all_experiments}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}; choices: {sorted(known)}")
        selected = [experiment for experiment in all_experiments if experiment.name in requested]
    return selected


def set_seed(seed: int) -> None:
    """Lock all random sources before constructing each model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def snapshot_data_yaml(source: Path, project: Path) -> Path:
    """Save the exact user-supplied YAML without changing its path semantics."""
    content = source.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    target = project / "_protocol" / "dataset_source_snapshot.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def repository_state() -> dict[str, str | bool]:
    """Record the exact source state used by the run."""

    def run(*command: str) -> str:
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def append_event(path: Path, event: dict) -> None:
    """Append one durable JSON event to the run ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def transfer_compatible_head(target, source_head_state: dict[str, torch.Tensor]) -> int:
    """Transfer shape-compatible official Segment tensors after neck replacement."""
    target_head = target.model.model[-1]
    target_state = target_head.state_dict()
    compatible = {
        key: value
        for key, value in source_head_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    target_head.load_state_dict(compatible, strict=False)
    return len(compatible)


def main() -> None:
    """Build or train Light models sequentially on one device."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    default_project = ROOT / "1_results" / "Light_series" / f"CITRUS_LIGHT_{args.suite.upper()}_{args.epochs}EP"
    project = (args.project or default_project).expanduser().resolve()
    cache: bool | str = False if args.cache == "false" else args.cache
    deviations = validate_locked_runtime(
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=cache,
        amp=args.amp,
        allow_amp_audit=args.amp,
    )
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.imgsz != 640 and not args.dry_run:
        raise ValueError("Formal Light comparisons lock imgsz=640.")
    if args.suite == "smoke" and not 1 <= args.epochs <= 3:
        raise ValueError("Smoke runs must use 1--3 epochs.")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    experiments = choose_experiments(args)

    from ultralytics import YOLO

    if args.dry_run:
        for experiment in experiments:
            model = YOLO(str(YAML_DIR / experiment.yaml), task="segment", verbose=False)
            parameters = sum(parameter.numel() for parameter in model.model.parameters())
            print(f"BUILD OK {experiment.name}: params={parameters:,}")
        return

    source = YOLO(str(pretrained), task="segment", verbose=False)
    source_head_state = {key: value.detach().cpu() for key, value in source.model.model[-1].state_dict().items()}
    del source
    data_snapshot = snapshot_data_yaml(data, project)
    ledger = project / "_protocol" / "ledger.jsonl"
    state = repository_state()
    protocol_path, protocol_hash = write_protocol_snapshot(
        project,
        {
            "series": "Light",
            "architecture_goal": "lower measured latency and complexity with competitive mask accuracy",
            "experiment_specific_overrides": {
                experiment.name: dict(experiment.train_overrides) for experiment in experiments
            },
            "declared_protocol_deviations": deviations,
        },
    )

    for experiment in experiments:
        model_path = YAML_DIR / experiment.yaml
        if not model_path.is_file():
            raise FileNotFoundError(f"Model YAML not found: {model_path}")
        for seed in seeds:
            run_name = experiment.name if len(seeds) == 1 else f"{experiment.name}_seed{seed}"
            run_dir = project / run_name
            complete = (run_dir / "weights" / "best.pt").is_file() and (run_dir / "results.csv").is_file()
            if complete and args.skip_completed:
                print(f"SKIP completed: {run_name}")
                continue
            if run_dir.exists():
                raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")

            event = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "started",
                "run_name": run_name,
                "experiment": asdict(experiment),
                "seed": seed,
                "epochs": args.epochs,
                "data": str(data),
                "dataset_yaml_snapshot": str(data_snapshot),
                "pretrained": str(pretrained),
                "repository": state,
                "formal_protocol": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "protocol_deviations": deviations,
            }
            append_event(ledger, event)
            model = None
            try:
                set_seed(seed)
                model = YOLO(str(model_path), task="segment", verbose=False).load(str(pretrained))
                event["compatible_head_tensors_transferred"] = transfer_compatible_head(model, source_head_state)
                training = fixed_train_args()
                training.update(
                    data=str(data),
                    project=str(project),
                    name=run_name,
                    epochs=args.epochs,
                    device=args.device,
                    seed=seed,
                    exist_ok=False,
                )
                training.update(dict(experiment.train_overrides))
                if deviations:
                    training["amp"] = args.amp
                model.train(**training)
                append_event(ledger, {**event, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "completed"})
            except Exception as error:
                append_event(
                    ledger,
                    {**event, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "failed", "error": repr(error)},
                )
                if args.fail_fast:
                    raise
                print(f"FAILED {run_name}: {error!r}")
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
