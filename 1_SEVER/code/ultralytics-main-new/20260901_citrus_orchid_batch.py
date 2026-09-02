"""Sequential batch runner for the task-decoupled ORCHID citrus series.

Examples:
    python 20260901_citrus_orchid_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite smoke --epochs 3
    python 20260901_citrus_orchid_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite screen --epochs 50
    python 20260901_citrus_orchid_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --epochs 300
    python 20260901_citrus_orchid_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite final --only ORCHID03_supervised_query_router --epochs 300
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
YAML_DIR = ROOT / "0_orange_yaml" / "ORCHID_series"
FIXED_TRAIN = fixed_train_args()


@dataclass(frozen=True)
class Experiment:
    """One controlled ORCHID experiment."""

    name: str
    yaml: str
    hypothesis: str
    train_overrides: tuple[tuple[str, float], ...] = ()


CONTROL = Experiment(
    "ORCHID00_official_control",
    "ORCHID00_official_control.yaml",
    "exact YOLO11n-seg paired control",
)

STRUCTURE_EXPERIMENTS = (
    Experiment(
        "ORCHID01_task_decoupled",
        "ORCHID01_task_decoupled.yaml",
        "separate detection PAN from the raw-backbone mask evidence path",
    ),
    Experiment(
        "ORCHID02_latent_query_router",
        "ORCHID02_latent_query_router.yaml",
        "mask loss alone learns where P2 detail should enter the prototype",
    ),
    Experiment(
        "ORCHID03_supervised_query_router",
        "ORCHID03_supervised_query_router.yaml",
        "QueryDet-style tiny-centre supervision stabilizes candidate routing",
        (("citrus_query", 0.10),),
    ),
    Experiment(
        "ORCHID04_single_canvas_neck",
        "ORCHID04_single_canvas_neck.yaml",
        "one P3 evidence canvas replaces recurrent top-down/bottom-up PAN fusion",
        (("citrus_query", 0.10),),
    ),
    Experiment(
        "ORCHID05_decam_reference",
        "ORCHID05_decam_reference.yaml",
        "candidate evidence is contrasted with a local leaf/branch reference",
        (("citrus_query", 0.10), ("citrus_contrast", 0.05)),
    ),
)

PARETO_EXPERIMENTS = (
    Experiment(
        "ORCHID06_query_router_lite",
        "ORCHID06_query_router_lite.yaml",
        "supervised ORCHID routing with one-block prediction towers",
        (("citrus_query", 0.10),),
    ),
)

EXPERIMENTS = (*STRUCTURE_EXPERIMENTS, *PARETO_EXPERIMENTS)
ALL_EXPERIMENTS = (CONTROL, *EXPERIMENTS)


def parse_args() -> argparse.Namespace:
    """Parse the fixed-protocol sequential interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Your server-side dataset YAML path.")
    parser.add_argument(
        "--suite",
        choices=("smoke", "screen", "pareto", "all", "control", "final"),
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
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Explicit AMP audit only.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=FIXED_TRAIN["amp"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def choose_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Return a deterministic queue; the old baseline is never run implicitly."""
    if args.suite == "smoke":
        selected = [STRUCTURE_EXPERIMENTS[2], STRUCTURE_EXPERIMENTS[3]]
    elif args.suite == "screen":
        selected = list(STRUCTURE_EXPERIMENTS)
    elif args.suite == "pareto":
        selected = [STRUCTURE_EXPERIMENTS[2], STRUCTURE_EXPERIMENTS[4], *PARETO_EXPERIMENTS]
    elif args.suite == "control":
        selected = [CONTROL]
    elif args.suite == "final":
        selected = [STRUCTURE_EXPERIMENTS[2], STRUCTURE_EXPERIMENTS[4], *PARETO_EXPERIMENTS]
    else:
        selected = list(EXPERIMENTS)

    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        known = {experiment.name for experiment in ALL_EXPERIMENTS}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}; choices: {sorted(known)}")
        selected = [experiment for experiment in ALL_EXPERIMENTS if experiment.name in requested]
    return selected


def set_seed(seed: int) -> None:
    """Lock all random sources before constructing a model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def snapshot_data_yaml(source: Path, project: Path) -> Path:
    """Record the user-selected dataset path and exact YAML without inventing a path or fingerprint gate."""
    content = source.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    target = project / "_protocol" / "dataset_source_snapshot.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def repository_state() -> dict[str, str | bool]:
    """Record the source revision used by the experiment."""

    def run(*command: str) -> str:
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def append_event(path: Path, event: dict) -> None:
    """Append one durable event to the run ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def transfer_compatible_head(target, source_state: dict[str, torch.Tensor]) -> int:
    """Copy every shape-compatible official Segment tensor into the custom head."""
    target_head = target.model.model[-1]
    target_state = target_head.state_dict()
    compatible = {
        key: value for key, value in source_state.items() if key in target_state and target_state[key].shape == value.shape
    }
    target_head.load_state_dict(compatible, strict=False)
    return len(compatible)


def main() -> None:
    """Build or train ORCHID models sequentially on one device."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    default_project = ROOT / "1_results" / "ORCHID_series" / f"CITRUS_ORCHID_{args.suite.upper()}_{args.epochs}EP"
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
            "series": "ORCHID",
            "architecture_goal": "candidate-conditioned task-decoupled fusion for tiny camouflaged citrus",
            "experiment_specific_overrides": {
                experiment.name: dict(experiment.train_overrides) for experiment in experiments
            },
            "declared_protocol_deviations": deviations,
        },
    )

    for experiment in experiments:
        model_path = YAML_DIR / experiment.yaml
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
