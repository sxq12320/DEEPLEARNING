"""Sequential batch runner for the G_0839 citrus SDR ablation series.

Examples:
    python 20260830_citrus_g0839_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite smoke --epochs 3
    python 20260830_citrus_g0839_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite screen --epochs 50
    python 20260830_citrus_g0839_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --epochs 300
    python 20260830_citrus_g0839_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --only G04_boundary_refine,G05_full_sdr --epochs 300
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
YAML_DIR = ROOT / "0_orange_yaml" / "G_0839_series"
FIXED_TRAIN = fixed_train_args()
G0839_AUXILIARY_GAINS = {
    "citrus_query": 0.03,
    "citrus_contrast": 0.05,
    "citrus_boundary": 0.10,
    "citrus_topology": 0.05,
}


@dataclass(frozen=True)
class Experiment:
    """One controlled G_0839 architecture stage."""

    name: str
    yaml: str
    losses: dict[str, float]
    hypothesis: str


EXPERIMENTS = (
    Experiment("G00_lite_control", "00_g00_lite_control.yaml", {}, "controlled lightweight head"),
    Experiment("G01_preserve", "01_g01_preserve.yaml", {}, "persistent P2 shape preservation"),
    Experiment("G02_search", "02_g02_search.yaml", {"citrus_query": 0.03}, "coarse tiny-fruit search"),
    Experiment(
        "G03_discriminate",
        "03_g03_discriminate.yaml",
        {"citrus_query": 0.03, "citrus_contrast": 0.05},
        "fruit-inner versus context-ring discrimination",
    ),
    Experiment(
        "G04_boundary_refine",
        "04_g04_boundary_refine.yaml",
        {"citrus_query": 0.03, "citrus_contrast": 0.05, "citrus_boundary": 0.10},
        "visible-boundary prototype refinement",
    ),
    Experiment(
        "G05_full_sdr",
        "05_g05_full_sdr.yaml",
        {
            "citrus_query": 0.03,
            "citrus_contrast": 0.05,
            "citrus_boundary": 0.10,
            "citrus_topology": 0.05,
        },
        "context/interior/boundary/separator topology",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the locked sequential experiment protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Your dataset data.yaml; no fingerprint gate.")
    parser.add_argument("--suite", choices=("smoke", "screen", "all", "final"), default="screen")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=FIXED_TRAIN["batch"])
    parser.add_argument("--imgsz", type=int, default=FIXED_TRAIN["imgsz"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=FIXED_TRAIN["workers"])
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds; use 42,43,44 for final repeats.")
    parser.add_argument("--only", default="", help="Comma-separated experiment names.")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Use only for a paired AMP control; the formal grouped-clean protocol is AMP-off.",
    )
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=FIXED_TRAIN["amp"])
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def choose_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Select the deterministic queue without changing model definitions."""
    selected = list(EXPERIMENTS[-2:] if args.suite == "final" else EXPERIMENTS)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        known = {experiment.name for experiment in EXPERIMENTS}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}; choices: {sorted(known)}")
        selected = [experiment for experiment in EXPERIMENTS if experiment.name in requested]
    return selected


def set_seed(seed: int) -> None:
    """Lock random sources for each independent run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_yaml(source: Path, project: Path) -> Path:
    """Bind the supplied dataset root while preserving its split membership."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    config["path"] = str(source.parent.resolve())
    target = project / "_protocol" / "runtime_data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def repository_state() -> dict[str, str | bool]:
    """Return the source commit and dirty flag for the run ledger."""

    def run(*command: str) -> str:
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def append_event(path: Path, event: dict) -> None:
    """Append one durable JSON event to the experiment ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    """Build or train the requested G_0839 models one at a time on one device."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    default_project = ROOT / "1_results" / "G_0839_series" / f"CITRUS_G0839_{args.suite.upper()}_{args.epochs}EP"
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
        raise ValueError("Formal G_0839 comparisons lock imgsz=640.")
    if args.suite == "smoke" and not 1 <= args.epochs <= 3:
        raise ValueError("Smoke runs must use 1--3 epochs.")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    experiments = choose_experiments(args)

    from ultralytics import YOLO

    if args.dry_run:
        for experiment in experiments:
            model_path = YAML_DIR / experiment.yaml
            model = YOLO(str(model_path), task="segment", verbose=False)
            parameters = sum(parameter.numel() for parameter in model.model.parameters())
            print(
                f"BUILD OK {experiment.name}: params={parameters:,}, active_outputs={list(experiment.losses)}, "
                f"fixed_gain_vector={G0839_AUXILIARY_GAINS}"
            )
        return

    runtime_data = prepare_data_yaml(data, project)
    ledger = project / "_protocol" / "ledger.jsonl"
    state = repository_state()
    protocol_path, protocol_hash = write_protocol_snapshot(
        project,
        {
            "series": "G_0839",
            "auxiliary_gains_passed_to_every_model": G0839_AUXILIARY_GAINS,
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
                "runtime_data": str(runtime_data),
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
                training = fixed_train_args()
                training.update(
                    data=str(runtime_data),
                    project=str(project),
                    name=run_name,
                    epochs=args.epochs,
                    device=args.device,
                    seed=seed,
                    exist_ok=False,
                    **G0839_AUXILIARY_GAINS,
                )
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
