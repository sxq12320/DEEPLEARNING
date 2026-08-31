"""Controlled sequential runner for the evidence-driven G_0830 citrus series.

Examples:
    python 20260830_citrus_g0830_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite smoke --epochs 3
    python 20260830_citrus_g0830_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite structure --epochs 50
    python 20260830_citrus_g0830_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --epochs 300
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
YAML_DIR = ROOT / "0_orange_yaml" / "G_0830_series"
FIXED_TRAIN = fixed_train_args()
TOPOLOGY_GAINS = {"citrus_boundary": 0.15, "citrus_query": 0.03}


@dataclass(frozen=True)
class Experiment:
    """One architecture or loss ablation with an explicit causal question."""

    name: str
    yaml: str
    family: str
    losses: dict[str, float]
    hypothesis: str


EXPERIMENTS = (
    Experiment(
        "G00_official_control",
        "00_g00_official_control.yaml",
        "structure",
        {},
        "complete the failed T00 official baseline under the same protocol",
    ),
    Experiment(
        "G01_t04_anchor",
        "01_g01_t04_anchor.yaml",
        "structure",
        TOPOLOGY_GAINS,
        "retest the most efficient T-series topology-head anchor",
    ),
    Experiment(
        "G02_bilateral_backbone",
        "02_g02_bilateral_backbone.yaml",
        "structure",
        TOPOLOGY_GAINS,
        "preserve narrow P2 shape evidence through three semantic stages",
    ),
    Experiment(
        "G03_frequency_neck",
        "03_g03_frequency_neck.yaml",
        "structure",
        TOPOLOGY_GAINS,
        "align low-frequency semantics and high-frequency lateral detail before PAN fusion",
    ),
    Experiment(
        "G04_deep_repmixer",
        "04_g04_deep_repmixer.yaml",
        "structure",
        TOPOLOGY_GAINS,
        "test a lighter non-CSP P4/P5 micro-architecture without changing the macro topology",
    ),
    Experiment(
        "G05_g03_nwd",
        "03_g03_frequency_neck.yaml",
        "loss",
        {**TOPOLOGY_GAINS, "nwd_ratio": 0.25},
        "reduce one-pixel localization sensitivity for targets below roughly 32 pixels",
    ),
    Experiment(
        "G06_g03_vfl",
        "03_g03_frequency_neck.yaml",
        "loss",
        {**TOPOLOGY_GAINS, "citrus_vfl": 0.50},
        "improve quality-aware candidate ranking at the high-recall PR elbow",
    ),
    Experiment(
        "G07_g03_nwd_vfl",
        "03_g03_frequency_neck.yaml",
        "loss",
        {**TOPOLOGY_GAINS, "nwd_ratio": 0.25, "citrus_vfl": 0.50},
        "test whether tiny localization robustness and confidence ranking are complementary",
    ),
    Experiment(
        "G08_g03_strong_aux_audit",
        "03_g03_frequency_neck.yaml",
        "loss",
        {"citrus_boundary": 0.50, "citrus_query": 0.10},
        "audit whether T04 benefited from its larger auxiliary weights rather than architecture",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the sequential, fixed-protocol experiment queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Exact server data.yaml; no fingerprint gate.")
    parser.add_argument("--suite", choices=("smoke", "structure", "loss", "all", "final"), default="structure")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=FIXED_TRAIN["batch"])
    parser.add_argument("--imgsz", type=int, default=FIXED_TRAIN["imgsz"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=FIXED_TRAIN["workers"])
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42", help="Comma-separated; final reporting uses 42,43,44.")
    parser.add_argument("--only", default="", help="Comma-separated experiment names, overriding --suite selection.")
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def choose_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Select a deterministic queue while keeping every model definition unchanged."""
    if args.suite in {"smoke", "structure"}:
        selected = [experiment for experiment in EXPERIMENTS if experiment.family == "structure"]
    elif args.suite == "loss":
        selected = [experiment for experiment in EXPERIMENTS if experiment.family == "loss"]
    elif args.suite == "final":
        selected = [experiment for experiment in EXPERIMENTS if experiment.name in {"G03_frequency_neck", "G04_deep_repmixer"}]
    else:
        selected = list(EXPERIMENTS)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        known = {experiment.name for experiment in EXPERIMENTS}
        if unknown := requested - known:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}; choices: {sorted(known)}")
        selected = [experiment for experiment in EXPERIMENTS if experiment.name in requested]
    return selected


def set_seed(seed: int) -> None:
    """Lock Python, NumPy, PyTorch, CUDA, and cuDNN random behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_yaml(source: Path, project: Path) -> Path:
    """Bind a user-supplied dataset root without changing split membership."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    config["path"] = str(source.parent.resolve())
    target = project / "_protocol" / "runtime_data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def repository_state() -> dict[str, str | bool]:
    """Capture the exact source commit and dirty flag used by training."""

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


def main() -> None:
    """Build or train each selected model sequentially on one device."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    default_project = ROOT / "1_results" / "G_0830_series" / f"CITRUS_G0830_{args.suite.upper()}_{args.epochs}EP"
    project = (args.project or default_project).expanduser().resolve()
    cache: bool | str = False if args.cache == "false" else args.cache
    deviations = validate_locked_runtime(
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=cache,
        amp=FIXED_TRAIN["amp"],
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
            print(f"BUILD OK {experiment.name}: params={parameters:,}, losses={experiment.losses}")
        return

    runtime_data = prepare_data_yaml(data, project)
    ledger = project / "_protocol" / "ledger.jsonl"
    protocol_path, protocol_hash = write_protocol_snapshot(
        project,
        {
            "series": "G_0830",
            "architecture_loss_separation": True,
            "declared_protocol_deviations": deviations,
        },
    )
    state = repository_state()

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
                "source_data": str(data),
                "runtime_data": str(runtime_data),
                "pretrained": str(pretrained),
                "repository": state,
                "formal_protocol": str(protocol_path),
                "protocol_sha256": protocol_hash,
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
                    **experiment.losses,
                )
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
