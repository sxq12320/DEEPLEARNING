"""Controlled sequential runner for the CitrusD shape-semantic backbone series.

Examples:
    python 20260828_citrus_d_batch.py --data /data/orange/data.yaml --suite smoke --epochs 3
    python 20260828_citrus_d_batch.py --data /data/orange/data.yaml --suite controls --epochs 50
    python 20260828_citrus_d_batch.py --data /data/orange/data.yaml --suite core --epochs 50
    python 20260828_citrus_d_batch.py --data /data/orange/data.yaml --suite architectures --epochs 50
    python 20260828_citrus_d_batch.py --data /data/orange/data.yaml --suite core --epochs 300 --only D06_shape_semantic_full
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "D_series"
LOCAL_DATA = ROOT.parents[2] / "data" / "orange_yolo_grouped_dedup_20260820" / "data.yaml"

EDGE_QUERY = {"citrus_boundary": 0.15, "citrus_query": 0.03}
TOPOLOGY = {"citrus_topology": 0.05}


@dataclass(frozen=True)
class Experiment:
    """One controlled architecture or loss hypothesis."""

    name: str
    yaml: str
    losses: dict[str, float]
    purpose: str


ARCHITECTURES = (
    Experiment(
        "D01_bilateral_pdc_core",
        "01_d01_bilateral_pdc_core.yaml",
        {},
        "persistent P2 shape stream with P3/P4/P5 semantic gates",
    ),
    Experiment(
        "D02_regular_conv_control",
        "02_d02_regular_conv_control.yaml",
        {},
        "replace pixel differences by ordinary depthwise convolution",
    ),
    Experiment(
        "D03_p3_gate_only",
        "03_d03_p3_gate_only.yaml",
        {},
        "remove P4/P5 gates to test whether global semantics help",
    ),
    Experiment(
        "D04_achromatic_stem",
        "04_d04_achromatic_stem.yaml",
        {},
        "add luminance structure stem while retaining the RGB path",
    ),
    Experiment(
        "D05_edge_supervised",
        "05_d05_edge_supervised.yaml",
        EDGE_QUERY,
        "training-only visible-boundary and tiny-centre supervision",
    ),
    Experiment(
        "D06_shape_semantic_full",
        "06_d06_shape_semantic_full.yaml",
        EDGE_QUERY,
        "primary accuracy hypothesis: D04 structure plus D05 supervision",
    ),
    Experiment(
        "D07_deploy_lite",
        "07_d07_deploy_lite.yaml",
        EDGE_QUERY,
        "D06 backbone with the empirically supported lightweight mask head",
    ),
    Experiment(
        "D08_topology_masks",
        "08_d08_topology_masks.yaml",
        TOPOLOGY,
        "D06 backbone with detail/semantic prototypes for split-merge errors",
    ),
    Experiment(
        "D09_empirical_context",
        "09_d09_empirical_context.yaml",
        EDGE_QUERY,
        "D06 plus the only context operator supported by clean B-series evidence",
    ),
)

LOSSES = (
    Experiment("DL00_aux_off", "06_d06_shape_semantic_full.yaml", {}, "architecture without auxiliary objectives"),
    Experiment(
        "DL01_edge_query_mild",
        "06_d06_shape_semantic_full.yaml",
        {"citrus_boundary": 0.10, "citrus_query": 0.02},
        "mild auxiliary supervision",
    ),
    Experiment("DL02_edge_query_default", "06_d06_shape_semantic_full.yaml", EDGE_QUERY, "default supervision"),
    Experiment(
        "DL03_edge_query_strong",
        "06_d06_shape_semantic_full.yaml",
        {"citrus_boundary": 0.25, "citrus_query": 0.05},
        "strong B-series-compatible supervision",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the locked comparison protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LOCAL_DATA)
    parser.add_argument(
        "--suite",
        choices=("smoke", "controls", "core", "architectures", "losses"),
        default="core",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--only", default="", help="Comma-separated experiment names.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def experiments_for(suite: str) -> list[Experiment]:
    """Return a deterministic queue with causal controls before full candidates."""
    if suite == "controls":
        return [ARCHITECTURES[index] for index in (0, 1, 2, 3)]
    if suite == "core":
        return [ARCHITECTURES[index] for index in (0, 3, 4, 5, 6, 7, 8)]
    if suite == "losses":
        return list(LOSSES)
    return list(ARCHITECTURES)


def set_seed(seed: int) -> None:
    """Lock random sources used by the training stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def repository_state() -> dict[str, str]:
    """Record the exact source state for every run."""

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": str(bool(run("git", "status", "--porcelain")))}


def prepare_data_yaml(source: Path, project: Path) -> Path:
    """Bind the server dataset root without changing its split membership."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    config["path"] = str(source.parent.resolve())
    target = project / "_protocol" / "runtime_data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def append_jsonl(path: Path, record: dict) -> None:
    """Append one durable experiment event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed(ledger: Path, run_name: str, run_dir: Path, epochs: int) -> bool:
    """Recognize a completed run without overwriting it."""
    if not (run_dir / "weights" / "best.pt").is_file() or not (run_dir / "results.csv").is_file():
        return False
    if not ledger.is_file():
        return False
    for line in reversed(ledger.read_text(encoding="utf-8", errors="ignore").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("run_name") == run_name and event.get("status") == "completed":
            return int(event.get("requested_epochs", -1)) == epochs
    return False


def main() -> None:
    """Run selected D experiments sequentially on one GPU."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    default_project = ROOT / "1_results" / "D_series" / f"CITRUS_D_{args.suite.upper()}_{args.epochs}EP"
    project = (args.project or default_project).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.imgsz != 640:
        raise ValueError("Formal CitrusD comparisons lock imgsz=640.")
    if args.suite == "smoke" and args.epochs > 3:
        raise ValueError("Smoke runs are limited to 1-3 epochs.")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    experiments = experiments_for(args.suite)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        available = {experiment.name for experiment in experiments}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown or out-of-suite experiments: {sorted(unknown)}")
        experiments = [experiment for experiment in experiments if experiment.name in requested]
    queue = [(experiment, seed) for experiment in experiments for seed in seeds]
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    print(f"Dataset: {data}\nProject: {project}\nQueue ({len(queue)} sequential runs):")
    for experiment, seed in queue:
        print(f"  {experiment.name:<27} seed={seed:<4} {experiment.yaml:<42} {experiment.losses}")
    if args.dry_run:
        return

    from ultralytics import YOLO

    project.mkdir(parents=True, exist_ok=True)
    runtime_data = prepare_data_yaml(data, project)
    ledger = project / "experiment_ledger.jsonl"
    repo = repository_state()
    for index, (experiment, seed) in enumerate(queue, 1):
        yaml_path = YAML_DIR / experiment.yaml
        if not yaml_path.is_file():
            raise FileNotFoundError(f"D model YAML not found: {yaml_path}")
        run_name = experiment.name if len(seeds) == 1 else f"{experiment.name}_seed{seed}"
        run_dir = project / run_name
        if completed(ledger, run_name, run_dir, args.epochs):
            print(f"[{index}/{len(queue)}] SKIP completed: {run_name}")
            continue

        protocol = {
            "data": str(runtime_data),
            "project": str(project),
            "name": run_name,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 100,
            "close_mosaic": 10,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "amp": False,
            "seed": seed,
            "deterministic": True,
            "cache": False,
            "exist_ok": False,
            **experiment.losses,
        }
        event = {
            "experiment": asdict(experiment),
            "run_name": run_name,
            "seed": seed,
            "requested_epochs": args.epochs,
            "yaml": str(yaml_path.resolve()),
            "pretrained": str(pretrained),
            "source_data_yaml": str(data),
            "protocol": protocol,
            "repository": repo,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "running",
        }
        append_jsonl(ledger, event)
        model = None
        try:
            last = run_dir / "weights" / "last.pt"
            if run_dir.exists() and last.is_file():
                print(f"[{index}/{len(queue)}] RESUME {run_name}")
                model = YOLO(str(last), task="segment")
                model.train(resume=True)
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no weights/last.pt: {run_dir}")
            else:
                print(f"[{index}/{len(queue)}] START {run_name}")
                model = YOLO(str(yaml_path), task="segment").load(str(pretrained))
                set_seed(seed)
                model.train(**protocol)
            append_jsonl(ledger, {**event, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "completed"})
        except Exception as error:
            append_jsonl(
                ledger,
                {
                    **event,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "failed",
                    "error": repr(error),
                },
            )
            if args.fail_fast:
                raise
            print(f"[{index}/{len(queue)}] FAILED {run_name}: {error}")
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
