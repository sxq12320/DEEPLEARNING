"""Batch runner for the evidence-grounded CitrusTopo-Seg experiment matrix.

The default ``architectures`` suite trains ten controlled YAMLs for 300 epochs. Use ``losses``
to isolate auxiliary losses on one fixed full architecture, or ``all`` to run both.
Every run uses the same optimizer, initialization, seed, image size, and dataset.

Server example:
    python 20260824_citrus_topo_batch.py \
        --data /data/orange_yolo_grouped_dedup_20260820/data.yaml \
        --suite architectures --epochs 300 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
YAML_DIR = ROOT / "0_orange_yaml" / "L_series"
DEFAULT_PROJECT = ROOT / "1_results" / "L_series" / "grouped_clean_300ep"

CORE_TOPO_LOSS = {
    "citrus_boundary": 0.50,
    "citrus_concavity": 0.00,
    "citrus_query": 0.10,
    "citrus_exclusive": 0.00,
}
FULL_TOPO_LOSS = {
    "citrus_boundary": 0.50,
    "citrus_concavity": 0.25,
    "citrus_query": 0.10,
    "citrus_exclusive": 0.10,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    yaml: str
    losses: dict[str, float]
    purpose: str


ARCHITECTURE_EXPERIMENTS = (
    Experiment("A00_reference", "00_yolo11n_seg_reference.yaml", {}, "YOLO11n-seg protocol reference"),
    Experiment("A01_lska", "01_lska_context.yaml", {}, "P5 large-kernel global context"),
    Experiment("A02_scale", "02_scale_fusion.yaml", {}, "P3 adaptive cross-scale neck fusion"),
    Experiment("A03_p2cfs", "03_p2cfs_detail.yaml", {}, "older P2 mask-detail control"),
    Experiment("A04_topology", "04_topology_head.yaml", CORE_TOPO_LOSS, "P2 query and mutual boundary head"),
    Experiment("A05_lska_scale", "05_lska_scale_fusion.yaml", {}, "backbone plus neck pair"),
    Experiment("A06_lska_topology", "06_lska_topology.yaml", CORE_TOPO_LOSS, "backbone plus topology head"),
    Experiment(
        "A07_full_core",
        "07_citrus_toposeg_full.yaml",
        CORE_TOPO_LOSS,
        "full architecture with core supervision",
    ),
    Experiment("A08_scale_topology", "08_scale_topology.yaml", CORE_TOPO_LOSS, "neck plus topology head"),
    Experiment("A09_full_p2cfs", "09_full_p2cfs_control.yaml", {}, "full structure with older P2 head control"),
)

LOSS_EXPERIMENTS = (
    Experiment("L00_no_aux", "07_citrus_toposeg_full.yaml", {}, "full architecture without auxiliary supervision"),
    Experiment(
        "L01_boundary",
        "07_citrus_toposeg_full.yaml",
        {"citrus_boundary": 0.50},
        "per-instance boundary BCE plus Dice",
    ),
    Experiment(
        "L02_boundary_query",
        "07_citrus_toposeg_full.yaml",
        {"citrus_boundary": 0.50, "citrus_query": 0.10},
        "boundary plus sparse small-object query",
    ),
    Experiment(
        "L03_boundary_concavity",
        "07_citrus_toposeg_full.yaml",
        {"citrus_boundary": 0.50, "citrus_concavity": 0.25},
        "boundary plus concave-notch focus",
    ),
    Experiment(
        "L04_boundary_exclusive",
        "07_citrus_toposeg_full.yaml",
        {"citrus_boundary": 0.50, "citrus_exclusive": 0.10},
        "boundary plus adjacent-instance leakage suppression",
    ),
    Experiment("L05_full_loss", "07_citrus_toposeg_full.yaml", FULL_TOPO_LOSS, "all task-specific losses"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch train the CitrusTopo-Seg controlled experiment matrix.")
    parser.add_argument("--data", required=True, help="Server path to grouped-dedup data.yaml.")
    parser.add_argument("--suite", choices=("architectures", "losses", "all", "final"), default="architectures")
    parser.add_argument("--epochs", type=int, default=300, help="Epochs for every selected model (default: 300).")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--pretrained", default=str(ROOT / "yolo11n-seg.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="", help="Comma-separated experiment names to run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print the queue without training.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": str(bool(run("git", "status", "--porcelain"))),
    }


def select_experiments(suite: str) -> list[Experiment]:
    if suite == "architectures":
        return list(ARCHITECTURE_EXPERIMENTS)
    if suite == "losses":
        return list(LOSS_EXPERIMENTS)
    if suite == "final":
        return [ARCHITECTURE_EXPERIMENTS[0], LOSS_EXPERIMENTS[-1]]
    return [*ARCHITECTURE_EXPERIMENTS, *LOSS_EXPERIMENTS]


def is_complete(run_dir: Path, epochs: int) -> bool:
    csv_path = run_dir / "results.csv"
    best_path = run_dir / "weights" / "best.pt"
    if not csv_path.exists() or not best_path.exists():
        return False
    try:
        with csv_path.open(encoding="utf-8", errors="ignore") as handle:
            completed_epochs = max(sum(1 for _ in handle) - 1, 0)
        return completed_epochs >= epochs
    except OSError:
        return False


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_runtime_data_yaml(source: Path, project: Path) -> Path:
    """Create a portable dataset YAML whose root is the source YAML's server directory."""
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "train" not in config or "val" not in config or "names" not in config:
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    config["path"] = str(source.parent.resolve())
    for split in ("train", "val", "test"):
        value = config.get(split)
        if value and not Path(value).is_absolute() and not (source.parent / value).exists():
            raise FileNotFoundError(f"Dataset split path does not exist: {source.parent / value}")
    runtime_yaml = project / "_protocol" / "grouped_dedup_runtime.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    with runtime_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return runtime_yaml


def main() -> None:
    args = parse_args()
    data = Path(args.data).expanduser().resolve()
    pretrained = Path(args.pretrained).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.imgsz != 640:
        raise ValueError("Formal CitrusTopo experiments lock imgsz=640; use a separate smoke command for other sizes.")

    experiments = select_experiments(args.suite)
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}
        experiments = [experiment for experiment in experiments if experiment.name in selected]
        missing = selected - {experiment.name for experiment in experiments}
        if missing:
            raise ValueError(f"Unknown or out-of-suite experiment names: {sorted(missing)}")

    print(f"Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    print(f"Dataset: {data}\nProject: {args.project.resolve()}\nQueue ({len(experiments)} runs):")
    for experiment in experiments:
        print(f"  {experiment.name:<24} {experiment.yaml:<38} {experiment.losses} | {experiment.purpose}")
    if args.dry_run:
        return

    from ultralytics import YOLO

    args.project.mkdir(parents=True, exist_ok=True)
    runtime_data = prepare_runtime_data_yaml(data, args.project)
    print(f"Portable runtime dataset YAML: {runtime_data}")
    ledger = args.project / "experiment_ledger.jsonl"
    common = {
        "data": str(runtime_data),
        "project": str(args.project.resolve()),
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
        "amp": False,
        "seed": args.seed,
        "deterministic": True,
        "cache": False,
        "exist_ok": False,
    }
    repository = git_state()

    for index, experiment in enumerate(experiments, start=1):
        yaml_path = YAML_DIR / experiment.yaml
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Model YAML not found: {yaml_path}")
        run_dir = args.project / experiment.name
        if is_complete(run_dir, args.epochs):
            print(f"[{index}/{len(experiments)}] SKIP complete: {experiment.name}")
            continue
        set_seed(args.seed)
        record = {
            "experiment": asdict(experiment),
            "yaml": str(yaml_path.resolve()),
            "pretrained": str(pretrained),
            "source_data_yaml": str(data),
            "protocol": common,
            "repository": repository,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "running",
        }
        append_ledger(ledger, record)
        print(f"[{index}/{len(experiments)}] START {experiment.name}")
        model = None
        try:
            last_path = run_dir / "weights" / "last.pt"
            if run_dir.exists() and last_path.is_file():
                print(f"[{index}/{len(experiments)}] RESUME {experiment.name}: {last_path}")
                record["resumed_from"] = str(last_path.resolve())
                model = YOLO(str(last_path))
                model.train(resume=True)
            elif run_dir.exists():
                raise RuntimeError(
                    f"Incomplete run directory has no resumable last.pt: {run_dir}. "
                    "Inspect it and move it aside before retrying."
                )
            else:
                model = YOLO(str(yaml_path)).load(str(pretrained))
                model.train(name=experiment.name, **common, **experiment.losses)
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = repr(error)
            append_ledger(ledger, {**record, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(f"[{index}/{len(experiments)}] FAILED {experiment.name}: {error!r}")
            if args.fail_fast:
                raise
            continue
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        append_ledger(ledger, record)
        print(f"[{index}/{len(experiments)}] DONE {experiment.name}")


if __name__ == "__main__":
    # Avoid excessive OpenMP thread contention when the runner is launched repeatedly on a server.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
