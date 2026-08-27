"""Batch runner for the S-result-adjusted CitrusB ablation and loss suites.

Examples:
    python 20260826_citrus_b_batch.py --data /data/orange_grouped/data.yaml --suite architectures
    python 20260826_citrus_b_batch.py --data /data/orange_grouped/data.yaml --suite smoke --epochs 3
    python 20260826_citrus_b_batch.py --data /data/orange_grouped/data.yaml --suite screening --epochs 50
    python 20260826_citrus_b_batch.py --data /data/orange_grouped/data.yaml --suite losses --epochs 50
    python 20260826_citrus_b_batch.py --data /data/orange_grouped/data.yaml --suite final --seeds 42,43,44
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
YAML_DIR = ROOT / "0_orange_yaml" / "20260826_citrus_b"
LOCAL_DATA = Path(r"E:\mastercode\data\orange_yolo_grouped_dedup_20260820\data.yaml")
DEFAULT_PROJECT = ROOT / "1_results" / "CITRUS_B_GROUPED_DEDUP_300EP"

BQ = {"citrus_boundary": 0.25, "citrus_query": 0.05}


@dataclass(frozen=True)
class Experiment:
    """One controlled model/loss run."""

    name: str
    yaml: str
    losses: dict[str, float]
    purpose: str


ARCHITECTURES = (
    Experiment("B00_reference", "00_b00_yolo11n_reference.yaml", {}, "cleaned-data YOLO11n-seg reference"),
    Experiment("B01_lite_head", "01_b01_lite_head_reference.yaml", {}, "replicate the S04 head result"),
    Experiment("B02_context_lite", "02_b02_repcontext_lite.yaml", {}, "recall-oriented RepContext plus lite head"),
    Experiment("B03_scale_lite", "03_b03_scalefusion_lite.yaml", {}, "isolate P3 scale fusion"),
    Experiment("B04_topology_lite", "04_b04_topology_lite.yaml", BQ, "isolate inference topology refinement"),
    Experiment("B05_context_scale", "05_b05_context_scale_lite.yaml", {}, "backbone-neck interaction"),
    Experiment("B06_context_topology", "06_b06_context_topology_lite.yaml", BQ, "backbone-head interaction"),
    Experiment("B07_scale_topology", "07_b07_scale_topology_lite.yaml", BQ, "neck-head interaction"),
    Experiment("B08_full_factorial", "08_b08_full_factorial.yaml", BQ, "all three factors with prototype refinement"),
    Experiment(
        "B09_recall_balanced",
        "09_b09_recall_balanced_final.yaml",
        BQ,
        "all factors with training-only topology supervision",
    ),
)

LOSSES = (
    Experiment("BL00_none", "09_b09_recall_balanced_final.yaml", {}, "no task-specific loss"),
    Experiment("BL01_boundary", "09_b09_recall_balanced_final.yaml", {"citrus_boundary": 0.25}, "boundary only"),
    Experiment("BL02_query", "09_b09_recall_balanced_final.yaml", {"citrus_query": 0.05}, "tiny query only"),
    Experiment("BL03_boundary_query", "09_b09_recall_balanced_final.yaml", BQ, "S09-supported B/Q pair"),
    Experiment("BL04_nwd010", "09_b09_recall_balanced_final.yaml", {"nwd_ratio": 0.10}, "small-box NWD 0.10"),
    Experiment("BL05_nwd025", "09_b09_recall_balanced_final.yaml", {"nwd_ratio": 0.25}, "small-box NWD 0.25"),
    Experiment("BL06_vfl025", "09_b09_recall_balanced_final.yaml", {"citrus_vfl": 0.25}, "BCE/VFL 0.25 blend"),
    Experiment("BL07_vfl050", "09_b09_recall_balanced_final.yaml", {"citrus_vfl": 0.50}, "BCE/VFL 0.50 blend"),
    Experiment("BL08_bq_nwd", "09_b09_recall_balanced_final.yaml", {**BQ, "nwd_ratio": 0.10}, "B/Q plus mild NWD"),
    Experiment("BL09_bq_vfl", "09_b09_recall_balanced_final.yaml", {**BQ, "citrus_vfl": 0.25}, "B/Q plus mild VFL"),
)


def parse_args() -> argparse.Namespace:
    """Parse the reproducible batch protocol."""
    parser = argparse.ArgumentParser(description="Train the CitrusB controlled ablation suite.")
    parser.add_argument("--data", type=Path, default=LOCAL_DATA, help="Grouped, deduplicated segmentation data.yaml.")
    parser.add_argument(
        "--suite",
        choices=("architectures", "smoke", "screening", "losses", "all", "final"),
        default="architectures",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,43,44.")
    parser.add_argument("--only", default="", help="Comma-separated experiment names.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def experiments_for(suite: str) -> list[Experiment]:
    """Return a deterministic queue for the selected suite."""
    if suite == "architectures":
        return list(ARCHITECTURES)
    if suite == "smoke":
        # B00 and B01 are exact S00/S04 controls already available under the
        # identical grouped-dedup protocol; smoke spends GPU time on new graphs only.
        return list(ARCHITECTURES[2:])
    if suite == "screening":
        # A 50-epoch screen must include same-duration controls; it cannot use
        # the existing 300-epoch S00/S04 metrics as numerical references.
        return list(ARCHITECTURES)
    if suite == "losses":
        return list(LOSSES)
    if suite == "all":
        return [*ARCHITECTURES, *LOSSES]
    return [ARCHITECTURES[0], ARCHITECTURES[-1]]


def set_seed(seed: int) -> None:
    """Lock random sources for a controlled comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def repository_state() -> dict[str, str]:
    """Record source identity without requiring a clean worktree."""
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": str(bool(run("git", "status", "--porcelain")))}


def prepare_data_yaml(source: Path, project: Path) -> Path:
    """Make dataset paths server-safe while preserving split membership."""
    with source.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not {"train", "val", "names"}.issubset(data):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    data["path"] = str(source.parent.resolve())
    target = project / "_protocol" / "grouped_dedup_runtime.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return target


def completed(run_dir: Path, epochs: int) -> bool:
    """Return true only for a run with weights and the requested epoch count."""
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file() or not (run_dir / "weights" / "best.pt").is_file():
        return False
    with csv_path.open(encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0) >= epochs


def append_jsonl(path: Path, record: dict) -> None:
    """Append a durable experiment event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run all selected experiments sequentially on one GPU."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    project = args.project.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.imgsz != 640:
        raise ValueError("Formal CitrusB comparisons lock imgsz=640.")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    experiments = experiments_for(args.suite)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        experiments = [experiment for experiment in experiments if experiment.name in requested]
        unknown = requested - {experiment.name for experiment in experiments}
        if unknown:
            raise ValueError(f"Unknown or out-of-suite experiments: {sorted(unknown)}")
    queue = [(experiment, seed) for experiment in experiments for seed in seeds]
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    print(f"Dataset: {data}\nProject: {project}\nQueue ({len(queue)} runs):")
    for experiment, seed in queue:
        print(f"  {experiment.name:<22} seed={seed:<4} {experiment.yaml:<42} {experiment.losses}")
    if args.dry_run:
        return

    from ultralytics import YOLO

    runtime_data = prepare_data_yaml(data, project)
    ledger = project / "experiment_ledger.jsonl"
    repo = repository_state()
    for index, (experiment, seed) in enumerate(queue, 1):
        yaml_path = YAML_DIR / experiment.yaml
        run_name = experiment.name if len(seeds) == 1 else f"{experiment.name}_seed{seed}"
        run_dir = project / run_name
        if completed(run_dir, args.epochs):
            print(f"[{index}/{len(queue)}] SKIP complete: {run_name}")
            continue
        common = {
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
        }
        event = {
            "experiment": asdict(experiment),
            "seed": seed,
            "yaml": str(yaml_path.resolve()),
            "pretrained": str(pretrained),
            "source_data_yaml": str(data),
            "protocol": common,
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
                model = YOLO(str(last))
                model.train(resume=True)
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no weights/last.pt: {run_dir}")
            else:
                print(f"[{index}/{len(queue)}] START {run_name}")
                model = YOLO(str(yaml_path), task="segment").load(str(pretrained))
                model.train(**common, **experiment.losses)
            event["status"] = "completed"
        except Exception as error:
            event["status"] = "failed"
            event["error"] = repr(error)
            print(f"[{index}/{len(queue)}] FAILED {run_name}: {error!r}")
            if args.fail_fast:
                raise
        finally:
            event["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            append_jsonl(ledger, event)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
