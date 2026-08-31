"""Batch runner for the latency-aware CitrusSwift architecture and loss studies.

Run the 50-epoch architecture screen first. Promote only statistically meaningful
variants to 300 epochs and three seeds; the script supports both stages without
changing the controlled protocol.

Examples:
    python 20260824_citrus_swift_batch.py --data /data/orange/data.yaml --suite architectures
    python 20260824_citrus_swift_batch.py --data /data/orange/data.yaml --suite final \
        --epochs 300 --seeds 42,43,44
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
YAML_DIR = ROOT / "0_orange_yaml" / "S_series"
DEFAULT_PROJECT = ROOT / "1_results" / "S_series" / "grouped_clean_screen50"

AUX_CORE = {"citrus_boundary": 0.25, "citrus_query": 0.05, "citrus_contrast": 0.10}
AUX_FULL = {
    "citrus_boundary": 0.25,
    "citrus_concavity": 0.10,
    "citrus_query": 0.05,
    "citrus_contrast": 0.10,
    "citrus_exclusive": 0.05,
    "nwd_ratio": 0.25,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    yaml: str
    losses: dict[str, float]
    purpose: str


ARCHITECTURE_EXPERIMENTS = (
    Experiment("S00_reference", "00_reference.yaml", {}, "unchanged YOLO11n-seg reference"),
    Experiment("S01_repcontext", "01_repcontext_backbone.yaml", {}, "fused 7x7 P5 context"),
    Experiment("S02_lska", "02_lska_backbone.yaml", {}, "historically strongest P5 LSKA context"),
    Experiment("S03_train_aux", "03_train_aux_head.yaml", AUX_CORE, "inference-free task supervision"),
    Experiment("S04_lite_head", "04_lite_head.yaml", {}, "one-block prediction heads"),
    Experiment("S05_fpn_only", "05_fpn_only_neck.yaml", {}, "top-down-only maximum-speed neck"),
    Experiment("S06_asym_pan", "06_asym_pan_neck.yaml", {}, "one-step bottom-up asymmetric PAN"),
    Experiment("S07_lska_asym", "07_lska_asym_pan.yaml", {}, "backbone plus neck structure"),
    Experiment("S08_swift_full", "08_citrus_swift_full.yaml", AUX_FULL, "complete latency-aware candidate"),
    Experiment(
        "S09_dense_control",
        "09_dense_topology_control.yaml",
        {"citrus_boundary": 0.25, "citrus_query": 0.05},
        "previous dense P2 control",
    ),
)

LOSS_EXPERIMENTS = (
    Experiment("L00_standard", "03_train_aux_head.yaml", {}, "standard loss"),
    Experiment("L01_boundary", "03_train_aux_head.yaml", {"citrus_boundary": 0.25}, "visible boundary"),
    Experiment("L02_query", "03_train_aux_head.yaml", {"citrus_query": 0.05}, "tiny-center focal query"),
    Experiment("L03_contrast", "03_train_aux_head.yaml", {"citrus_contrast": 0.10}, "fruit/context ring"),
    Experiment(
        "L04_boundary_query",
        "03_train_aux_head.yaml",
        {"citrus_boundary": 0.25, "citrus_query": 0.05},
        "boundary plus tiny-center",
    ),
    Experiment(
        "L05_boundary_contrast",
        "03_train_aux_head.yaml",
        {"citrus_boundary": 0.25, "citrus_contrast": 0.10},
        "boundary plus camouflage contrast",
    ),
    Experiment("L06_aux_core", "03_train_aux_head.yaml", AUX_CORE, "all inference-free auxiliary tasks"),
    Experiment(
        "L07_concavity",
        "03_train_aux_head.yaml",
        {**AUX_CORE, "citrus_concavity": 0.10},
        "occlusion-notch focus",
    ),
    Experiment(
        "L08_nwd",
        "03_train_aux_head.yaml",
        {**AUX_CORE, "nwd_ratio": 0.25},
        "small-object NWD/CIoU blend",
    ),
    Experiment("L09_full", "03_train_aux_head.yaml", AUX_FULL, "all task losses; interaction control"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch train the controlled CitrusSwift studies.")
    parser.add_argument("--data", required=True, help="Path to grouped-dedup data.yaml on the server.")
    parser.add_argument("--suite", choices=("architectures", "losses", "all", "final"), default="architectures")
    parser.add_argument("--epochs", type=int, default=50, help="Use 50 for screening; 300 only for promoted models.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--pretrained", default=str(ROOT / "yolo11n-seg.pt"))
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,43,44.")
    parser.add_argument("--only", default="", help="Comma-separated experiment names.")
    parser.add_argument("--dry-run", action="store_true")
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

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": str(bool(run("git", "status", "--porcelain")))}


def select_experiments(suite: str) -> list[Experiment]:
    if suite == "architectures":
        return list(ARCHITECTURE_EXPERIMENTS)
    if suite == "losses":
        return list(LOSS_EXPERIMENTS)
    if suite == "final":
        return [ARCHITECTURE_EXPERIMENTS[0], ARCHITECTURE_EXPERIMENTS[8]]
    return [*ARCHITECTURE_EXPERIMENTS, *LOSS_EXPERIMENTS]


def is_complete(run_dir: Path, epochs: int) -> bool:
    csv_path, best_path = run_dir / "results.csv", run_dir / "weights" / "best.pt"
    if not csv_path.is_file() or not best_path.is_file():
        return False
    try:
        with csv_path.open(encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0) >= epochs
    except OSError:
        return False


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_runtime_data_yaml(source: Path, project: Path) -> Path:
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    config["path"] = str(source.parent.resolve())
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
        raise ValueError("The controlled study locks imgsz=640.")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")

    experiments = select_experiments(args.suite)
    if args.only:
        selected = {value.strip() for value in args.only.split(",") if value.strip()}
        experiments = [experiment for experiment in experiments if experiment.name in selected]
        missing = selected - {experiment.name for experiment in experiments}
        if missing:
            raise ValueError(f"Unknown or out-of-suite experiments: {sorted(missing)}")

    jobs = [(experiment, seed) for experiment in experiments for seed in seeds]
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    print(f"Dataset: {data}\nProject: {args.project.resolve()}\nQueue ({len(jobs)} runs):")
    for experiment, seed in jobs:
        print(f"  {experiment.name:<20} seed={seed:<4} {experiment.yaml:<32} {experiment.losses}")
    if args.dry_run:
        return

    from ultralytics import YOLO

    args.project.mkdir(parents=True, exist_ok=True)
    runtime_data = prepare_runtime_data_yaml(data, args.project)
    ledger = args.project / "experiment_ledger.jsonl"
    repository = git_state()

    for index, (experiment, seed) in enumerate(jobs, start=1):
        yaml_path = YAML_DIR / experiment.yaml
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Model YAML not found: {yaml_path}")
        run_name = experiment.name if len(seeds) == 1 else f"{experiment.name}_seed{seed}"
        run_dir = args.project / run_name
        if is_complete(run_dir, args.epochs):
            print(f"[{index}/{len(jobs)}] SKIP complete: {run_name}")
            continue

        common = {
            "data": str(runtime_data),
            "project": str(args.project.resolve()),
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
            "amp": False,
            "seed": seed,
            "deterministic": True,
            "cache": False,
            "exist_ok": False,
        }
        set_seed(seed)
        record = {
            "experiment": asdict(experiment),
            "seed": seed,
            "yaml": str(yaml_path.resolve()),
            "pretrained": str(pretrained),
            "source_data_yaml": str(data),
            "protocol": common,
            "repository": repository,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "running",
        }
        append_ledger(ledger, record)
        model = None
        try:
            last_path = run_dir / "weights" / "last.pt"
            if run_dir.exists() and last_path.is_file():
                print(f"[{index}/{len(jobs)}] RESUME {run_name}: {last_path}")
                model = YOLO(str(last_path))
                model.train(resume=True)
            elif run_dir.exists():
                raise RuntimeError(f"Incomplete run has no weights/last.pt: {run_dir}")
            else:
                print(f"[{index}/{len(jobs)}] START {run_name}")
                model = YOLO(str(yaml_path)).load(str(pretrained))
                model.train(**common, **experiment.losses)
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = repr(error)
            print(f"[{index}/{len(jobs)}] FAILED {run_name}: {error!r}")
            if args.fail_fast:
                raise
        finally:
            record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            append_ledger(ledger, record)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
