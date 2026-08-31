"""Unified batch runner for the CitrusT finalists.

The dataset YAML is always supplied explicitly by the user:

    python 20260829_citrus_t_batch.py --data /data/orange/data.yaml --suite smoke --epochs 3
    python 20260829_citrus_t_batch.py --data /data/orange/data.yaml --suite all --epochs 300
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
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "T_series"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Experiment:
    """One selected historical or current architecture."""

    name: str
    yaml: str
    source: str
    losses: dict[str, float]
    purpose: str


EXPERIMENTS = (
    Experiment("T00_yolo11n_reference", "00_t00_yolo11n_reference.yaml", "A/official", {}, "uniform baseline"),
    Experiment("T01_f14_sppf_lska", "01_t01_f14_sppf_lska.yaml", "F14", {}, "old-data best single factor"),
    Experiment("T02_g10_hybrid", "02_t02_g10_hybrid.yaml", "G10", {}, "old-data overall winner"),
    Experiment("T03_n02_moce_lska", "03_t03_n02_moce_lska.yaml", "N02", {}, "old-data N-series winner"),
    Experiment(
        "T04_l06_lska_topology",
        "04_t04_l06_lska_topology.yaml",
        "L06",
        {"citrus_boundary": 0.50, "citrus_query": 0.10},
        "L-series context/topology representative",
    ),
    Experiment("T05_s04_lite_head", "05_t05_s04_lite_head.yaml", "S04", {}, "clean-data efficient head"),
    Experiment(
        "T06_b06_context_topology_lite",
        "06_t06_b06_context_topology_lite.yaml",
        "B06",
        {"citrus_boundary": 0.25, "citrus_query": 0.05},
        "completed clean-data B winner",
    ),
    Experiment(
        "T07_c03_dualproto_core",
        "07_t07_c03_dualproto_core.yaml",
        "C03",
        {"citrus_topology": 0.10},
        "task-specific dual-prototype core",
    ),
    Experiment(
        "T08_d06_shape_semantic_full",
        "08_t08_d06_shape_semantic_full.yaml",
        "D06",
        {"citrus_boundary": 0.15, "citrus_query": 0.03},
        "new backbone accuracy candidate",
    ),
    Experiment(
        "T09_d07_deploy_lite",
        "09_t09_d07_deploy_lite.yaml",
        "D07",
        {"citrus_boundary": 0.15, "citrus_query": 0.03},
        "new backbone deployment candidate",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse the locked T-series protocol and mandatory dataset confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Exact server data.yaml to audit and train.")
    parser.add_argument("--suite", choices=("smoke", "priority", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--only", default="", help="Comma-separated T experiment names.")
    parser.add_argument("--verify-data-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def resolve_dataset_root(source: Path, config: dict[str, Any]) -> tuple[Path, str]:
    """Resolve the effective root while safely handling stale Windows paths on a server."""
    configured = config.get("path")
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        if candidate.is_dir():
            return candidate.resolve(), "configured data.yaml path"
    return source.parent.resolve(), "data.yaml parent (configured path missing or stale)"


def resolve_entry(entry: str, root: Path, list_parent: Path | None = None) -> Path:
    """Resolve one split or list-file entry using YOLO-style root-relative paths."""
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = root / path
    if candidate.exists() or list_parent is None:
        return candidate.resolve()
    return (list_parent / path).resolve()


def split_images(value: Any, root: Path) -> list[Path]:
    """Expand a directory, image, list file, or list of these into image paths."""
    if value is None:
        return []
    entries = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for entry in entries:
        path = resolve_entry(str(entry), root)
        if path.is_dir():
            images.extend(item.resolve() for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    image = resolve_entry(line.strip(), root, path.parent)
                    if image.suffix.lower() in IMAGE_SUFFIXES:
                        images.append(image)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(f"Dataset split entry does not resolve: {entry!r} -> {path}")
    return sorted(set(images), key=lambda item: item.as_posix().lower())


def label_path(image: Path, root: Path) -> Path:
    """Map a conventional YOLO images path to its labels path without modifying data."""
    try:
        relative = image.relative_to(root)
    except ValueError:
        relative = image
    parts = list(relative.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return (root / Path(*parts)).with_suffix(".txt")
    return (root / "labels" / relative).with_suffix(".txt")


def audit_dataset(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the user-supplied dataset YAML and summarize its splits."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    root, root_reason = resolve_dataset_root(source, config)
    split_paths = {split: split_images(config.get(split), root) for split in ("train", "val", "test")}
    if not split_paths["train"] or not split_paths["val"]:
        raise ValueError("Both train and val splits must contain images.")

    seen_paths: dict[Path, str] = {}
    exact_path_overlap = []
    split_stats = {}
    for split, images in split_paths.items():
        instances = 0
        labels_present = 0
        missing_labels = 0
        for image in images:
            if image in seen_paths and seen_paths[image] != split:
                exact_path_overlap.append({"image": str(image), "splits": [seen_paths[image], split]})
            seen_paths[image] = split
            label = label_path(image, root)
            label_bytes = b""
            if label.is_file():
                labels_present += 1
                label_bytes = label.read_bytes()
                instances += sum(bool(line.strip()) for line in label_bytes.decode("utf-8", errors="ignore").splitlines())
            else:
                missing_labels += 1
        split_stats[split] = {
            "images": len(images),
            "labels_present": labels_present,
            "missing_labels": missing_labels,
            "instances": instances,
            "entry": config.get(split),
        }
    report = {
        "source_yaml": str(source.resolve()),
        "resolved_root": str(root),
        "root_resolution": root_reason,
        "names": config.get("names"),
        "splits": split_stats,
        "exact_path_overlap": exact_path_overlap,
    }
    runtime = dict(config)
    runtime["path"] = str(root)
    return report, runtime


def print_audit(report: dict[str, Any]) -> None:
    """Print the resolved user-supplied dataset path before the queue."""
    print("\n" + "=" * 72)
    print("DATASET PATH CHECK")
    print(f"YAML:        {report['source_yaml']}")
    print(f"Root:        {report['resolved_root']} ({report['root_resolution']})")
    print(f"Classes:     {report['names']}")
    for split, stats in report["splits"].items():
        print(
            f"{split:<5}: images={stats['images']:<5} instances={stats['instances']:<6} "
            f"labels={stats['labels_present']:<5} missing-label-files={stats['missing_labels']}"
        )
    print(f"Path overlap: {len(report['exact_path_overlap'])}")
    print("=" * 72 + "\n")


def selected_experiments(suite: str) -> list[Experiment]:
    """Return all models for smoke/all or the smallest high-value audit subset."""
    if suite == "priority":
        return [EXPERIMENTS[index] for index in (0, 1, 2, 5, 6, 8, 9)]
    return list(EXPERIMENTS)


def set_seed(seed: int) -> None:
    """Lock all random sources used by the training stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def repository_state() -> dict[str, str]:
    """Record exact source state for every run."""

    def run(*command: str) -> str:
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": str(bool(run("git", "status", "--porcelain")))}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one durable experiment event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed(ledger: Path, run_name: str, run_dir: Path, epochs: int) -> bool:
    """Skip only runs completed under the same requested epoch count."""
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
    """Audit one dataset and sequentially train selected T models."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    report, runtime_config = audit_dataset(data)
    print_audit(report)
    if args.verify_data_only:
        return

    pretrained = args.pretrained.expanduser().resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.imgsz != 640:
        raise ValueError("Formal T-series comparisons lock imgsz=640.")
    if args.suite == "smoke" and args.epochs > 3:
        raise ValueError("Smoke runs are limited to 1-3 epochs.")
    default_project = ROOT / "1_results" / "T_series" / f"CITRUS_T_{args.suite.upper()}_{args.epochs}EP"
    project = (args.project or default_project).expanduser().resolve()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    experiments = selected_experiments(args.suite)
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        available = {experiment.name for experiment in experiments}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown or out-of-suite experiments: {sorted(unknown)}")
        experiments = [experiment for experiment in experiments if experiment.name in requested]
    queue = [(experiment, seed) for experiment in experiments for seed in seeds]
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    print(f"Project: {project}\nQueue ({len(queue)} sequential runs):")
    for experiment, seed in queue:
        print(f"  {experiment.name:<34} seed={seed:<4} source={experiment.source:<10} losses={experiment.losses}")
    if args.dry_run:
        return

    from ultralytics import YOLO

    project.mkdir(parents=True, exist_ok=True)
    protocol_dir = project / "_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_data = protocol_dir / "runtime_data.yaml"
    runtime_data.write_text(yaml.safe_dump(runtime_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ledger = project / "experiment_ledger.jsonl"
    repo = repository_state()

    for index, (experiment, seed) in enumerate(queue, 1):
        yaml_path = YAML_DIR / experiment.yaml
        if not yaml_path.is_file():
            raise FileNotFoundError(f"T model YAML not found: {yaml_path}")
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
            "dataset_yaml": report["source_yaml"],
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
                set_seed(seed)
                model = YOLO(str(yaml_path), task="segment").load(str(pretrained))
                model.train(**protocol)
            append_jsonl(ledger, {**event, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "completed"})
        except Exception as error:
            append_jsonl(
                ledger,
                {**event, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "failed", "error": repr(error)},
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
