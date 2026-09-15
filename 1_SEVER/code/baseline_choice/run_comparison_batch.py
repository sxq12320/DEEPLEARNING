"""CITRUS-BL master batch runner for the paper-1 cross-paradigm baseline comparison.

E-series style: fixed protocol, seed-blocked randomized sequential queue on ONE
physical device, ledger.jsonl provenance, implementation snapshot, dry-run,
skip-completed and fail-fast. Training/evaluation are delegated to the existing
worker scripts under scripts/ so every number stays comparable with the
per-family launchers (run_yolo_baselines.py, run_mmdet.py, ...).

Examples:
    python run_comparison_batch.py --suite smoke --dry-run
    python run_comparison_batch.py --suite smoke
    python run_comparison_batch.py --suite screen --seeds 42
    python run_comparison_batch.py --suite formal --seeds 42 --skip-completed
    python run_comparison_batch.py --suite formal --seeds 3407,2026 --only yolo11n_seg --skip-completed
    python run_comparison_batch.py --suite eval --skip-completed
    python run_comparison_batch.py --suite report

No shell, nohup, background queue or concurrent model training is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# =============================================================================
# USER SETTINGS: edit this block only
# =============================================================================

# Windows paths
WINDOWS_SOURCE_DATASET = Path(r"E:\mastercode\data\orange_yolo")
WINDOWS_PREPARED_DATASET = Path(r"E:\mastercode\4_baseline_choice\datasets\citrus_prepared")
WINDOWS_WORKSPACE = Path(r"E:\mastercode\4_baseline_choice\runs")
WINDOWS_MMDET_ROOT = Path(r"E:\mastercode\4_baseline_choice\third_party\mmdetection")

# Linux server paths. The source dataset is read-only.
SERVER_SOURCE_DATASET = Path("/data/sxq/datasets/orange_yolo")
SERVER_PREPARED_DATASET = Path("/data/sxq/results/002_retrain/_prepared/citrus_prepared")
SERVER_WORKSPACE = Path("/data/sxq/results/002_retrain")
SERVER_MMDET_ROOT = Path("/data/sxq/code/mmdetection")

# Formal protocol (kept identical to 对比实验方案.md §4)
FORMAL_EPOCHS = 300
SCREEN_EPOCHS = 50
SMOKE_EPOCHS = 2
YOLO_BATCH = 4
MMDET_BATCH = 16
TORCHVISION_BATCH = 2
RFDETR_BATCH = 2
UNET_BATCH = 8
WORKERS = 4

# =============================================================================
# END USER SETTINGS
# =============================================================================

SUITE_ROOT = Path(__file__).resolve().parent
SCRIPTS = SUITE_ROOT / "scripts"
REGISTRY_PATH = SUITE_ROOT / "configs" / "baselines.yaml"


@dataclass(frozen=True)
class Experiment:
    """One controlled baseline run (one model, one seed)."""

    eid: str
    baseline: str
    family: str  # yolo | mmdet | torchvision | rfdetr | unet
    tier: str  # core | journal | aux | optional
    purpose: str


EXPERIMENTS = (
    Experiment("B0", "yolov8n_seg", "yolo", "core", "previous-generation nano YOLO reference"),
    Experiment("E0", "yolo11n_seg", "yolo", "core", "PRIMARY ablation baseline"),
    Experiment("B1", "yolo26n_seg", "yolo", "core", "current YOLO strong reference"),
    Experiment("B2", "rtmdet_ins_tiny", "mmdet", "core", "non-YOLO lightweight one-stage reference"),
    Experiment("B3", "mask_rcnn_r50_torchvision", "torchvision", "core", "classic two-stage reference"),
    Experiment("B4", "rfdetr_seg_nano", "rfdetr", "core", "current transformer reference"),
    Experiment("B5", "solov2_light", "mmdet", "journal", "box-free position-based reference (journal tier)"),
    Experiment("S0", "unet", "unet", "aux", "semantic-to-instance auxiliary baseline"),
    Experiment("B6", "yolo11s_seg", "yolo", "optional", "same-family accuracy ceiling"),
    Experiment("B1b", "yolo12n_seg", "yolo", "optional", "extra YOLO generation reference"),
    Experiment("B3b", "mask_rcnn_r50", "mmdet", "optional", "MMDetection Mask R-CNN cross-check"),
)

TIER_ORDER = {"core": 0, "journal": 1, "aux": 2, "optional": 3}

SUITE_TIERS = {
    "smoke": ("core", "journal", "aux"),
    "screen": ("core",),
    "formal": ("core", "journal", "aux"),
    "eval": ("core", "journal", "aux", "optional"),
    "all": ("core", "journal", "aux"),
    "report": ("core", "journal", "aux", "optional"),
}

SUITE_EPOCHS = {"smoke": SMOKE_EPOCHS, "screen": SCREEN_EPOCHS, "formal": FORMAL_EPOCHS, "all": FORMAL_EPOCHS}
SUITE_PREFIX = {"smoke": "SMOKE", "screen": "S50", "formal": "E", "all": "E", "eval": "E"}

# Per-family fixed train hyperparameters; cross-framework fairness follows each
# framework's stable official settings (对比实验方案.md §4.2).
FAMILY_BATCH = {
    "yolo": YOLO_BATCH,
    "mmdet": MMDET_BATCH,
    "torchvision": TORCHVISION_BATCH,
    "rfdetr": RFDETR_BATCH,
    "unet": UNET_BATCH,
}
FAMILY_TRAIN_SCRIPT = {
    "yolo": "train_yolo.py",
    "mmdet": "train_mmdet.py",
    "torchvision": "train_torchvision_maskrcnn.py",
    "rfdetr": "train_rfdetr.py",
    "unet": "train_unet.py",
}
FAMILY_EVAL_SCRIPT = {
    "yolo": "eval_yolo.py",
    "mmdet": "eval_mmdet.py",
    "torchvision": "eval_torchvision_maskrcnn.py",
    "rfdetr": "eval_rfdetr.py",
    "unet": "eval_unet.py",
}
FAMILY_OUTPUT_SUBDIR = {
    "yolo": "yolo",
    "mmdet": "mmdet",
    "torchvision": "maskrcnn",
    "rfdetr": "rfdetr",
    "unet": "unet_watershed",
}


def parse_args() -> argparse.Namespace:
    """Parse the reproducible batch protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "screen", "formal", "eval", "report", "all"), required=True)
    parser.add_argument("--seeds", default="42", help="Comma-separated distinct seeds, e.g. 42,3407,2026.")
    parser.add_argument("--only", default="", help="Comma-separated baseline IDs or experiment IDs (B0, E0, ...).")
    parser.add_argument("--device", default="0", help="ONE physical GPU index, or 'cpu' (dry-run only).")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--order-seed", type=int, default=20260909, help="Seed for the queue shuffle.")
    parser.add_argument("--split", choices=("val", "test"), default=None, help="Eval split; default val for smoke/screen, test otherwise.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def platform_paths() -> dict[str, Path]:
    """Select Windows or Linux paths automatically."""
    if os.name == "nt":
        return {
            "source": WINDOWS_SOURCE_DATASET,
            "prepared": WINDOWS_PREPARED_DATASET,
            "workspace": WINDOWS_WORKSPACE,
            "mmdet_root": WINDOWS_MMDET_ROOT,
        }
    return {
        "source": SERVER_SOURCE_DATASET,
        "prepared": SERVER_PREPARED_DATASET,
        "workspace": SERVER_WORKSPACE,
        "mmdet_root": SERVER_MMDET_ROOT,
    }


def run_name(experiment: Experiment, seed: int, prefix: str) -> str:
    """Stable run name; formal seed42 names match the per-family launchers."""
    if experiment.family == experiment.baseline:
        return f"{prefix}_{experiment.family}_seed{seed}"
    return f"{prefix}_{experiment.family}_{experiment.baseline}_seed{seed}"


def output_root(paths: dict[str, Path], experiment: Experiment) -> Path:
    return paths["workspace"] / FAMILY_OUTPUT_SUBDIR[experiment.family]


def find_weights(paths: dict[str, Path], experiment: Experiment, name: str) -> Path | None:
    """Locate the best checkpoint of one finished run, per family convention."""
    run_dir = output_root(paths, experiment) / name
    if experiment.family == "yolo":
        candidate = run_dir / "weights" / "best.pt"
        return candidate if candidate.is_file() else None
    if experiment.family in ("torchvision", "unet"):
        candidate = run_dir / "model_best.pth"
        return candidate if candidate.is_file() else None
    if experiment.family == "rfdetr":
        for checkpoint in ("checkpoint_best_total.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
            candidate = run_dir / checkpoint
            if candidate.is_file():
                return candidate
        candidates = sorted(run_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
    candidates = sorted(run_dir.glob("best_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def train_completed(paths: dict[str, Path], experiment: Experiment, name: str) -> bool:
    return find_weights(paths, experiment, name) is not None


def eval_metrics_path(paths: dict[str, Path], name: str, split: str) -> Path:
    return paths["workspace"] / "evaluation" / f"{name}_{split}" / "metrics.json"


def build_train_command(
    paths: dict[str, Path], experiment: Experiment, name: str, epochs: int, seed: int, args: argparse.Namespace
) -> list[str]:
    """Assemble the worker-script command for one training run."""
    script = SCRIPTS / FAMILY_TRAIN_SCRIPT[experiment.family]
    prepared = paths["prepared"]
    batch = FAMILY_BATCH[experiment.family]
    if experiment.family == "yolo":
        return [
            str(script), "--baseline", experiment.baseline, "--dataset", str(prepared),
            "--name", name, "--output-root", str(output_root(paths, experiment)),
            "--epochs", str(epochs), "--imgsz", "640", "--batch", str(batch),
            "--device", str(args.device), "--workers", str(args.workers),
            "--optimizer", "AdamW", "--lr0", "0.001", "--weight-decay", "0.0005",
            "--patience", "100", "--seed", str(seed), "--no-amp",
        ]
    if experiment.family == "mmdet":
        return [
            str(script), "--baseline", experiment.baseline, "--dataset", str(prepared),
            "--name", name, "--mmdet-root", str(paths["mmdet_root"]),
            "--output-root", str(output_root(paths, experiment)),
            "--epochs", str(epochs), "--batch", str(batch), "--workers", str(args.workers),
            "--seed", str(seed), "--val-interval", "5",
        ]
    if experiment.family == "torchvision":
        return [
            str(script), "--dataset", str(prepared), "--name", name,
            "--output-root", str(output_root(paths, experiment)),
            "--epochs", str(epochs), "--batch", str(batch), "--workers", str(args.workers),
            "--lr", "0.005", "--momentum", "0.9", "--weight-decay", "0.0005",
            "--imgsz", "640", "--detections-per-image", "50",
            "--val-interval", "5", "--seed", str(seed), "--device", "auto",
            "--initialization", "coco",
        ]
    if experiment.family == "rfdetr":
        return [
            str(script), "--baseline", experiment.baseline, "--dataset", str(prepared),
            "--name", name, "--output-root", str(output_root(paths, experiment)),
            "--epochs", str(epochs), "--batch", str(batch), "--grad-accum-steps", "4",
            "--workers", str(args.workers), "--device", "cuda", "--seed", str(seed),
            "--lr", "1e-4", "--lr-encoder", "1.5e-4", "--weight-decay", "1e-4",
        ]
    return [
        str(script), "--dataset", str(prepared), "--name", name,
        "--output-root", str(output_root(paths, experiment)),
        "--encoder", "resnet18", "--encoder-weights", "imagenet",
        "--epochs", str(epochs), "--batch", str(batch), "--workers", str(args.workers),
        "--lr", "0.0003", "--weight-decay", "0.0001", "--imgsz", "640",
        "--val-interval", "5", "--seed", str(seed), "--device", "auto",
    ]


def build_eval_command(
    paths: dict[str, Path], experiment: Experiment, name: str, weights: Path, split: str, args: argparse.Namespace
) -> list[str]:
    """Assemble the worker-script command for one COCO mask evaluation."""
    script = SCRIPTS / FAMILY_EVAL_SCRIPT[experiment.family]
    output = paths["workspace"] / "evaluation" / f"{name}_{split}"
    prepared = paths["prepared"]
    if experiment.family == "yolo":
        return [
            str(script), "--weights", str(weights), "--dataset", str(prepared),
            "--split", split, "--output", str(output), "--imgsz", "640",
            "--device", str(args.device), "--batch", "1", "--workers", str(args.workers),
        ]
    if experiment.family == "mmdet":
        return [
            str(script), "--weights", str(weights), "--dataset", str(prepared),
            "--split", split, "--output", str(output), "--device", f"cuda:{args.device}",
            "--score-threshold", "0.001",
        ]
    if experiment.family == "torchvision":
        return [
            str(script), "--weights", str(weights), "--dataset", str(prepared),
            "--split", split, "--output", str(output), "--workers", str(args.workers),
            "--device", "auto", "--score-threshold", "0.001",
        ]
    if experiment.family == "rfdetr":
        return [
            str(script), "--baseline", experiment.baseline, "--weights", str(weights),
            "--dataset", str(prepared), "--split", split, "--output", str(output),
            "--device", "cuda", "--batch", "1", "--score-threshold", "0.001",
        ]
    return [
        str(script), "--weights", str(weights), "--dataset", str(prepared),
        "--split", split, "--output", str(output), "--batch", "4",
        "--workers", str(args.workers), "--device", "auto",
    ]


def run_worker(command: list[str], experiment: Experiment, device: str) -> None:
    """Run one worker sequentially; non-YOLO frameworks get the visibility mask."""
    env = dict(os.environ)
    if experiment.family != "yolo" and device != "cpu":
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    printable = subprocess.list2cmdline([sys.executable, *command])
    print(f"    $ {printable}", flush=True)
    subprocess.run([sys.executable, *command], cwd=SUITE_ROOT, env=env, check=True)


def append_event(path: Path, event: dict) -> None:
    """Append a durable experiment event (E-series ledger)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def git_snapshot() -> dict[str, str]:
    """Record source identity without requiring a clean worktree."""
    def run(*argv: str) -> str:
        try:
            return subprocess.check_output(argv, cwd=SUITE_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": str(bool(run("git", "status", "--porcelain")))}


def implementation_snapshot(protocol_dir: Path) -> str:
    """Hash this runner, the registry and every worker; refuse mixed implementations."""
    files = [Path(__file__).resolve(), REGISTRY_PATH, *sorted(SCRIPTS.glob("*.py"))]
    hashes = {str(path.relative_to(SUITE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    content = json.dumps(hashes, sort_keys=True, indent=2)
    snapshot = protocol_dir / "implementation_sha256.json"
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != content:
        raise FileExistsError(
            f"Implementation changed within {protocol_dir.parent}; move outputs to a new workspace "
            "or revert the code, so differently-computed numbers never share one table."
        )
    protocol_dir.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def prepared_dataset_exists(path: Path) -> bool:
    """Check whether YOLO and COCO converted splits exist."""
    return all(
        (path / "yolo" / "images" / split).is_dir()
        and (path / "yolo" / "labels" / split).is_dir()
        and (path / "coco" / "annotations" / f"instances_{split}.json").is_file()
        for split in ("train", "val", "test")
    )


def ensure_prepared(paths: dict[str, Path], dry_run: bool) -> None:
    """Convert the source polygons once if the prepared dataset is absent."""
    if prepared_dataset_exists(paths["prepared"]):
        return
    if dry_run:
        print(f"PREPARE NEEDED: {paths['prepared']} does not exist yet")
        return
    if not paths["source"].is_dir():
        raise FileNotFoundError(f"Source YOLO dataset not found: {paths['source']}")
    run_worker(
        [
            str(SCRIPTS / "prepare_dataset.py"), "--source", str(paths["source"]),
            "--output", str(paths["prepared"]), "--class-name", "orange_immature", "--mode", "auto",
        ],
        Experiment("PP", "prepare", "yolo", "aux", "dataset preparation"),
        device="cpu",
    )


def select_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Resolve suite tiers and the --only filter into a deterministic model list."""
    tiers = SUITE_TIERS[args.suite]
    experiments = [exp for exp in EXPERIMENTS if exp.tier in tiers]
    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        experiments = [exp for exp in experiments if exp.baseline in requested or exp.eid in requested]
        missing = requested - {exp.baseline for exp in experiments} - {exp.eid for exp in experiments}
        if missing:
            raise ValueError(f"Unknown or out-of-suite experiments: {sorted(missing)}")
    return experiments


def main() -> None:
    """Run the selected suite sequentially on one device."""
    args = parse_args()
    paths = platform_paths()
    paths["workspace"].mkdir(parents=True, exist_ok=True)
    protocol_dir = paths["workspace"] / "_protocol"
    ledger = protocol_dir / "ledger.jsonl"
    digest = implementation_snapshot(protocol_dir)
    repo = git_snapshot()

    if args.suite == "report":
        command = [
            str(SCRIPTS / "report_comparison.py"), "--evaluation", str(paths["workspace"] / "evaluation"),
            "--registry", str(REGISTRY_PATH),
        ]
        run_worker(command, Experiment("RP", "report", "yolo", "aux", "aggregate report"), device="cpu")
        return

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use a nonempty list of distinct seeds")
    if args.suite in ("smoke", "screen") and len(seeds) > 1:
        raise ValueError("smoke/screen are single-seed suites; formal comparisons use multiple seeds")
    split = args.split or ("val" if args.suite in ("smoke", "screen") else "test")
    epochs = SUITE_EPOCHS.get(args.suite)
    prefix = SUITE_PREFIX.get(args.suite, "E")
    experiments = select_experiments(args)
    if not experiments:
        raise ValueError("Empty experiment selection")

    # Seed-blocked randomized order (E-series): each seed sees the same shuffled list.
    queue: list[tuple[Experiment, int]] = []
    for seed in seeds:
        ordered = list(experiments)
        random.Random(args.order_seed + seed).shuffle(ordered)
        queue.extend((exp, seed) for exp in ordered)

    print(f"Python: {sys.executable}\nWorkspace: {paths['workspace']}\nPrepared: {paths['prepared']}", flush=True)
    print(f"Suite={args.suite} epochs={epochs} split={split} seeds={seeds} snapshot={digest}", flush=True)
    print(f"Queue ({len(queue)} runs):", flush=True)
    for exp, seed in queue:
        print(f"  {exp.eid:<4} {run_name(exp, seed, prefix):<44} [{exp.family}/{exp.tier}] {exp.purpose}", flush=True)
    append_event(
        ledger,
        {
            "status": "queue", "suite": args.suite, "queue": [(exp.eid, exp.baseline, seed) for exp, seed in queue],
            "epochs": epochs, "split": split, "order_seed": args.order_seed,
            "snapshot": digest, "git": repo, "command": sys.argv, "time": time.time(),
        },
    )
    if args.dry_run:
        print("DRY RUN ONLY: no training.", flush=True)
        ensure_prepared(paths, dry_run=True)
        return
    if args.device == "cpu":
        raise ValueError("--device cpu is only meaningful for --dry-run; training requires one physical GPU")

    ensure_prepared(paths, dry_run=False)
    failures: list[str] = []
    for index, (exp, seed) in enumerate(queue, 1):
        name = run_name(exp, seed, prefix)
        event = {
            "eid": exp.eid, "experiment": asdict(exp), "seed": seed, "name": name,
            "suite": args.suite, "epochs": epochs, "split": split, "snapshot": digest,
        }
        weights = find_weights(paths, exp, name)
        need_train = args.suite in ("smoke", "screen", "formal", "all")
        need_eval = args.suite in ("eval", "all")
        if need_train and train_completed(paths, exp, name):
            if not args.skip_completed:
                raise FileExistsError(f"Run already complete (use --skip-completed to reuse): {name}")
            print(f"[{index}/{len(queue)}] SKIP trained: {name}", flush=True)
            append_event(ledger, {**event, "status": "skip_trained", "time": time.time()})
            need_train = False
        if need_eval and eval_metrics_path(paths, name, split).is_file():
            if not args.skip_completed:
                raise FileExistsError(f"Evaluation already complete (use --skip-completed): {name}_{split}")
            print(f"[{index}/{len(queue)}] SKIP evaluated: {name}_{split}", flush=True)
            append_event(ledger, {**event, "status": "skip_evaluated", "time": time.time()})
            need_eval = False
        if not need_train and not need_eval:
            continue
        append_event(ledger, {**event, "status": "started", "time": time.time()})
        try:
            if need_train:
                print(f"[{index}/{len(queue)}] TRAIN {name} ({epochs} epochs)", flush=True)
                command = build_train_command(paths, exp, name, epochs, seed, args)
                run_worker(command, exp, args.device)
                weights = find_weights(paths, exp, name)
                if weights is None:
                    raise FileNotFoundError(f"Training finished but no checkpoint was found for {name}")
                append_event(ledger, {**event, "status": "trained", "weights": str(weights), "time": time.time()})
            if need_eval:
                if weights is None:
                    raise FileNotFoundError(
                        f"No checkpoint for {name}; train it first (--suite formal) or check the output root"
                    )
                print(f"[{index}/{len(queue)}] EVAL {name} split={split}", flush=True)
                command = build_eval_command(paths, exp, name, weights, split, args)
                run_worker(command, exp, args.device)
                if not eval_metrics_path(paths, name, split).is_file():
                    raise FileNotFoundError(f"Evaluation finished but metrics.json is missing for {name}")
                append_event(ledger, {**event, "status": "completed", "time": time.time()})
        except KeyboardInterrupt:
            append_event(ledger, {**event, "status": "interrupted", "time": time.time()})
            raise  # never continue the queue after Ctrl+C
        except Exception as error:
            append_event(ledger, {**event, "status": "failed", "error": repr(error), "time": time.time()})
            failures.append(name)
            print(f"[{index}/{len(queue)}] FAILED {name}: {error!r}", flush=True)
            if args.fail_fast:
                raise
    if failures:
        raise RuntimeError(f"Failed experiments: {failures}; inspect {ledger}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
