"""Edit DATA/DEVICE/PROJECT, then click VSCode Run Python File for sequential foreground J6 training."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ("E:/mastercode/data/orange_yolo/data.yaml" if os.name == "nt"
        else "/data/sxq/datasets/orange_yolo/data.yaml")
DEVICE = "1"  # PHYSICAL GPU number, or "cpu". It is bound before importing torch.
PROJECT = (str(ROOT / "1_results/CITRUS_J6_ALL_300EP_20261017_R2") if os.name == "nt"
           else "/data/sxq/results/J/v6/CITRUS_J6_ALL_300EP_20261017_R2")
SUITE = "all"  # all: ten arms; screen/priority: J90, J91, J92, J94.
EPOCHS = 300
SEEDS = [42]  # Use [42, 43, 44] for the final paired comparison.
ONLY = ""  # Example: "J94_full,J97_tiny_balanced"; no implicit extra jobs.
RESUME = ""  # Explicit names, e.g. "J94_full"; uses that run's weights/last.pt, one seed only.
DRY_RUN = False
PLOTS = True  # Keep PR/confusion plots by default; --no-plots is an explicit engineering-only opt-out.
CPU_THREADS = 4
WORKERS = 4
BATCH = 16


def configure_device(device, cpu_threads):
    """Bind one physical GPU before any torch import; do not acquire an occupancy guard."""
    device = str(device).strip()
    if cpu_threads < 1:
        raise ValueError("CPU_THREADS must be positive")
    if device != "cpu":
        if not device.isascii() or not device.isdigit():
            raise ValueError("DEVICE must be a single physical GPU integer string, or cpu")
        if "torch" in sys.modules:
            raise RuntimeError("Configure DEVICE before importing torch; run this file in a fresh Python process")
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    return device


def parse_args(argv=None):
    """CLI options mirror the editable constants without requiring a shell background job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--suite", default=SUITE,
                        choices=("all", "screen", "priority", "neighbors", "selection", "detail", "loss", "gain"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--only", default=ONLY)
    parser.add_argument("--resume", default=RESUME, metavar="MODEL[,MODEL]")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--no-plots", action="store_false", dest="plots", default=PLOTS,
                        help="Explicitly disable saved plots; does not change matching/AP computations")
    parser.add_argument("--smoke", action="store_true", help="One epoch in a NEW _SMOKE_1EP project, never 300EP")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS)
    args = parser.parse_args(argv)
    try:
        args.seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if args.smoke:
        if args.resume:
            parser.error("--smoke cannot resume a formal run; choose a separate explicit project")
        args.project = str(Path(args.project).with_name(Path(args.project).name + "_SMOKE_1EP"))
        args.epochs = 1
    return args


def main(argv=None):
    args = parse_args(argv)
    device = configure_device(args.device, args.cpu_threads)
    # Keep this import after device binding. Passing physical DEVICE through is intentional:
    # this checkout's select_device would otherwise re-expose GPU0 when passed a logical "0".
    from citrus_j6_batch import run

    run(data=args.data, device=device, project=args.project, suite=args.suite, epochs=args.epochs,
        seeds=args.seeds, only=args.only, dry_run=args.dry_run, cpu_threads=args.cpu_threads,
        workers=args.workers, batch=args.batch, resume=args.resume, plots=args.plots)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J6 stopped by Ctrl+C. No next model starts. Resume is explicit, never automatic.", flush=True)
        raise SystemExit(130)
