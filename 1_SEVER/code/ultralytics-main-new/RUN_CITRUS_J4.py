"""Edit DATA/DEVICE/PROJECT, then click VSCode Run Python File. Foreground, serial."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ("E:/mastercode/data/orange_yolo_grouped_dedup_20260820/data.yaml" if os.name == "nt"
        else "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml")
DEVICE = "1"  # ONE physical GPU, or "cpu"; mapped to logical 0 before torch import.
SUITE = "priority"  # priority: J60/J61/J64/J65; screen/all: eight arms.
EPOCHS = 50  # 1-3 smoke, 50 screen, then selected models 300 with a NEW PROJECT.
SEEDS = [42]
PROJECT = (str(ROOT / "1_results/CITRUS_J4_PRIORITY_50EP_20261009") if os.name == "nt"
           else "/data/sxq/results/J/v4/CITRUS_J4_PRIORITY_50EP_20261009")
ONLY = ""  # Comma-separated model IDs; use this to skip an unfinished arm safely.
DRY_RUN = False
CPU_THREADS = 4
WORKERS = 4
BATCH = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if DEVICE != "cpu":
        if not DEVICE.isascii() or not DEVICE.isdigit():
            raise ValueError("DEVICE must be one physical GPU index, or cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    from citrus_j4_batch import run

    run(DATA, "cpu" if DEVICE == "cpu" else "0", PROJECT, SUITE, EPOCHS, SEEDS, ONLY,
        DRY_RUN or args.dry_run, CPU_THREADS, WORKERS, BATCH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J4 stopped by Ctrl+C. No next model starts.", flush=True)
        raise SystemExit(130)
