"""Edit the block below, then click VSCode Run Python File: foreground, sequential, no nohup."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ("E:/mastercode/data/orange_yolo_grouped_dedup_20260820/data.yaml" if os.name == "nt"
        else "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml")
DEVICE = "1"  # ONE physical GPU index, or "cpu". Bound before Torch import; logical device becomes 0.
SUITE = "priority"  # priority: J100/J102/J103/J104. all: J100...J109 (untrained research controls).
EPOCHS = 300  # First use 1-3 for a smoke test, with a NEW PROJECT for each epoch budget.
SEEDS = [42]  # After selecting a final method: [42, 43, 44], same protocol/data for the controls.
PROJECT = (str(ROOT / "1_results/CITRUS_J7_PRIORITY_300EP_20261018") if os.name == "nt"
           else "/data/sxq/results/J/v7/CITRUS_J7_PRIORITY_300EP_20261018")
ONLY = ""  # e.g. "J102,J103,J104" to omit a control already completed in another experiment.
DRY_RUN = False
CPU_THREADS = 4
WORKERS = 4
BATCH = 16
PLOTS = True  # Use False only explicitly, e.g. in a known unsupported local plotting environment.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build/profile; do not load data or train")
    args = parser.parse_args()
    from citrus_j7_batch import bind_device, run

    logical_device = bind_device(DEVICE, CPU_THREADS)
    run(DATA, logical_device, PROJECT, SUITE, EPOCHS, SEEDS, ONLY,
        DRY_RUN or args.dry_run, CPU_THREADS, WORKERS, BATCH, PLOTS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J7 stopped by Ctrl+C. The next model will NOT start.", flush=True)
        raise SystemExit(130)
