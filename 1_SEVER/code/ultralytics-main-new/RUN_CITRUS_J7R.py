"""Edit this block, select the intended Python, click VSCode Run Python File."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ("E:/mastercode/data/orange_yolo_grouped_dedup_20260820/data.yaml" if os.name == "nt"
        else "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml")
DEVICE = "1"  # Physical GPU index; visibility bound BEFORE Torch; CPU checks may use "cpu".
SUITE = "policy"  # First J119/J116; inspect input-policy effect, then "evidence" for J118/J123/J120.
# "priority" explicitly runs both blocks (5 models) without a review stop; "phase" adds optional Q tests.
EPOCHS = 300  # First use 1-3 in a NEW PROJECT to check the server environment.
SEEDS = [42]  # Final method/control: [42,43,44]. Do not select architecture on test.
PROJECT = (str(ROOT / "1_results/CITRUS_J7R_EARLY_300EP_20261020") if os.name == "nt"
           else "/data/sxq/results/J/v7r/CITRUS_J7R_EARLY_300EP_20261020")
ONLY = ""  # e.g. "J120". Partial runs are never overwritten. "phase" is optional; "all" selects 14.
BATCH = 16
WORKERS = 4
CPU_THREADS = 4
PLOTS = True
DRY_RUN = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from citrus_j7r_batch import bind_device, run

    device = bind_device(DEVICE, CPU_THREADS)
    run(DATA, device, PROJECT, SUITE, EPOCHS, SEEDS, ONLY,
        DRY_RUN or args.dry_run, CPU_THREADS, WORKERS, BATCH, PLOTS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J7R stopped by Ctrl+C; no next model starts.", flush=True)
        raise SystemExit(130)
