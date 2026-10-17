"""Edit DATA/DEVICE/PROJECT, then VSCode Run Python File: foreground sequential J5."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ("E:/mastercode/data/orange_yolo/data.yaml" if os.name == "nt"
        else "/data/sxq/datasets/orange_yolo/data.yaml")
DEVICE = "1"  # Physical GPU index; one visible GPU is internally logical cuda:0. Or "cpu".
PROJECT = (str(ROOT / "1_results/CITRUS_J5_ALL_300EP_20261015") if os.name == "nt"
           else "/data/sxq/results/J/v5/CITRUS_J5_ALL_300EP_20261015")
SUITE = "all"  # priority/screen = 4 arms; all = 6 arms; see docs/J_V5_EVIDENCE_AND_RUN_20261015.md.
EPOCHS = 300  # First do 1-3 smoke or 50 screen in a DIFFERENT PROJECT.
SEEDS = [42]  # For final comparison use [42, 43, 44], not one seed as proof.
TEACHER = "auto"  # J81 seed42 from THIS PROJECT. Explicit other checkpoints are recorded, not silently selected.
ONLY = ""  # Optional comma-separated J5 names. Automatic KD adds its J81 seed42 dependency if needed.
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
            raise ValueError("DEVICE must be one physical GPU integer string, or cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    # Pass the SAME physical ID. Passing '0' would let this project's select_device
    # overwrite CUDA_VISIBLE_DEVICES and accidentally expose physical GPU0.
    from citrus_j5_batch import run

    run(DATA, DEVICE, PROJECT, SUITE, EPOCHS, SEEDS, ONLY, DRY_RUN or args.dry_run,
        CPU_THREADS, WORKERS, BATCH, TEACHER)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J5 stopped by Ctrl+C. No next model starts.", flush=True)
        raise SystemExit(130)
