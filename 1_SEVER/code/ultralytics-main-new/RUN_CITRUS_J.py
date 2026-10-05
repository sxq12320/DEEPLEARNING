"""Edit DATA/DEVICE/SUITE; click VS Code Run Python File. Foreground only."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"  # Physical GPU. This script exposes only that GPU before importing torch.
SUITE = "priority"  # priority=7 arms; all=20; structure/losses/optimization also available.
EPOCHS = 50  # Screen first; then 300 with a NEW project path.
SEEDS = [42]  # Final shortlisted method + control: [42, 43, 44].
PROJECT = "/data/sxq/results/J/CITRUS_J_{}_{}EP_NATIVEVAL".format(SUITE.upper(), EPOCHS)
ONLY = ""  # E.g. "J11_synthesis,J17_budget"; overrides SUITE.
DRY_RUN = False
SOURCE_BALANCED = True  # Same original/coarse/fine uniform training views for ALL arms.
COMMON_VALIDATION = False  # J series uses the I-series EV6Validator, NOT the baseline COCO evaluator.
# Compatibility flag only: True is rejected. Baseline validation stays in its own runner.
DIAGNOSE_PR = False  # Optional extra pass for raw-match CSV; ordinary native PR/CM plots are already enabled.
CPU_THREADS = 4


def main():
    if DEVICE != "cpu":
        if not DEVICE.isdigit():
            raise ValueError("DEVICE must name one physical GPU or 'cpu'; no distributed launcher")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    from citrus_j_batch import run

    run(
        DATA,
        "cpu" if DEVICE == "cpu" else "0",
        PROJECT,
        SUITE,
        EPOCHS,
        SEEDS,
        ONLY,
        DRY_RUN,
        SOURCE_BALANCED,
        COMMON_VALIDATION,
        DIAGNOSE_PR,
        CPU_THREADS,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J queue stopped by Ctrl+C. No next model will start.", flush=True)
        raise SystemExit(130)
