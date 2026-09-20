"""Edit the SETTINGS below, select a Python interpreter, and click VSCode's Run Python File triangle."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ===== Only edit this block. Keep each model's AMP0/AMP1 batch identical. =====
SETTINGS = {
    "DATA": (
        "E:/mastercode/data/orange_yolo_grouped_dedup_20260820/data.yaml"
        if os.name == "nt"
        else "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
    ),
    "PROJECT": (
        str(ROOT / "1_results/BASELINES_LEGACY78_SCRATCH_300EP")
        if os.name == "nt"
        else "/data/sxq/results/BASELINES/BASELINES_LEGACY78_SCRATCH_300EP"
    ),
    "DEVICE": 1,  # Physical GPU index. No occupancy guard; only this GPU is visible to children.
    "SUITE": "all",  # all / yolo / mmdet / rfdetr / anchor
    "EPOCHS": 300,  # First use 1-3 in a NEW PROJECT for smoke testing; then 300.
    "SEEDS": [42],  # Screening: [42]. Final paper repeats: [42, 43, 44].
    "AMP_MODES": [1],  # Legacy78 defaults to AMP=1. Set [1, 0] to run the previously requested paired audit.
    "WORKERS": 4,
    "BATCHES": {},  # e.g. {"mask_rcnn_r50": 1}; changes BOTH AMP jobs, requires a NEW PROJECT.
    "PYTHONS": {
        "modern": str(Path.home() / "miniconda3/envs/citrus_baseline/python.exe")
        if os.name == "nt"
        else str(Path.home() / ".conda/envs/citrus_baseline/bin/python"),
        "mmdet": str(Path.home() / "miniconda3/envs/citrus_mmdet/python.exe")
        if os.name == "nt"
        else str(Path.home() / ".conda/envs/citrus_mmdet/bin/python"),
    },
    "START_FROM": "",  # Optional EXACT job name printed by --dry-run. Earlier jobs are skipped explicitly.
    "SKIP_RUNS": [],  # Explicit incomplete-job skip list; never treated as successful runs.
}
# ========================================================================

if __name__ == "__main__":
    from baseline_amp.batch import main

    try:
        main(SETTINGS)
    except KeyboardInterrupt:
        print("Stopped by user. No next baseline will start.")
        raise SystemExit(130)
