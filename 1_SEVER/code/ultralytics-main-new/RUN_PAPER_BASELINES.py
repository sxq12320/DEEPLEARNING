"""Current baseline batch entry. Edit SETTINGS, then click VSCode Run Python File.

Legacy78 launchers were retired on 2026-10-11; saved results remain unchanged.
This track uses official COCO checkpoints,
backend-native optimization conventions and identical common COCO validation.
It is NOT a controlled initialization ablation against scratch J experiments.
"""

import os
from pathlib import Path

from baseline_amp import default_pythons
from baseline_amp.registry import COMMON_VALIDATION

ROOT = Path(__file__).resolve().parent
SETTINGS = {
    "DATA": ("E:/mastercode/data/orange_yolo_grouped_dedup_20260820/data.yaml" if os.name == "nt"
             else "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"),
    "PROJECT": (str(ROOT / "1_results/PAPER_BASELINES_TRANSFER_300EP_20261011") if os.name == "nt"
                else "/data/sxq/results/BASELINES/PAPER_BASELINES_TRANSFER_300EP_20261011"),
    "DEVICE": 1,
    "SUITE": "all",  # Seven original model families, each AMP=1 and AMP=0.
    "MODELS": None,  # Or ["rtmdet_ins_tiny"] for the first smoke run.
    "RECIPE_POLICY": "official_transfer",
    "VALIDATION": dict(COMMON_VALIDATION),
    "EPOCHS": 300,  # Each selected model/AMP job runs 300 epochs; shorter checks use a NEW PROJECT.
    "SEEDS": [42],  # Final baseline and final method: [42, 43, 44].
    "AMP_MODES": [1, 0],
    "WORKERS": 4,
    "BATCHES": {},
    "PYTHONS": default_pythons(),
    "START_FROM": "",
    "SKIP_RUNS": [],
}

if __name__ == "__main__":
    from baseline_amp.batch import main

    try:
        main(SETTINGS)
    except KeyboardInterrupt:
        print("Stopped. No next model starts.", flush=True)
        raise SystemExit(130)
