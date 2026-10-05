"""Foreground speed check / resume for an existing baseline. Edit SETTINGS, then click Run in VSCode."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SETTINGS = {
    "RUN_DIR": "",  # Existing model run directory containing job.json/effective_config.py/train/.
    "DEVICE": 1,
    "MODE": "profile",  # profile: time 5 large photos; resume: continue MMDet; evaluate: re-evaluate saved weights.
    "PROFILE_IMAGES": 5,
    "CHECKPOINT": "",  # Optional epoch/best checkpoint. Blank: trained.json or train/last_checkpoint.
    "PYTHONS": {
        "modern": str(Path.home() / "miniconda3/envs/citrus_baseline/python.exe")
        if os.name == "nt"
        else str(Path.home() / ".conda/envs/citrus_baseline/bin/python"),
        "mmdet": str(Path.home() / "miniconda3/envs/citrus_mmdet/python.exe")
        if os.name == "nt"
        else str(Path.home() / ".conda/envs/citrus_mmdet/bin/python"),
    },
}

if __name__ == "__main__":
    from baseline_amp.recovery import main

    try:
        main(SETTINGS)
    except KeyboardInterrupt:
        print("Stopped. No next stage starts.")
        raise SystemExit(130)
