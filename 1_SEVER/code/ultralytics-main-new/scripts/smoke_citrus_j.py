"""Isolated one-epoch CPU integration test; never use its metrics as research results."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import citrus_j_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--only", default="J16_joint_mask_loss")
    args = parser.parse_args()
    original_settings = citrus_j_batch.settings

    def smoke_settings(epochs, seed):
        recipe = original_settings(epochs, seed)
        recipe.update(
            imgsz=128, batch=2, workers=0, mosaic=0, close_mosaic=0, copy_paste=0, warmup_epochs=0, deterministic=False
        )
        return recipe

    citrus_j_batch.settings = smoke_settings
    citrus_j_batch.PROTOCOL_ID = "J_SMOKE_NOT_RESEARCH"
    citrus_j_batch.run(args.data, "cpu", args.project, epochs=1, only=args.only, cpu_threads=2)


if __name__ == "__main__":
    main()
