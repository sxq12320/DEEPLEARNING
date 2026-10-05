"""One-epoch isolated engineering smoke; small fixture metrics are NOT research results."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import citrus_j2_batch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--only", default="J36_structure_complete,J37_density_complete")
    args = parser.parse_args()
    standard = citrus_j2_batch.settings

    def recipe(epochs, seed, imgsz=640, cache=True):
        values = standard(epochs, seed, imgsz, cache)
        values.update(imgsz=128, batch=2, workers=0, mosaic=0, close_mosaic=0,
                      copy_paste=0, warmup_epochs=0, deterministic=False)
        return values

    citrus_j2_batch.settings = recipe
    citrus_j2_batch.PROTOCOL_ID = "J2_STRUCTURE_SMOKE_NOT_RESEARCH"
    citrus_j2_batch.run(args.data, "cpu", args.project, only=args.only, epochs=1, cpu_threads=2)


if __name__ == "__main__":
    main()
