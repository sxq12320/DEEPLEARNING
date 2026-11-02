"""Eleven matched-recipe controls, sequentially. See docs/THREE_RUNNERS_20261025.md."""
from pathlib import Path

from citrus_paper_batch import main

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
PROJECT = "/data/sxq/results/PAPER_V12_ABLATION_20261025"
PRETRAINED = str(Path(__file__).resolve().parent / "yolo11n-seg.pt")
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]  # Additional seeds repeat P00/P01; individual removals use the first seed.
CACHE = True
ONLY = ""  # Empty = all 11; or exact comma-separated P00...P10 names from the documentation.
DRY_RUN = False
SMOKE = False


def arguments():
    argv = [
        "--data", DATA, "--device", DEVICE, "--project", PROJECT, "--pretrained", PRETRAINED,
        "--suite", "paper", "--epochs", str(EPOCHS), "--batch", str(BATCH),
        "--workers", str(WORKERS), "--seeds", ",".join(map(str, SEEDS)),
        "--cache", str(CACHE).lower(), "--resume-partial",
    ]
    if ONLY:
        argv.extend(["--only", ONLY])
    if DRY_RUN:
        argv.append("--dry-run")
    if SMOKE:
        argv.append("--smoke")
    return argv


if __name__ == "__main__":
    main(arguments())
