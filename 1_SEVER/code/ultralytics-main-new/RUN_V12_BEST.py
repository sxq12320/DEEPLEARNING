"""V12_04 full system: pretrained + mixed views + paired whole/tile validation."""
from pathlib import Path

from citrus_paper_batch import main

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
PROJECT = "/data/sxq/results/PAPER_V12_BEST_20261025"
PRETRAINED = str(Path(__file__).resolve().parent / "yolo11n-seg.pt")
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]  # For a new final repeat project, use [42, 43, 44].
CACHE = True
DRY_RUN = False
SMOKE = False


def arguments():
    argv = [
        "--data", DATA, "--device", DEVICE, "--project", PROJECT, "--pretrained", PRETRAINED,
        "--suite", "core", "--only", "P01_v12_full", "--epochs", str(EPOCHS),
        "--batch", str(BATCH), "--workers", str(WORKERS), "--seeds", ",".join(map(str, SEEDS)),
        "--cache", str(CACHE).lower(), "--resume-partial",
    ]
    if DRY_RUN:
        argv.append("--dry-run")
    if SMOKE:
        argv.append("--smoke")
    return argv


if __name__ == "__main__":
    main(arguments())
