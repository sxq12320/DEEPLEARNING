"""78 baseline versus the strongest verified V12 recipe, plus V12 component controls."""
from pathlib import Path

from citrus_paper_batch import main

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
PROJECT = "/data/sxq/results/PAPER_V12_VS78_SLICE_20261105"
BASELINE_RECIPE = "legacy78"  # legacy78 = historical P00; matched = P00 uses the V12 recipe.
BASELINE_TRAIN_VIEWS = "both"  # Both adds P00_mixed with identical 78 parameters; only training views differ.
PRETRAINED = str(Path(__file__).resolve().parent / "yolo11n-seg.pt")
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]
REPEAT_ABLATIONS = False  # True also repeats every model selected by ONLY at each seed.
CACHE = True
ONLY = ""  # Empty = all 12 runs (two P00 variants + P01-P10).
DRY_RUN = False
SMOKE = False


def arguments():
    argv = [
        "--data", DATA, "--device", DEVICE, "--project", PROJECT, "--pretrained", PRETRAINED,
        "--suite", "paper", "--epochs", str(EPOCHS), "--batch", str(BATCH),
        "--baseline-recipe", BASELINE_RECIPE,
        "--baseline-train-views", BASELINE_TRAIN_VIEWS,
        "--workers", str(WORKERS), "--seeds", ",".join(map(str, SEEDS)),
        "--cache", str(CACHE).lower(), "--resume-partial",
    ]
    if ONLY:
        argv.extend(["--only", ONLY])
    if REPEAT_ABLATIONS:
        argv.append("--repeat-ablations")
    if DRY_RUN:
        argv.append("--dry-run")
    if SMOKE:
        argv.append("--smoke")
    return argv


if __name__ == "__main__":
    main(arguments())
