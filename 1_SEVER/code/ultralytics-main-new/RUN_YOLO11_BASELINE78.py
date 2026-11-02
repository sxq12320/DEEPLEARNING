"""78 recipe: train whole-image and mixed-tile YOLO11n, then paired validation."""
from citrus_control78_batch import main

# Edit these paths for YOUR server. Run Python File in VS Code; no nohup needed.
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
PROJECT = "/data/sxq/results/PAPER_YOLO11_78_20261025"
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]
TRAIN_VIEWS = "both"  # both / whole / mixed. Both trains TWO models sequentially.
DRY_RUN = False
SMOKE = False  # One epoch in a separate _SMOKE project; limited post-training evaluation.


def arguments():
    argv = [
        "--data", DATA, "--device", DEVICE, "--project", PROJECT,
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--workers", str(WORKERS),
        "--seeds", ",".join(map(str, SEEDS)), "--only", "baseline", "--train-views", TRAIN_VIEWS,
    ]
    if DRY_RUN:
        argv.append("--dry-run")
    if SMOKE:
        argv.append("--smoke")
    return argv


if __name__ == "__main__":
    main(arguments())
