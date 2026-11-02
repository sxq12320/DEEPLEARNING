"""Edit DATA/DEVICE/PROJECT, then VS Code -> Run Python File. Foreground, sequential."""
from citrus_control78_batch import main

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"  # One physical CUDA index, or "cpu" for an engineering smoke test.
PROJECT = "/data/sxq/results/PAPER_CONTROL78_20261025"
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]  # After screening, [42, 43, 44] reuses completed seed42.
ONLY = "all"  # all / baseline / candidate
DRY_RUN = False
SMOKE = False  # Separate project suffix, 1 epoch, two images in post-training eval.


if __name__ == "__main__":
    argv = [
        "--data", DATA, "--device", DEVICE, "--project", PROJECT,
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--workers", str(WORKERS),
        "--seeds", ",".join(map(str, SEEDS)), "--only", ONLY,
    ]
    if DRY_RUN:
        argv.append("--dry-run")
    if SMOKE:
        argv.append("--smoke")
    main(argv)
