"""Edit DATA/DEVICE and click VS Code's Run Python File; models train sequentially in foreground."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
SUITE = "priority"  # control/priority/screen/losses/recognition/all/smoke
EPOCHS = 50
PROJECT = f"/data/sxq/results/I/I_V3/CITRUS_IV3_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False  # Click Run Python File to train directly; set True only for a build/profile check.
SEEDS = "42"
ONLY = ""  # Optional exact stems separated by commas.
PRETRAINED = ""  # Empty means the common official yolo11n-seg.pt.


def main():
    run_foreground(
        series="CITRUS_I_V3",
        data=DATA,
        device=DEVICE,
        suite=SUITE,
        epochs=EPOCHS,
        project=PROJECT,
        seeds=SEEDS,
        only=ONLY,
        pretrained=PRETRAINED,
        batch=16,
        imgsz=640,
        workers=4,
        cache=True,
        amp=False,
        dry_run=DRY_RUN,
        skip_completed=True,
        fail_fast=True,
        device_lock=False,
        refuse_busy_gpu=False,
        single_gpu_only=False,
    )


if __name__ == "__main__":
    main()
