"""Edit DATA and DEVICE; click VS Code Run Python File for sequential foreground experiments."""
from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
SUITE = "priority"  # first five isolated arms; all adds three mechanism controls
EPOCHS = 50
PROJECT = f"/data/sxq/results/I/I_V4/CITRUS_IV4_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False
SEEDS = "42"
ONLY = ""
PRETRAINED = ""  # Same official YOLO11n-seg initialization as I V2/V3; NOT legacy78 scratch.


def main():
    run_foreground(series="CITRUS_I_V4", data=DATA, device=DEVICE, suite=SUITE, epochs=EPOCHS,
                   project=PROJECT, seeds=SEEDS, only=ONLY, pretrained=PRETRAINED,
                   batch=16, imgsz=640, workers=4, cache=True, amp=False,
                   dry_run=DRY_RUN, skip_completed=True, fail_fast=True,
                   device_lock=False, refuse_busy_gpu=False, single_gpu_only=False)


if __name__ == "__main__":
    main()
