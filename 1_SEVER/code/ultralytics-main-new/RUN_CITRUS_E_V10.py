"""Edit DATA/DEVICE, select this Python environment, then click VS Code Run."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"  # Your server path; it must point to your cleaned split.
DEVICE = "1"
SUITE = "all"  # all=10; priority=01/02/03/04/07/08
EPOCHS = 300
PROJECT = f"/data/sxq/results/E/E_V10/CITRUS_EV10_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False
SEEDS = "42"
ONLY = ""
PRETRAINED = ""


def main():
    run_foreground(
        series="CITRUS_E_V10",
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
