"""Edit DATA/DEVICE, then click VS Code Run. Sequential foreground E V5 training."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"  # Physical GPU index, matches nvidia-smi in the usual PCI ordering.
SUITE = "all"  # all=8 arms; priority=V5_00--03; combined=V5_04--07
EPOCHS = 300
PROJECT = f"/data/sxq/results/E_V5/CITRUS_EV5_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False
SEEDS = "42"
ONLY = ""
PRETRAINED = ""


def main():
    run_foreground(
        series="CITRUS_E_V5",
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
        # The server GPU may already be occupied by your own jobs.  Do not
        # abort on occupancy or create a cross-process device lock; the queue
        # itself remains foreground and sequential.
        device_lock=False,
        refuse_busy_gpu=False,
        single_gpu_only=False,
    )


if __name__ == "__main__":
    main()
