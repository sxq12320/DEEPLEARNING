"""Edit DATA/DEVICE, then click VS Code Run. Sequential foreground E V2 training."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"  # Physical GPU index, matches nvidia-smi in the usual PCI ordering.
SUITE = "all"  # all=8 arms; priority=E20/E21/E22/E25; guided=E25/E27
EPOCHS = 300
PROJECT = f"/data/sxq/results/E_V2/CITRUS_EV2_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False
SEEDS = "42"
ONLY = ""
PRETRAINED = ""


def main():
    run_foreground(
        series="CITRUS_E_V2",
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
        device_lock=True,
        refuse_busy_gpu=True,
        single_gpu_only=True,
    )


if __name__ == "__main__":
    main()
