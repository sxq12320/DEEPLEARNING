"""Select your Python environment; edit DATA/DEVICE; click VS Code Run Python File."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"  # Your own cleaned dataset path; no confirmation/fingerprint gate.
DEVICE = "1"
SUITE = "all"  # all=12 (10 core + paper pair); priority=7 core; paper=06/10/11; control=00/01
EPOCHS = 300
PROJECT = f"/data/sxq/results/E/E_V11/CITRUS_EV11_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False
SEEDS = "42"
ONLY = ""
PRETRAINED = ""  # Empty = same official yolo11n-seg.pt initialization for ALL arms, not V10 trained weights.


def main():
    run_foreground(
        series="CITRUS_E_V11",
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
