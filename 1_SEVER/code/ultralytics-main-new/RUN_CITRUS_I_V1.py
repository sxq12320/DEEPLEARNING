"""Edit DATA/DEVICE, select your Python environment, then click VS Code Run Python File."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"  # Your real server path; no confirmation/fingerprint gate.
DEVICE = "1"
SUITE = "priority"  # priority=00/01/04/05; all=10 arms; control=00/01; losses=04/06
# Recommended diagnostic queue: "mechanism" = 00/01/02/03/04.
# "feedback" = 04/06; historical "losses" is only an alias, not a loss ablation.
# See docs/I_V1_REASSESSMENT_20260921.md before comparing to scratch/AMP1 baselines.
EPOCHS = 300
PROJECT = f"/data/sxq/results/I/I_V1/CITRUS_IV1_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False  # True only builds/profiles; it does not train.
SEEDS = "42"  # Screening; final comparison uses "42,43,44" in a NEW project.
ONLY = ""  # Optional comma-separated exact YAML stems, e.g. "I04_sync_msca".
PRETRAINED = ""  # Same official yolo11n-seg.pt for every arm, NOT a cherry-picked citrus weight.


def main():
    run_foreground(
        series="CITRUS_I_V1",
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
