"""Edit DATA/DEVICE, select your Python environment, then click VS Code Run Python File."""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"  # Edit only if server path differs.
DEVICE = "1"
SUITE = "priority"  # priority=control+three P2 mechanisms; all=10 only after screening.
EPOCHS = 50
PROJECT = f"/data/sxq/results/I/I_V2/CITRUS_IV2_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = True  # First click: build/profile only. Change to False after all BUILD OK lines.
SEEDS = "42"  # Screening; final comparison uses "42,43,44" in a NEW project.
ONLY = ""  # Optional exact stems, e.g. "I20_corrected_control,I23_p2_semantic".
PRETRAINED = ""  # Same official yolo11n-seg.pt for every arm, NOT a cherry-picked citrus weight.


def main():
    run_foreground(
        series="CITRUS_I_V2",
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
