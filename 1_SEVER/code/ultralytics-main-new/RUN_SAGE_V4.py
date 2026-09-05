"""Reconstructed SAGE-v4: edit configuration and click VS Code's Run triangle.

Foreground, one model at a time. Ctrl+C stops the entire queue. No nohup needed.
Old SAGE30--35 and their runner remain available unchanged.
"""

from citrus_foreground import run_foreground

# ========================= EDIT THESE ON THE SERVER ==========================
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"  # Your own cleaned dataset YAML.
DEVICE = "0"
SUITE = "screen"  # screen / geometry / backbone / all / control / smoke
EPOCHS = 50  # First screen at a common budget. Change to 300 AND use a new PROJECT later.
PROJECT = "/data/sxq/results/SAGE/CITRUS_SAGE_V4R_SCREEN_50EP"
DRY_RUN = True  # First build-check. After success, False enables actual training.
SEEDS = "42"  # Final selected comparison: "42,43,44".
ONLY = ""  # Optional exact YAML stem(s), comma-separated; see the guide.
PRETRAINED = ""  # Empty = code-directory/yolo11n-seg.pt.
# Fixed protocol: batch=16, imgsz=640, workers=4, AdamW, lr=.001, AMP=False.
# ============================================================================


def main():
    run_foreground(
        series="SAGE_V4R",
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
        cache="false",
        amp=None,
        dry_run=DRY_RUN,
        skip_completed=True,
        fail_fast=True,
        device_lock=True,
        refuse_busy_gpu=True,
        single_gpu_only=True,
    )


if __name__ == "__main__":
    main()
