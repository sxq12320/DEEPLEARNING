"""Edit DATA/DEVICE below; click VS Code Run to train sequentially in this terminal.

E trains nine experiments sequentially, including E08 guided crops. Ctrl+C stops the queue.
Previous series are untouched. See docs/CITRUS_E_DESIGN.md for the evidence and one-model API.
"""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"
SUITE = "all"  # Nine experiments E00--E08; guided=E03 versus E08 only
EPOCHS = 300
PROJECT = f"/data/sxq/results/E/CITRUS_E9_GUIDED_DEVICEBOUND_{SUITE.upper()}_{EPOCHS}EP"  # Fresh device-binding protocol
DRY_RUN = False  # True only checks construction and exits WITHOUT training.
SEEDS = "42"
ONLY = ""  # Optional exact YAML stem(s), comma-separated.
PRETRAINED = ""  # Empty uses this code folder's yolo11n-seg.pt.


def main():
    run_foreground(
        series="CITRUS_E",
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
