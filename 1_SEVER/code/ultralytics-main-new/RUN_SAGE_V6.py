"""Edit DATA/DEVICE below; click VS Code Run to train sequentially in this terminal.

The default trains five structural configurations sequentially. Ctrl+C stops the queue.
V5 is untouched. See docs/SAGE_V6_TRAINING.md for the evidence and one-model API.
"""

from citrus_foreground import run_foreground

DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"
SUITE = "structure"  # structure=60--64; all=6 incl. slower geometry; priority=60/62/64
EPOCHS = 300
PROJECT = f"/data/sxq/results/SAGE/CITRUS_SAGE_V6_{SUITE.upper()}_{EPOCHS}EP"
DRY_RUN = False  # True only checks construction and exits WITHOUT training.
SEEDS = "42"
ONLY = ""  # Optional exact YAML stem(s), comma-separated.
PRETRAINED = ""  # Empty uses this code folder's yolo11n-seg.pt.


def main():
    run_foreground(
        series="SAGE_V6", data=DATA, device=DEVICE, suite=SUITE, epochs=EPOCHS,
        project=PROJECT, seeds=SEEDS, only=ONLY, pretrained=PRETRAINED,
        batch=16, imgsz=640, workers=4, cache=True, amp=None,
        dry_run=DRY_RUN, skip_completed=True, fail_fast=True,
        device_lock=True, refuse_busy_gpu=True, single_gpu_only=True,
    )


if __name__ == "__main__":
    main()
