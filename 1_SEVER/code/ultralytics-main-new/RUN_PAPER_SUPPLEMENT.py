"""VS Code foreground entry. Edit the paths, then click Run Python File."""

from citrus_paper_batch import main

DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"  # Physical CUDA index, or "cpu"; no busy-GPU lock or protection.
PROJECT = "/data/sxq/results/PAPER_SUPPLEMENT_20261014"
PRETRAINED = "/data/sxq/code/ultralytics-main-new/yolo11n-seg.pt"
SUITE = "paper"  # paper=15 trainings; core=6 main trainings.
EPOCHS = 300
SEEDS = [42, 43, 44]
CACHE = True
AMP_AUDIT = False  # Optional extra two seed42 AMP=True runs, reported separately.
EVALUATE_TEST = True  # Only the two PREDECLARED main networks; not all ablations.
COCO_SECONDARY = True  # Post-training original-image COCO, never replaces EV6 validation.
RESUME_PARTIAL = True
DRY_RUN = False  # Print queue without importing Torch, preparing views or training.


if __name__ == "__main__":
    arguments = [
        "--data", DATA, "--device", DEVICE, "--project", PROJECT, "--pretrained", PRETRAINED,
        "--suite", SUITE, "--epochs", str(EPOCHS), "--seeds", ",".join(map(str, SEEDS)),
        "--cache", "true" if CACHE else "false",
    ]
    if AMP_AUDIT:
        arguments.append("--amp-audit")
    if EVALUATE_TEST:
        arguments.append("--evaluate-test")
    if COCO_SECONDARY:
        arguments.append("--coco-secondary")
    if RESUME_PARTIAL:
        arguments.append("--resume-partial")
    if DRY_RUN:
        arguments.append("--dry-run")
    main(arguments)
