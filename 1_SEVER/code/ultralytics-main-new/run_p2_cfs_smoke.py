"""One-click local smoke run for the P2CFS YOLO11n-seg architecture.

Edit only the constants below. Set ``EPOCHS = 300`` after the 3-epoch smoke run
passes and use a new ``NAME`` for every completed experiment.
"""

from __future__ import annotations

import os

# Must be set before importing torch through the training driver.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from train_citrus_seg import FIXED, PROJECT, build_model, set_seed


MODEL = r"E:/mastercode/ultralytics-main-new/0_orange_yaml/012_yolo11-seg-p2-cfs.yaml"
PRETRAINED = "yolo11n-seg.pt"  # Ultralytics downloads it if it is not present locally.
DATASET = r"E:/mastercode/data/orange_yolo/data.yaml"
NAME = "E1_yolo11n_seg_p2_cfs_smoke"
EPOCHS = 3
BATCH = 4
IMGSZ = 640
DEVICE = "0"


def main() -> None:
    set_seed(42)
    model = build_model(MODEL, PRETRAINED)
    model.train(
        data=DATASET,
        project=PROJECT,
        name=NAME,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        **FIXED,
    )


if __name__ == "__main__":
    main()
