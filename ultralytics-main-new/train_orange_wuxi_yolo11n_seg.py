import random

import numpy as np
import torch
from ultralytics import YOLO


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # yolo11n-seg baseline for orange_wuxi citrus instance segmentation.
    # If this machine has no CUDA, change device=0 to device="cpu".
    yolo = YOLO(r"E:\mastercode\ultralytics-main-new\yolo11n-seg.pt")
    yolo.train(
        data=r"E:/mastercode/data/test/orange_wuxi_seg.yaml",
        project=r"E:/mastercode/ultralytics-main-new/1_results/ORANGE_WUXI_SEG",
        name="002_yolo11n_seg_orange_wuxi_baseline",
        optimizer="AdamW",
        epochs=300,
        patience=100,
        imgsz=640,
        batch=4,
        lr0=0.01,
        workers=4,
        device=0,
        cache=False,
        seed=SEED,
        amp=0,
        dropout=0.0,
    )
