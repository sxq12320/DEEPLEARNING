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
    # yolo = YOLO(r"E:\mastercode\ultralytics-main-new\0_orange_yaml\001_yolo11-seg.yaml")
    # yolo = YOLO(r"E:\mastercode\ultralytics-main-new\0_orange_yaml\002_yolo11-seg-starnet.yaml")
    yolo = YOLO(r"E:\mastercode\ultralytics-main-new\0_orange_yaml\003_yolo11-seg-mobilenetv4.yaml")
    yolo.train(
        data=r"E:\mastercode\ultralytics-main-new\200orange_wuxi_seg.yaml",
        project=r"E:\mastercode\ultralytics-main-new\1_results",
        name="003_yolo11n_mobilenetv4_seg_AdamW",
        optimizer="AdamW",
        epochs=300,
        patience=100,
        imgsz=640,
        batch=4,
        lr0=0.001,
        workers=4,
        device=0,
        cache=False,
        seed=SEED,
        amp=0,
        dropout=0.1,
    )