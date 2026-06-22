"""
SHR Watermelon Flower Pose 训练脚本 (6关键点版)
================================================
使用 YOLO11n-pose 训练西瓜花关键点检测模型
关键点: stigma(蕊心) + petal1-5(5片花瓣)
"""

import torch
import random
import numpy as np
from ultralytics import YOLO

SEED = 42
EPOCHS = 300
BATCH = 16
IMGSZ = 640
DATA = r"E:/mastercode/ultralytics-main-new/207_shr_watermelon_6pt.yaml"
PROJECT = r"E:/mastercode/ultralytics-main-new/results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)

    yolo = YOLO("yolo11n-pose.pt")
    
    yolo.train(
        data=DATA,
        project=PROJECT,
        name="08_shr_watermelon_6pt_nano",
        epochs=EPOCHS,
        patience=300,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer="MuSGD",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        copy_paste=0.0,
        mosaic=1.0,
        mixup=0.0,
        pose=6.0,
        kobj=1.0,
        device=0,
        exist_ok=True,
    )

    print("\n训练完成！")


if __name__ == "__main__":
    main()
