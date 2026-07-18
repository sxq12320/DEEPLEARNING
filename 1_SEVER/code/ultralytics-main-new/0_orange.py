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
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/0_orange_yaml/001_1_yolov8-seg.yaml")
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/0_orange_yaml/001_2_yolov9c-seg.yaml")
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/0_orange_yaml/001_3_yolo11-seg.yaml")
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/0_orange_yaml/001_4_yolo12-seg.yaml")
    yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/0_orange_yaml/001_5_yolo26-seg.yaml")
    yolo.train(
        data=r"/data/sxq/code/ultralytics-main-new/200_orange_wuxi_seg.yaml",
        project=r"/data/sxq/results",
        name="001_5_yolo26-seg_adamw",
        optimizer="AdamW",
        epochs=300,
        patience=100,
        imgsz=640,
        batch=16,
        lr0=0.001,
        workers=4,
        device=1,
        cache=True,
        seed=SEED,
        amp=1,
        dropout=0.1,


        # hsv_h=0.015,           # 色调变化
        # hsv_s=0.7,             # 饱和度变化
        # hsv_v=0.5,             # 明度变化
        # degrees=15.0,          # 旋转角度
        # translate=0.1,         # 平移比例
        # scale=0.5,             # 缩放比例
        # shear=2.0,             # 剪切变换
        # perspective=0.0,       # 透视变换
        # flipud=0.0,            # 垂直翻转
        # fliplr=0.5,            # 水平翻转
        # mosaic=1.0,            # 马赛克增强
        # mixup=0.1,             # 混合增强
        # copy_paste=0.2,        # 复制粘贴增强
        # erasing=0.4,           # 随机擦除
    )