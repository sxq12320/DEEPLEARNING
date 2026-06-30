import torch
import random
import numpy as np
from ultralytics import YOLO
from thop import profile

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    yolo = YOLO(r"/home/sxq/data/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg.yaml")
    yolo.train(
        data = r"/home/sxq/data/ultralytics-main/1_global_yaml/001_jeruk_datasets_segmentation.yaml",
        project = r"/home/sxq/data/ultralytics-main/0_results/JEURK_SEG",
        name = "000_yolo11nano_base",
        optimizer="AdamW",
        epochs=300,
        patience=300,
        imgsz=640,
        batch=4,
        lr0=0.001,
        workers=4,
        device=0,
        cache=False,
        seed=SEED,
        amp = False,
        dropout = 0.1,

        copy_paste=0.5,   # 图像复制
        mosaic=1.0,       # 图像拼接
        mixup=0.1,        # 图像融合

        fliplr=0.5,       # 左右翻转
        flipud=0.0,       # 上下翻转（橘子通常不需要上下翻转）
        degrees=15.0,     # 随机旋转 ±15度
        scale=0.5,        # 随机缩放
        translate=0.1,    # 随机平移
    )