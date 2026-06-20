"""
SHR Watermelon Flower Pose 训练脚本
=====================================

使用 YOLO11-pose 训练西瓜花关键点检测模型。
数据集: 5 类花 (blooming_male, unknown, closed_male, blooming_female, closed_female)
        1 个关键点 (蕊心) + 可见性 (0=invisible, 1=partial, 2=full)
        bbox 来源于分割多边形，关键点来源于 pose JSON

数据划分 (仅保留有关键点标注的目标):
    train: 132 张 (224 个目标, 全部有蕊心关键点)
    val:    37 张 (60 个目标)
    test:   20 张 (33 个目标)

使用方式：
    python 008_shr_watermelon.py

输出：
    results/ 目录下生成训练结果
"""

import torch
import random
import numpy as np
from ultralytics import YOLO

SEED = 42
EPOCHS = 600
BATCH = 12
IMGSZ = 640
DATA = r"E:/mastercode/ultralytics-main-new/207_shr_watermelon.yaml"
PROJECT = r"E:/mastercode/ultralytics-main-new/results"
MODEL = r"E:/mastercode/ultralytics-main-new/ultralytics/cfg/models/11/yolo11-pose.yaml"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)

    print("=" * 60)
    print("  SHR Watermelon Flower Pose 训练")
    print("=" * 60)
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH}  |  imgsz: {IMGSZ}")
    print(f"  Data: {DATA}")
    print(f"  Model: {MODEL}")
    print(f"  Classes: 5 (blooming_male, unknown, closed_male,")
    print(f"            blooming_female, closed_female)")
    print(f"  Keypoint: 1 (蕊心) with visibility")
    print()

    yolo = YOLO(r"E:/mastercode/ultralytics-main-new/ultralytics/cfg/models/11/yolo11-pose.yaml")
    yolo.train(
        data=r"E:/mastercode/ultralytics-main-new/207_shr_watermelon.yaml",
        project=r"E:/mastercode/ultralytics-main-new/results",
        name="08_shr_watermelon_flower_pose_base",
        epochs=600,
        patience=300,
        imgsz=(960,640),
        batch=12,
        optimizer="AdamW",
        copy_paste=0.3,
        mosaic=1.0,
        mixup=0.1,
        cos_lr=True
    )

    print("\n训练完成！")
    print(f"结果保存至: {PROJECT}/08_shr_watermelon_flower_pose")

    # 用 best.pt 进行验证
    print("\n开始验证...")
    best_model = YOLO(f"{PROJECT}/08_shr_watermelon_flower_pose/weights/best.pt")
    val_results = best_model.val(data=DATA, imgsz=IMGSZ, batch=BATCH)

    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Pose mAP50:    {val_results.pose.map50:.4f}")
    print(f"  Pose mAP50-95: {val_results.pose.map:.4f}")


if __name__ == "__main__":
    main()
