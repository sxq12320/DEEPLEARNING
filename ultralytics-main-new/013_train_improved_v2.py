"""
训练改进网络 V2：Multi-Scale 1D CNN + SE + Attention Pooling
=============================================================
使用和010相同的数据，对比训练效果
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
import random
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 导入改进网络
import sys
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

# 动态导入（避免文件名数字开头的问题）
net_module = import_module("013_improved_net_v2")
ImprovedContourNetV2 = net_module.ImprovedContourNetV2

# ============ 配置 ============
RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
NUM_BOUNDARY_POINTS = 64
SAVE_DIR = os.path.join(RESULTS_DIR, "13_improved_net_v2")


# ============ 数据集 ============
def extract_boundary_points(mask, num_points=64):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    indices = np.linspace(0, len(contour_points) - 1, num_points).astype(int)
    sampled_points = contour_points[indices]
    h, w = mask.shape
    normalized = sampled_points.astype(np.float32)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized.flatten()


def extract_hsv_features(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flower_pixels = hsv[mask > 0]
    if len(flower_pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.array([
        np.mean(flower_pixels[:, 0]) / 180.0,
        np.mean(flower_pixels[:, 1]) / 255.0,
        np.mean(flower_pixels[:, 2]) / 255.0
    ], dtype=np.float32)


class YOLOSegPollinationDataset(Dataset):
    def __init__(self, img_dir, label_dir, seg_model, num_boundary_points=64):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.num_boundary_points = num_boundary_points
        self.img_files = [f for f in os.listdir(img_dir)
                          if f.endswith('.jpg') and not f.startswith('annotations')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)

        image = cv2.imread(img_path)
        if image is None:
            return self._dummy()
        h, w = image.shape[:2]

        results = self.seg_model.predict(img_path, conf=0.25, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for r in results[0].masks:
                mask_data = r.data.cpu().numpy()[0]
                mask_resized = cv2.resize(mask_data, (w, h))
                mask[mask_resized > 0.5] = 255

        boundary = extract_boundary_points(mask, self.num_boundary_points)
        hsv_feat = extract_hsv_features(image, mask)

        json_name = img_file.replace('.jpg', '.json')
        label_path = os.path.join(self.label_dir, json_name)
        gt_center = np.array([0.5, 0.5])
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data['shapes']:
                if shape['label'] == 'fully_visible':
                    gt_center = np.array([shape['points'][0][0] / w, shape['points'][0][1] / h])
                    break

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flower_center = np.array([0.5, 0.5])
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                flower_center = np.array([M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h])

        return {
            'boundary': torch.tensor(boundary) if boundary is not None else torch.zeros(self.num_boundary_points * 2),
            'hsv': torch.tensor(hsv_feat),
            'flower_center': torch.tensor(flower_center),
            'gt_center': torch.tensor(gt_center),
            'valid': boundary is not None
        }

    def _dummy(self):
        return {
            'boundary': torch.zeros(self.num_boundary_points * 2),
            'hsv': torch.zeros(3),
            'flower_center': torch.tensor([0.5, 0.5]),
            'gt_center': torch.tensor([0.5, 0.5]),
            'valid': False
        }


# ============ 训练 ============
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("训练改进网络 V2: Multi-Scale CNN + SE + Attention Pool")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 加载分割模型
    print(f"\n加载分割模型: {SEG_MODEL_PATH}")
    seg_model = YOLO(SEG_MODEL_PATH)

    # 2. 创建数据集
    print("创建数据集...")
    train_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\train",
        r"E:\mastercode\data\shr_watermelon\pose\labels\train",
        seg_model, NUM_BOUNDARY_POINTS
    )
    val_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\val",
        r"E:\mastercode\data\shr_watermelon\pose\labels\val",
        seg_model, NUM_BOUNDARY_POINTS
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")

    # 3. 创建模型
    model = ImprovedContourNetV2(num_boundary_points=NUM_BOUNDARY_POINTS, base_channels=64, num_blocks=3)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    # 4. 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()  # Huber Loss, 比MSE对异常值更鲁棒

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n开始训练...")
    best_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(100), desc="训练进度", ncols=100)

    for epoch in epoch_pbar:
        # ---- 训练 ----
        model.train()
        train_loss = 0
        train_count = 0

        for batch in train_loader:
            if not batch['valid'].any():
                continue

            boundary = batch['boundary'][batch['valid']].to(device)
            hsv = batch['hsv'][batch['valid']].to(device)
            flower_center = batch['flower_center'][batch['valid']].to(device)
            gt_center = batch['gt_center'][batch['valid']].to(device)

            offset = model(boundary, hsv)
            pred_center = flower_center + offset

            loss = criterion(pred_center, gt_center)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        scheduler.step()

        # ---- 验证 ----
        model.eval()
        val_loss = 0
        val_count = 0
        errors = []

        with torch.no_grad():
            for batch in val_loader:
                if not batch['valid'].any():
                    continue

                boundary = batch['boundary'][batch['valid']].to(device)
                hsv = batch['hsv'][batch['valid']].to(device)
                flower_center = batch['flower_center'][batch['valid']].to(device)
                gt_center = batch['gt_center'][batch['valid']].to(device)

                offset = model(boundary, hsv)
                pred_center = flower_center + offset

                loss = criterion(pred_center, gt_center)
                val_loss += loss.item()
                val_count += 1

                for i in range(pred_center.shape[0]):
                    err = torch.sqrt(((pred_center[i] - gt_center[i]) * 640) ** 2).sum().item()
                    errors.append(err)

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        mean_error = np.mean(errors) if errors else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_pbar.set_postfix({
            'train': f"{train_loss:.6f}",
            'val': f"{val_loss:.6f}",
            'err': f"{mean_error:.1f}px"
        })

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))

    epoch_pbar.close()

    # ---- 保存训练曲线 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label='Train Loss', alpha=0.8)
    axes[0].plot(val_losses, label='Val Loss', alpha=0.8)
    axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch {best_epoch}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_losses, label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), dpi=150)

    # ---- 评估 ----
    print(f"\n{'=' * 60}")
    print(f"训练完成！最佳Epoch: {best_epoch + 1}")
    print(f"最佳验证损失: {best_loss:.6f}")
    print(f"模型保存: {SAVE_DIR}/best.pth")
    print(f"{'=' * 60}")

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best.pth"), map_location=device))
    model.eval()

    all_errors = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="最终评估"):
            if not batch['valid'].any():
                continue
            boundary = batch['boundary'][batch['valid']].to(device)
            hsv = batch['hsv'][batch['valid']].to(device)
            flower_center = batch['flower_center'][batch['valid']].to(device)
            gt_center = batch['gt_center'][batch['valid']].to(device)

            offset = model(boundary, hsv)
            pred_center = flower_center + offset

            for i in range(pred_center.shape[0]):
                err = torch.sqrt(((pred_center[i] - gt_center[i]) * 640) ** 2).sum().item()
                all_errors.append(err)

    all_errors = np.array(all_errors)
    print(f"\n{'=' * 60}")
    print("最终评估结果")
    print(f"{'=' * 60}")
    print(f"  总样本数:   {len(all_errors)}")
    print(f"  平均误差:   {np.mean(all_errors):.2f} px")
    print(f"  中位数误差: {np.median(all_errors):.2f} px")
    print(f"  <10px:      {np.sum(all_errors < 10)} ({np.sum(all_errors < 10) / len(all_errors) * 100:.1f}%)")
    print(f"  <20px:      {np.sum(all_errors < 20)} ({np.sum(all_errors < 20) / len(all_errors) * 100:.1f}%)")
    print(f"  <30px:      {np.sum(all_errors < 30)} ({np.sum(all_errors < 30) / len(all_errors) * 100:.1f}%)")

    # 保存结果
    with open(os.path.join(SAVE_DIR, "results.json"), 'w') as f:
        json.dump({
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_loss,
            'mean_error_px': float(np.mean(all_errors)),
            'median_error_px': float(np.median(all_errors)),
            'params': total_params,
        }, f, indent=2)


if __name__ == "__main__":
    main()
