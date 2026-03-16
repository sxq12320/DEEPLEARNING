"""
FCN 训练 + 验证脚本
用法：
    python train.py --data_root /path/to/dataset --num_classes 21 --model fcn8s
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import build_model
from dataset import build_dataloaders


# ──────────────────────────────────────────────
# 评估指标：mean IoU
# ──────────────────────────────────────────────

def compute_miou(preds, labels, num_classes, ignore_index=255):
    """
    preds:  (N, H, W) int64 预测类别
    labels: (N, H, W) int64 真实类别
    返回：每类IoU列表 + mean IoU
    """
    iou_list = []
    for cls in range(num_classes):
        pred_mask  = (preds == cls)
        label_mask = (labels == cls)
        ignore     = (labels == ignore_index)

        pred_mask  = pred_mask  & ~ignore
        label_mask = label_mask & ~ignore

        intersection = (pred_mask & label_mask).sum().item()
        union        = (pred_mask | label_mask).sum().item()

        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)

    valid = [x for x in iou_list if not np.isnan(x)]
    miou  = np.mean(valid) if valid else 0.0
    return iou_list, miou


# ──────────────────────────────────────────────
# 单次训练 epoch
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0

    for i, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)          # (N, C, H, W)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"  Epoch {epoch} [{i+1}/{len(loader)}]  loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss    = criterion(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)  # (N, H, W)
            all_preds.append(preds.cpu())
            all_labels.append(masks.cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    _, miou = compute_miou(all_preds, all_labels, num_classes)
    avg_loss = total_loss / len(loader)

    return avg_loss, miou


# ──────────────────────────────────────────────
# 主训练流程
# ──────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ── 数据 ──
    train_loader, val_loader = build_dataloaders(
        voc_root    = args.data_root,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── 模型 ──
    model = build_model(args.model, num_classes=args.num_classes)
    model = model.to(device)
    print(f"模型: {args.model.upper()}，类别数: {args.num_classes}")

    # ── 损失函数：忽略255（边界/未标注区域）──
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ── 优化器（论文使用 SGD + momentum）──
    optimizer = optim.SGD(
        [
            # 卷积层权重：正常学习率
            {'params': [p for n, p in model.named_parameters()
                        if 'bias' not in n], 'lr': args.lr},
            # 偏置：学习率×2（FCN原文设定）
            {'params': [p for n, p in model.named_parameters()
                        if 'bias' in n], 'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # 学习率调度：每 step_size 个 epoch 衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # ── 训练循环 ──
    best_miou  = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, miou = validate(model, val_loader, criterion, device, args.num_classes)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"mIoU: {miou:.4f} | "
              f"Time: {elapsed:.1f}s")

        # 保存最优模型
        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'miou':        miou,
            }, ckpt_path)
            print(f"  ✓ 保存最优模型 → {ckpt_path}  (mIoU={miou:.4f})")

        # 每N个epoch保存一次checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_epoch{epoch}.pth")
            torch.save({'epoch': epoch, 'model_state': model.state_dict()}, ckpt_path)

    print(f"\n训练完成！最优 mIoU = {best_miou:.4f}")


# ──────────────────────────────────────────────
# 参数配置
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCN 训练脚本')

    # 数据
    parser.add_argument('--data_root', type=str, default=r'E:\mastercode\data\VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007', help='数据集根目录')
    parser.add_argument('--num_classes', type=int, default=21, help='类别数（含背景）')
    parser.add_argument('--img_size',     type=int,   default=512,     help='输入图像尺寸')

    # 模型
    parser.add_argument('--model',        type=str,   default='fcn32s',
                        choices=['fcn32s', 'fcn16s', 'fcn8s'])

    # 训练
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--lr_step',      type=int,   default=50,      help='学习率衰减间隔(epoch)')
    parser.add_argument('--num_workers',  type=int,   default=4)

    # 保存
    parser.add_argument('--save_dir',     type=str,   default='checkpoints')
    parser.add_argument('--save_interval',type=int,   default=10,      help='每N个epoch保存一次')

    args = parser.parse_args()
    main(args)
