"""可视化工具库。

主要用于训练过程中损失曲线和验证时分割结果的可视化和保存。
"""

from typing import List

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses: List[float], save_path: str):
    """绘制并保存训练损失曲线。

    Args:
        losses (List[float]): 每个 epoch 的损失列表。
        save_path (str): 保存路径。
        
    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_sample_prediction(image: np.ndarray, mask_gt: np.ndarray, mask_pred: np.ndarray, save_path: str):
    """绘制原图、真值掩码与预测掩码对比图。

    Args:
        image (np.ndarray): 原始图像（BGR）。
        mask_gt (np.ndarray): 真值掩码。
        mask_pred (np.ndarray): 预测掩码。
        save_path (str): 保存路径。
        
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image')
    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(mask_pred, cmap='gray')
    axes[2].set_title('Prediction')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction comparison saved to {save_path}")
