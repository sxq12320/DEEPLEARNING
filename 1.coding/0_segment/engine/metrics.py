"""常用评估指标模块。

在此可根据不同的评估要求，计算 IoU, Dice 等常见分割指标。
"""

from typing import List

import numpy as np
import torch


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    """计算二分类分割 IoU（Jaccard Index）。

    Args:
        pred (torch.Tensor): 预测掩码张量。
        target (torch.Tensor): 真实掩码张量。
        smooth (float): 平滑项，避免除零。

    Returns:
        float: 批次平均 IoU。
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target).sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    """计算二分类分割 Dice 系数。

    Args:
        pred (torch.Tensor): 预测掩码张量。
        target (torch.Tensor): 真实掩码张量。
        smooth (float): 平滑项，避免除零。

    Returns:
        float: 批次平均 Dice。
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
    return dice.mean().item()


def calculate_map(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], iou_thresholds=None):
    """mAP 计算接口（预留，需要根据具体任务实现）。

    Args:
        pred_masks (List[np.ndarray]): 预测掩码列表。
        gt_masks (List[np.ndarray]): 真实掩码列表。
        iou_thresholds (List[float] | None): IoU 阈值列表。

    Returns:
        dict: mAP 结果字典。
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    print("mAP 计算尚未实现，请根据具体任务实现。")
    return {f'mAP@{t}': 0.0 for t in iou_thresholds}
