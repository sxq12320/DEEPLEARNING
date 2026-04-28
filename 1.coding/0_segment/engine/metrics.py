from typing import List

import numpy as np
import torch


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    """计算二分类分割 IoU (Jaccard Index)。"""
    if pred.ndim == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target).sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> float:
    """计算二分类分割 Dice 系数。"""
    if pred.ndim == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
    return dice.mean().item()


def calculate_map(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], iou_thresholds=None):
    """mAP 计算接口（预留，需要根据具体任务实现）。"""
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    print("mAP 计算尚未实现，请根据具体任务实现。")
    return {f'mAP@{t}': 0.0 for t in iou_thresholds}
