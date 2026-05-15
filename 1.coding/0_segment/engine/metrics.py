"""常用评估指标模块。

在此可根据不同的评估要求，计算 IoU, Dice 等常见分割指标。
"""

from typing import List, Tuple

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


def compute_iou_per_sample(
    pred: torch.Tensor, target: torch.Tensor, smooth=1e-6
) -> torch.Tensor:
    """计算每个样本的二分类 IoU。"""
    if pred.ndim == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target).sum(dim=(1, 2)) - intersection
    return (intersection + smooth) / (union + smooth)


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
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth
    )
    return dice.mean().item()


def compute_box_iou(
    pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps=1e-6
) -> torch.Tensor:
    """计算每个样本的 box IoU（xyxy，归一化坐标）。"""
    if pred_boxes.ndim != 2 or target_boxes.ndim != 2:
        raise ValueError("boxes must be (B, 4)")

    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    ).clamp(min=0)
    area_gt = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    ).clamp(min=0)

    union = area_pred + area_gt - inter
    return inter / (union + eps)


def compute_pr_from_iou(
    iou: torch.Tensor,
    pred_valid: torch.Tensor,
    gt_valid: torch.Tensor,
    thresholds,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 IoU 阈值计算 TP/FP/FN 统计。"""
    if iou.ndim != 1:
        raise ValueError("iou must be a 1D tensor")
    t = torch.tensor(thresholds, device=iou.device, dtype=iou.dtype).view(1, -1)
    iou = iou.view(-1, 1)
    pred_valid = pred_valid.view(-1, 1)
    gt_valid = gt_valid.view(-1, 1)

    tp = ((iou >= t) & pred_valid & gt_valid).sum(dim=0)
    fp = (pred_valid & (~gt_valid | (iou < t))).sum(dim=0)
    fn = (gt_valid & (~pred_valid | (iou < t))).sum(dim=0)
    return tp, fp, fn


def calculate_map(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], iou_thresholds=None
):
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
    return {f"mAP@{t}": 0.0 for t in iou_thresholds}


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP from recall-precision curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iou_thresholds: List[float],
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision/recall/AP per class (Ultralytics-style)."""
    if tp.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros(
            (0, len(iou_thresholds)), dtype=np.float32
        )

    sort_idx = np.argsort(-conf)
    tp = tp[sort_idx]
    conf = conf[sort_idx]
    pred_cls = pred_cls[sort_idx]

    unique_classes = np.unique(target_cls) if target_cls.size else np.array([], dtype=np.int64)
    nc = unique_classes.size
    ap = np.zeros((nc, len(iou_thresholds)), dtype=np.float32)
    p = np.zeros((nc,), dtype=np.float32)
    r = np.zeros((nc,), dtype=np.float32)

    for ci, c in enumerate(unique_classes):
        cls_mask = pred_cls == c
        n_p = int(cls_mask.sum())
        n_l = int((target_cls == c).sum())
        if n_p == 0 or n_l == 0:
            continue

        tpc = tp[cls_mask].cumsum(0)
        fpc = (1.0 - tp[cls_mask]).cumsum(0)

        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc + eps)

        for j in range(len(iou_thresholds)):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

        p_curve = precision[:, 0]
        r_curve = recall[:, 0]
        f1 = 2.0 * p_curve * r_curve / (p_curve + r_curve + eps)
        best = int(f1.argmax())
        p[ci] = p_curve[best]
        r[ci] = r_curve[best]

    return p, r, ap
