# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extended IoU-family losses + NWD for the citrus far-field small-object experiments.

针对痛点：远处柑橘极小（IoU 对小框的微小偏移极度敏感）且为估计标注（低质量框产生有害梯度）。

实现的变体与出处：
- EIoU      Zhang et al. 2022, "Focal and Efficient IOU Loss", doi:10.1016/j.neucom.2022.07.042
- SIoU      Gevorgyan 2022, arXiv:2205.12740
- MPDIoU    Ma et al. 2023, arXiv:2307.07662（此处用最小外接框对角线归一化的自包含变体）
- ShapeIoU  Zhang & Zhang 2023, arXiv:2312.17663
- WIoU v3   Tong et al. 2023, arXiv:2301.10051（动态非单调聚焦，对低质量标注鲁棒）
- Inner-IoU Zhang et al. 2023, arXiv:2311.02877（辅助尺度框加速收敛；ratio<1 关注核心区）
- Focaler   Zhang & Zhang 2024, arXiv:2401.10525（线性区间重映射，聚焦难/易样本）
- NWD       Wang et al. 2021, arXiv:2110.13389（高斯 Wasserstein 距离，对微小目标尺度不敏感）

所有函数输入均为 xyxy 格式、逐元素配对的 (N,4) 张量，返回 (N,1)。
"""

from __future__ import annotations

import math

import torch


def _box_info(box: torch.Tensor, eps: float = 1e-7):
    """Return x1, y1, x2, y2, w, h, cx, cy for an xyxy box tensor."""
    x1, y1, x2, y2 = box.chunk(4, -1)
    w, h = (x2 - x1).clamp_(min=eps), (y2 - y1).clamp_(min=eps)
    return x1, y1, x2, y2, w, h, (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _inner_boxes(box: torch.Tensor, ratio: float, eps: float = 1e-7) -> torch.Tensor:
    """Scale a box around its center by `ratio` (Inner-IoU auxiliary box)."""
    x1, y1, x2, y2, w, h, cx, cy = _box_info(box, eps)
    hw, hh = w * ratio * 0.5, h * ratio * 0.5
    return torch.cat([cx - hw, cy - hh, cx + hw, cy + hh], -1)


def _plain_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Plain pairwise IoU for xyxy boxes, shape (N,1)."""
    b1x1, b1y1, b1x2, b1y2, w1, h1, _, _ = _box_info(box1, eps)
    b2x1, b2y1, b2x2, b2y2, w2, h2, _, _ = _box_info(box2, eps)
    inter = (b1x2.minimum(b2x2) - b1x1.maximum(b2x1)).clamp_(0) * (
        b1y2.minimum(b2y2) - b1y1.maximum(b2y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter / union


def bbox_iou_ext(
    box1: torch.Tensor,
    box2: torch.Tensor,
    iou_type: str = "ciou",
    inner_ratio: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Extended IoU family (xyxy, elementwise-paired). Returns IoU-like similarity in (N,1).

    iou_type: iou / giou / diou / ciou / eiou / siou / mpdiou / shapeiou（小写）。
    inner_ratio != 1.0 时，IoU 项用 Inner-IoU 辅助框计算（惩罚项仍用原框几何）。
    """
    t = iou_type.lower()
    b1x1, b1y1, b1x2, b1y2, w1, h1, cx1, cy1 = _box_info(box1, eps)
    b2x1, b2y1, b2x2, b2y2, w2, h2, cx2, cy2 = _box_info(box2, eps)

    if inner_ratio != 1.0:
        iou = _plain_iou(_inner_boxes(box1, inner_ratio, eps), _inner_boxes(box2, inner_ratio, eps), eps)
    else:
        iou = _plain_iou(box1, box2, eps)

    if t == "iou":
        return iou

    cw = b1x2.maximum(b2x2) - b1x1.minimum(b2x1)  # enclosing width
    ch = b1y2.maximum(b2y2) - b1y1.minimum(b2y1)  # enclosing height

    if t == "giou":
        c_area = cw * ch + eps
        inter = (b1x2.minimum(b2x2) - b1x1.maximum(b2x1)).clamp_(0) * (
            b1y2.minimum(b2y2) - b1y1.maximum(b2y1)
        ).clamp_(0)
        union = w1 * h1 + w2 * h2 - inter + eps
        return iou - (c_area - union) / c_area

    c2 = cw.pow(2) + ch.pow(2) + eps  # enclosing diagonal squared
    rho2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2)  # center distance squared

    if t == "diou":
        return iou - rho2 / c2
    if t == "ciou":
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    if t == "eiou":
        return iou - rho2 / c2 - (w1 - w2).pow(2) / (cw.pow(2) + eps) - (h1 - h2).pow(2) / (ch.pow(2) + eps)
    if t == "siou":
        # angle cost
        sigma = rho2.clamp(min=eps).sqrt()
        sin_a = ((cy2 - cy1).abs() / sigma).clamp(-1 + eps, 1 - eps)
        angle = torch.cos(torch.arcsin(sin_a) * 2 - math.pi / 2)
        # distance cost
        gamma = 2 - angle
        dx = ((cx2 - cx1) / (cw + eps)).pow(2)
        dy = ((cy2 - cy1) / (ch + eps)).pow(2)
        dist = (1 - torch.exp(-gamma * dx)) + (1 - torch.exp(-gamma * dy))
        # shape cost
        ww = (w1 - w2).abs() / w1.maximum(w2)
        wh = (h1 - h2).abs() / h1.maximum(h2)
        shape = (1 - torch.exp(-ww)).pow(4) + (1 - torch.exp(-wh)).pow(4)
        return iou - 0.5 * (dist + shape)
    if t == "mpdiou":
        d1 = (b2x1 - b1x1).pow(2) + (b2y1 - b1y1).pow(2)
        d2 = (b2x2 - b1x2).pow(2) + (b2y2 - b1y2).pow(2)
        return iou - (d1 + d2) / c2  # 用外接框对角线归一化（自包含，不依赖图像尺寸）
    if t == "shapeiou":
        scale = 0.0
        ww = 2 * w2.pow(scale + 1) / (w2.pow(scale + 1) + h2.pow(scale + 1))
        hh = 2 * h2.pow(scale + 1) / (w2.pow(scale + 1) + h2.pow(scale + 1))
        dist_shape = hh * (cx2 - cx1).pow(2) / c2 + ww * (cy2 - cy1).pow(2) / c2
        omega_w = hh * (w1 - w2).abs() / w1.maximum(w2)
        omega_h = ww * (h1 - h2).abs() / h1.maximum(h2)
        shape_cost = (1 - torch.exp(-omega_w)).pow(4) + (1 - torch.exp(-omega_h)).pow(4)
        return iou - dist_shape - 0.5 * shape_cost
    raise ValueError(f"Unknown iou_type '{iou_type}'")


def wiou_terms(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7):
    """Wise-IoU 基础项：返回 (iou, R_WIoU)。R = exp(rho2 / c2*)，c2* 为分离计算图的外接对角线²."""
    iou = _plain_iou(box1, box2, eps)
    _, _, _, _, _, _, cx1, cy1 = _box_info(box1, eps)
    b2x1, b2y1, b2x2, b2y2, _, _, cx2, cy2 = _box_info(box2, eps)
    b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
    cw = b1x2.maximum(b2x2) - b1x1.minimum(b2x1)
    ch = b1y2.maximum(b2y2) - b1y1.minimum(b2y1)
    c2 = (cw.pow(2) + ch.pow(2) + eps).detach()  # detach per WIoU paper (梯度截断防发散)
    rho2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2)
    return iou, torch.exp(rho2 / c2)


def nwd(box1: torch.Tensor, box2: torch.Tensor, constant: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """Normalized Gaussian Wasserstein Distance similarity, in (0, 1], shape (N,1).

    把框建模为二维高斯，计算 2-Wasserstein 距离后指数归一化；对极小目标的像素级偏移
    不敏感（IoU 在 <16px 目标上会因 1-2px 偏移剧烈跳变，NWD 平滑得多）。
    Reference: Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
    Detection" (2021), arXiv:2110.13389.
    """
    _, _, _, _, w1, h1, cx1, cy1 = _box_info(box1, eps)
    _, _, _, _, w2, h2, cx2, cy2 = _box_info(box2, eps)
    w2d = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2) + ((w1 - w2).pow(2) + (h1 - h2).pow(2)) / 4
    return torch.exp(-w2d.clamp(min=0).sqrt() / constant)


def focaler_remap(iou: torch.Tensor, d: float = 0.0, u: float = 0.95) -> torch.Tensor:
    """Focaler-IoU 线性区间重映射：IoU^focaler = clamp((IoU - d) / (u - d), 0, 1)."""
    return ((iou - d) / (u - d)).clamp(0.0, 1.0)
