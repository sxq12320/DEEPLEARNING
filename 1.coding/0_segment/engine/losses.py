"""分割任务损失函数模块。

包含用于分割任务的主流损失函数的封装和计算。
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.fft


class SegmentationLoss(nn.Module):
    """分割任务损失函数包装器，默认使用 BCEWithLogitsLoss。

    该类根据损失类型初始化不同的损失进行计算。

    Attributes:
        loss_type (str): 损失类型（"bce" 或 "cross_entropy"）。
        **kwargs: 传给具体损失函数的参数。
    """

    def __init__(self, loss_type="bce", **kwargs):
        """初始化损失函数模块。"""
        super().__init__()
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred, target):
        """计算损失。

        Args:
            pred (torch.Tensor): 预测 logits。
            target (torch.Tensor): 目标掩码。

        Returns:
            torch.Tensor: 损失值。
        """
        return self.criterion(pred, target)
    