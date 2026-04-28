import torch.nn as nn


class SegmentationLoss(nn.Module):
    """分割任务损失函数包装器，默认使用 BCEWithLogitsLoss。"""

    def __init__(self, loss_type="bce", **kwargs):
        """初始化损失函数。

        Args:
            loss_type (str): 损失类型（"bce" 或 "cross_entropy"）。
            **kwargs: 传给具体损失函数的参数。

        Raises:
            ValueError: 不支持的损失类型时抛出。
        """
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
