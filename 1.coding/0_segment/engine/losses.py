import torch.nn as nn


class SegmentationLoss(nn.Module):
    """分割任务损失函数包装器，默认使用 BCEWithLogitsLoss。"""

    def __init__(self, loss_type="bce", **kwargs):
        super().__init__()
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred, target):
        return self.criterion(pred, target)
