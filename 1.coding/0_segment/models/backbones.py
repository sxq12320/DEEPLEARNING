import torch.nn as nn
from configs.config import RESNET_18_BACKBONE_CFG
from .builder import make_layers


class ResNet18(nn.Module):
    """ResNet-18 骨干网络，通过配置列表构建。"""

    def __init__(self, cfg=None):
        """初始化 ResNet-18 骨干。

        Args:
            cfg (list | None): 自定义配置列表，默认使用 RESNET_18_BACKBONE_CFG。
        """
        super().__init__()
        self.backbone = make_layers(cfg if cfg is not None else RESNET_18_BACKBONE_CFG)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: Backbone 输出特征。
        """
        return self.backbone(x)
