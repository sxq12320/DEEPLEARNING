import torch.nn as nn
from configs.config import RESNET_18_BACKBONE_CFG
from .builder import make_layers


class ResNet18(nn.Module):
    """ResNet-18 骨干网络，通过 make_layers 从配置构建。"""

    def __init__(self, cfg=None):
        super().__init__()
        self.backbone = make_layers(cfg if cfg is not None else RESNET_18_BACKBONE_CFG)

    def forward(self, x):
        return self.backbone(x)
