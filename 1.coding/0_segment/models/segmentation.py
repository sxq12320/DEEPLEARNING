import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import RESNET_18_BACKBONE_CFG
from .builder import make_layers


class MiniSegNet(nn.Module):
    """最小化分割网络，ResNet-18 backbone + 1x1 卷积 head。"""

    def __init__(self, in_ch=3, out_ch=1, backbone_cfg=None):
        """初始化分割网络。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            backbone_cfg (list | None): Backbone 配置列表。
        """
        super().__init__()
        cfg = backbone_cfg if backbone_cfg is not None else RESNET_18_BACKBONE_CFG
        self.backbone = make_layers(cfg)
        # 找到 backbone 最后一个特征图的通道数
        self.head = nn.Conv2d(512, out_ch, kernel_size=1)

    def forward(self, x):
        """前向传播并上采样到输入尺寸。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 输出 logits，大小与输入一致。
        """
        feat = self.backbone(x)
        logits = self.head(feat)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits
