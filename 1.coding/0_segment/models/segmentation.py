"""基础与复合分割网络模块。

包含诸如简单的 MiniSegNet 及结合了 FPN 结构的 FPNSegNet。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import RESNET_18_BACKBONE_CFG
from .builder import make_layers
from .backbones import MultiScaleResNet18
from .blocks import FPN


class MiniSegNet(nn.Module):
    """最小化分割网络，ResNet-18 backbone + 1x1 卷积 head。

    包含最基本的 BackBone 与 极简预测头的分割网络。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        backbone_cfg (list | None): Backbone 配置列表。
    """

    def __init__(self, in_ch=3, out_ch=1, backbone_cfg=None):
        """初始化最简分割网络类。"""
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


class FPNSegNet(nn.Module):
    """FPN 多尺度分割网络。

    结构：
        MultiScaleResNet18 backbone → FPN neck → 多尺度特征融合 → 分割 head

    Backbone 输出 4 个尺度 [c2, c3, c4, c5]，
    FPN 自顶向下融合为 [p2, p3, p4, p5]，
    全部上采样至 p2 分辨率拼接后经 head 输出分割图。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出类别数。
        backbone_channels (List[int] | None): backbone 各阶段通道数，默认 [64, 128, 256, 512]。
        fpn_channels (int): FPN 统一通道数。
    """

    def __init__(self, in_ch=3, out_ch=1, backbone_channels=None, fpn_channels=256):
        """初始化多尺度分割网络类。"""
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]

        self.backbone = MultiScaleResNet18()
        self.fpn = FPN(in_channels_list=backbone_channels, out_channels=fpn_channels)

        fuse_in_ch = fpn_channels * len(backbone_channels)
        self.head = nn.Sequential(
            nn.Conv2d(fuse_in_ch, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, out_ch, kernel_size=1),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入图像 (B, 3, H, W)。

        Returns:
            torch.Tensor: 分割 logits，大小与输入一致。
        """
        features = self.backbone(x)                    # [c2, c3, c4, c5]
        fpn_feats = self.fpn(features)                 # [p2, p3, p4, p5]

        # 全部上采样到 p2 分辨率并拼接
        h, w = fpn_feats[0].shape[2:]
        fused = torch.cat([
            F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
            for f in fpn_feats
        ], dim=1)

        logits = self.head(fused)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits
