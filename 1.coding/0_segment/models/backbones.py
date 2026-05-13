import torch.nn as nn
from configs.config import (
    RESNET_18_BACKBONE_CFG,
    RESNET_18_STEM_CFG, RESNET_18_STAGE1_CFG, RESNET_18_STAGE2_CFG,
    RESNET_18_STAGE3_CFG, RESNET_18_STAGE4_CFG,
)
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


class MultiScaleResNet18(nn.Module):
    """ResNet-18 多尺度骨干，返回四个阶段的特征图用于 FPN 融合。

    返回特征：
        c2 — (B, 64,  H/4,  W/4)
        c3 — (B, 128, H/8,  W/8)
        c4 — (B, 256, H/16, W/16)
        c5 — (B, 512, H/32, W/32)
    """

    def __init__(self, stem_cfg=None, s1_cfg=None, s2_cfg=None, s3_cfg=None, s4_cfg=None):
        """初始化多尺度 ResNet-18。

        Args:
            stem_cfg (list | None): stem 配置。
            s1_cfg (list | None): stage1 配置。
            s2_cfg (list | None): stage2 配置。
            s3_cfg (list | None): stage3 配置。
            s4_cfg (list | None): stage4 配置。
        """
        super().__init__()
        self.stem = make_layers(stem_cfg if stem_cfg is not None else RESNET_18_STEM_CFG)
        self.stage1 = make_layers(s1_cfg if s1_cfg is not None else RESNET_18_STAGE1_CFG)
        self.stage2 = make_layers(s2_cfg if s2_cfg is not None else RESNET_18_STAGE2_CFG)
        self.stage3 = make_layers(s3_cfg if s3_cfg is not None else RESNET_18_STAGE3_CFG)
        self.stage4 = make_layers(s4_cfg if s4_cfg is not None else RESNET_18_STAGE4_CFG)

    def forward(self, x):
        """前向传播，返回多尺度特征列表。

        Args:
            x (torch.Tensor): 输入图像 (B, 3, H, W)。

        Returns:
            List[torch.Tensor]: [c2, c3, c4, c5] 四个尺度的特征图。
        """
        x = self.stem(x)       # (B, 64,  H/4,  W/4)
        c2 = self.stage1(x)    # (B, 64,  H/4,  W/4)
        c3 = self.stage2(c2)   # (B, 128, H/8,  W/8)
        c4 = self.stage3(c3)   # (B, 256, H/16, W/16)
        c5 = self.stage4(c4)   # (B, 512, H/32, W/32)
        return [c2, c3, c4, c5]
