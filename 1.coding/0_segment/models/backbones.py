import torch.nn as nn
from configs.config import (
    RESNET_18_BACKBONE_CFG,
    RESNET_18_STEM_CFG, RESNET_18_STAGE1_CFG, RESNET_18_STAGE2_CFG,
    RESNET_18_STAGE3_CFG, RESNET_18_STAGE4_CFG,
)
from .builder import make_layers
from .blocks import C3k2, SPPF


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


class YOLO11Backbone(nn.Module):
    """YOLO11 骨干网络，输出三层多尺度特征 [P3, P4, P5]。

    YOLO11 的主干结构基于 CSP 思想，通过 C3K2 模块和 SPPF 模块
    分别进行跨阶段特征复用和多尺度池化，兼顾推理效率与检测精度。

    输出：
        P3 — 高分辨率 (H/8)、浅层语义，利于小目标检测。
        P4 — 中等分辨率 (H/16)、中间层语义。
        P5 — 低分辨率 (H/32)、深层语义 + SPPF 多感受野。

    Attributes:
        channels (List[int]): 各阶段输出通道数 [c1, c2, c3, c4, c5]。
        depth_scale (float): C3K2 重复次数缩放因子（nano=0.33, small=0.67, medium=1.0）。
    """

    def __init__(self, channels=None, depth_scale=1.0):
        """初始化 YOLO11 骨干。

        Args:
            channels (List[int] | None): 宽度配置 [c1,c2,c3,c4,c5]，默认 nano 规格。
            depth_scale (float): 深度缩放因子。
        """
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128, 256]
        c1, c2, c3, c4, c5 = channels

        # stem：快速降采样到 1/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # stage 1 → P3 (H/8)
        self.stage1 = C3k2(in_ch=c2, out_ch=c3, n=max(1, round(1 * depth_scale)), shortcut=True, e=0.5)

        # stage 2 → P4 (H/16)
        self.stage2 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
            C3k2(in_ch=c4, out_ch=c4, n=max(1, round(1 * depth_scale)), shortcut=True, e=0.5),
        )

        # stage 3 → P5 (H/32)
        self.stage3 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
            C3k2(in_ch=c5, out_ch=c5, n=max(1, round(1 * depth_scale)), shortcut=True, e=0.5),
            SPPF(in_ch=c5, out_ch=c5, k=5),
        )

    def forward(self, x):
        """前向传播，返回多尺度特征列表供 Neck 融合。

        Args:
            x (torch.Tensor): 输入图像 (B, 3, H, W)。

        Returns:
            List[torch.Tensor]: [P3, P4, P5] 三个尺度的特征图。
        """
        x = self.stem(x)        # (B, c2, H/4, W/4)
        p3 = self.stage1(x)     # (B, c3, H/8, W/8)
        p4 = self.stage2(p3)    # (B, c4, H/16, W/16)
        p5 = self.stage3(p4)    # (B, c5, H/32, W/32)
        return [p3, p4, p5]
