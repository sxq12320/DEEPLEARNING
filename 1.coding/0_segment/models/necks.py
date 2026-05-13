"""目标检测颈部网络模块。

提供 PAN-FPN 等颈部融合结构，用于连接 backbone 与检测 head。
所有 neck 遵循统一接口：接收多尺度特征列表，返回融合后同序特征列表。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import C3k2


class YOLO11Neck(nn.Module):
    """YOLO11 PAN-FPN 颈部网络，自顶向下再自底向上融合三层特征。

    PAN（Path Aggregation Network）在 FPN 基础上增加一条自底向上的
    二次融合路径，将浅层空间细节重新注入深层特征，改善多尺度定位精度。

    数据流：
        P5 ──→ Upsample ──+──→ C3K2 → N4 ──→ Upsample ──+──→ C3K2 → N3
        P4 ───────────────┘                                │
        P3 ───────────────────────────────────────────────┘
                                                          ↓
        N3 ──→ Conv(s=2) ──+──→ C3K2 → N4_out ──→ Conv(s=2) ──+──→ C3K2 → N5_out
        N4 ────────────────┘                                   │
        P5 ────────────────────────────────────────────────────┘

    Attributes:
        channels (List[int]): backbone 输入各层通道数 [c3, c4, c5]。
        depth_scale (float): C3K2 内部重复次数缩放因子。
    """

    def __init__(self, channels=None, depth_scale=1.0):
        """初始化 YOLO11 PAN-FPN 颈部。

        Args:
            channels (List[int] | None): 输入通道列表 [c3, c4, c5]，
                                         默认 nano 规格 [64, 128, 256]。
            depth_scale (float): 深度缩放因子。
        """
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        c3, c4, c5 = channels
        d = max(1, round(1 * depth_scale))

        # ---- 上采样层（top-down） ----
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # p5 → n4 侧向融合
        self.top_down_conv1 = nn.Sequential(
            nn.Conv2d(c5, c4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )
        self.top_down_c3k2_1 = C3k2(in_ch=c4 + c4, out_ch=c4, n=d, shortcut=True, e=0.5)

        # n4 → n3 侧向融合
        self.top_down_conv2 = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.top_down_c3k2_2 = C3k2(in_ch=c3 + c3, out_ch=c3, n=d, shortcut=True, e=0.5)

        # ---- 下采样层（bottom-up） ----
        # n3 → n4 融合
        self.bottom_up_conv1 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )
        self.bottom_up_c3k2_1 = C3k2(
            in_ch=c4 + c4, out_ch=c4, n=d, shortcut=True, e=0.5
        )

        # n4 → n5 融合
        self.bottom_up_conv2 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
        )
        self.bottom_up_c3k2_2 = C3k2(
            in_ch=c5 + c5, out_ch=c5, n=d, shortcut=True, e=0.5
        )

    def forward(self, features):
        """前向传播，完成 top-down + bottom-up 双路径融合。

        Args:
            features (List[torch.Tensor]): backbone 输出的三尺度特征 [P3, P4, P5]。

        Returns:
            List[torch.Tensor]: PAN 融合后的三尺度特征 [N3, N4_out, N5_out]。
        """
        p3, p4, p5 = features

        # ---- top-down ----
        p5_up = self.upsample(self.top_down_conv1(p5))
        n4 = self.top_down_c3k2_1(torch.cat([p4, p5_up], dim=1))

        n4_up = self.upsample(self.top_down_conv2(n4))
        n3 = self.top_down_c3k2_2(torch.cat([p3, n4_up], dim=1))

        # ---- bottom-up ----
        n3_down = self.bottom_up_conv1(n3)
        n4_out = self.bottom_up_c3k2_1(torch.cat([n4, n3_down], dim=1))

        n4_down = self.bottom_up_conv2(n4_out)
        n5_out = self.bottom_up_c3k2_2(torch.cat([p5, n4_down], dim=1))

        return [n3, n4_out, n5_out]
