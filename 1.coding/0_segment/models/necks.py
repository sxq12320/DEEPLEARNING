"""目标检测颈部网络模块。

提供 PAN-FPN 等颈部融合结构，用于连接 backbone 与检测 head。
所有 neck 遵循统一接口：接收多尺度特征列表，返回融合后同序特征列表。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    C3k2,
    ConvBNAct,
    ScaleAwareAttention,
    SpatialAwareAttention,
    TaskAwareAttention,
)
from .registry import register_neck


@register_neck("yolo11_neck")
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


@register_neck("afpn_neck")
class AFPNNeck(nn.Module):
    """渐进式特征金字塔 AFPN。

    先融合浅层特征，再逐步引入高层语义，输出单尺度融合特征。

    Args:
        in_channels (List[int] | None): 输入特征通道数 [P2, P3, P4]。
        out_channels (int): 融合输出通道数。
        activation (str): 激活函数名称。
    """

    def __init__(
        self,
        in_channels: Optional[List[int]] = None,
        out_channels: int = 128,
        activation: str = "silu",
    ):
        """初始化 AFPN。

        Args:
            in_channels (List[int] | None): 输入通道数。
            out_channels (int): 输出通道数。
            activation (str): 激活函数名称。
        """
        super().__init__()
        if in_channels is None:
            in_channels = [32, 64, 128]

        self.lateral = nn.ModuleList(
            [
                ConvBNAct(ch, out_channels, k=1, s=1, activation=activation)
                for ch in in_channels
            ]
        )
        self.fuse_l1 = ConvBNAct(
            out_channels * 2, out_channels, k=3, s=1, activation=activation
        )
        self.fuse_l2 = ConvBNAct(
            out_channels * 2, out_channels, k=3, s=1, activation=activation
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播并输出融合特征。

        Args:
            features (List[torch.Tensor]): 多尺度特征 [P2, P3, P4]。

        Returns:
            torch.Tensor: 融合后的特征图 (B, C, H/4, W/4)。
        """
        if len(features) != 3:
            raise ValueError("AFPN 需要 3 个尺度特征 [P2, P3, P4]")
        p2, p3, p4 = features

        l2 = self.lateral[0](p2)
        l3 = self.lateral[1](p3)
        l4 = self.lateral[2](p4)

        p3_up = F.interpolate(l3, size=l2.shape[2:], mode="nearest")
        f_l1 = self.fuse_l1(torch.cat([l2, p3_up], dim=1))

        p4_up = F.interpolate(l4, size=f_l1.shape[2:], mode="nearest")
        f_l2 = self.fuse_l2(torch.cat([f_l1, p4_up], dim=1))

        return f_l2


@register_neck("dyhead_neck")
class DyHeadNeck(nn.Module):
    """动态聚合 DyHead 颈部。

    依次执行尺度、空间与任务注意力，输出两条任务特征。

    Args:
        channels (int): 输入通道数。
        reduction (int): 通道压缩比。
        activation (str): 激活函数名称。
    """

    def __init__(self, channels: int = 128, reduction: int = 4, activation: str = "silu"):
        """初始化 DyHead 颈部。

        Args:
            channels (int): 输入通道数。
            reduction (int): 通道压缩比。
            activation (str): 激活函数名称。
        """
        super().__init__()
        self.scale_attn = ScaleAwareAttention(channels, reduction, activation)
        self.spatial_attn = SpatialAwareAttention()
        self.task_attn = TaskAwareAttention(channels, activation)

    def forward(self, x: torch.Tensor) -> dict:
        """前向传播并输出任务特征。

        Args:
            x (torch.Tensor): 输入特征。

        Returns:
            dict: {"bbox": bbox_feat, "mask": mask_feat}。
        """
        x = self.scale_attn(x)
        x = self.spatial_attn(x)
        return self.task_attn(x)
