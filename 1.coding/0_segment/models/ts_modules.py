"""TS-Dual 架构基础模块。

包含多模态双主干、AFPN、DyHead 以及解耦预测头等组件。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import ACTIVATION_MAP
from utils.common import autopad, get_activation

from .arch_registry import register_backbone, register_head, register_neck


class ConvBNAct(nn.Module):
    """卷积 + BN + 激活的基础模块。

    Args:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int | None): 填充大小，None 时自动计算。
        d (int): 空洞率。
        g (int): 分组卷积组数。
        activation (str): 激活函数名称。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        d: int = 1,
        g: int = 1,
        activation: str = "silu",
    ):
        """初始化基础卷积模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 步幅。
            p (int | None): 填充大小，None 时自动计算。
            d (int): 空洞率。
            g (int): 分组卷积组数。
            activation (str): 激活函数名称。
        """
        super().__init__()
        padding = autopad(k, p, d)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=padding,
            dilation=d,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.act(self.bn(self.conv(x)))


class CrossTokenStatsAttention(nn.Module):
    """轻量级跨模态统计注意力。

    使用全局统计量建立 RGB 与 Depth 的双向交互，
    以较低开销模拟跨模态特征交换。

    Args:
        channels (int): 输入通道数。
        reduction (int): 通道压缩比。
        activation (str): 激活函数名称。
    """

    def __init__(self, channels: int, reduction: int = 4, activation: str = "silu"):
        """初始化跨模态统计注意力。

        Args:
            channels (int): 输入通道数。
            reduction (int): 通道压缩比。
            activation (str): 激活函数名称。
        """
        super().__init__()
        hidden = max(1, channels // reduction)
        self.rgb_to_depth = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            get_activation(activation, ACTIVATION_MAP),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.depth_to_rgb = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            get_activation(activation, ACTIVATION_MAP),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行跨模态交换。

        Args:
            rgb_feat (torch.Tensor): RGB 特征 (B, C, H, W)。
            depth_feat (torch.Tensor): Depth 特征 (B, C, H, W)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                交换后的 (rgb_feat, depth_feat)。
        """
        rgb_stat = rgb_feat.mean(dim=(2, 3), keepdim=True)
        depth_stat = depth_feat.mean(dim=(2, 3), keepdim=True)

        rgb_gate = self.depth_to_rgb(depth_stat)
        depth_gate = self.rgb_to_depth(rgb_stat)

        rgb_out = rgb_feat + rgb_gate * depth_feat
        depth_out = depth_feat + depth_gate * rgb_feat
        return rgb_out, depth_out


@register_backbone("ts_dual_backbone")
class TSDualBackbone(nn.Module):
    """TS-Dual 双主干特征提取器。

    输入 RGB + Mask 先验 与 Depth 两路数据，分别提取多尺度特征，
    并通过跨模态统计注意力进行交互后融合输出。

    Args:
        in_ch_rgb (int): RGB 分支输入通道数（RGB+先验=4）。
        in_ch_depth (int): Depth 分支输入通道数（默认 1）。
        channels (list[int]): 各尺度输出通道数 [P2, P3, P4]。
        activation (str): 激活函数名称。
        exchange_reduction (int): 跨模态注意力通道压缩比。
    """

    def __init__(
        self,
        in_ch_rgb: int = 4,
        in_ch_depth: int = 1,
        channels: Optional[List[int]] = None,
        activation: str = "silu",
        exchange_reduction: int = 4,
    ):
        """初始化 TS-Dual 双主干。

        Args:
            in_ch_rgb (int): RGB 分支输入通道数。
            in_ch_depth (int): Depth 分支输入通道数。
            channels (list[int] | None): 输出通道配置。
            activation (str): 激活函数名称。
            exchange_reduction (int): 通道压缩比。
        """
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        c2, c3, c4 = channels

        # RGB 分支
        self.rgb_stem = nn.Sequential(
            ConvBNAct(in_ch_rgb, c2, k=3, s=2, activation=activation),
            ConvBNAct(c2, c2, k=3, s=2, activation=activation),
        )
        self.rgb_stage3 = ConvBNAct(c2, c3, k=3, s=2, activation=activation)
        self.rgb_stage4 = ConvBNAct(c3, c4, k=3, s=2, activation=activation)

        # Depth 分支
        self.depth_stem = nn.Sequential(
            ConvBNAct(in_ch_depth, c2, k=3, s=2, activation=activation),
            ConvBNAct(c2, c2, k=3, s=2, activation=activation),
        )
        self.depth_stage3 = ConvBNAct(c2, c3, k=3, s=2, activation=activation)
        self.depth_stage4 = ConvBNAct(c3, c4, k=3, s=2, activation=activation)

        # 跨模态交互
        self.exchange = nn.ModuleList(
            [
                CrossTokenStatsAttention(c2, exchange_reduction, activation),
                CrossTokenStatsAttention(c3, exchange_reduction, activation),
                CrossTokenStatsAttention(c4, exchange_reduction, activation),
            ]
        )

        # 融合
        self.fusion = nn.ModuleList(
            [
                ConvBNAct(c2 * 2, c2, k=1, s=1, activation=activation),
                ConvBNAct(c3 * 2, c3, k=1, s=1, activation=activation),
                ConvBNAct(c4 * 2, c4, k=1, s=1, activation=activation),
            ]
        )

    def forward(
        self,
        rgb: Union[torch.Tensor, dict],
        prompt: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """前向传播并输出多尺度融合特征。

        Args:
            rgb (torch.Tensor | dict): RGB 输入，或包含 rgb/prompt/depth 的字典。
            prompt (torch.Tensor | None): Mask 先验提示 (B,1,H,W)。
            depth (torch.Tensor | None): Depth 输入 (B,1,H,W)。

        Returns:
            list[torch.Tensor]: 融合后的多尺度特征 [P2, P3, P4]。
        """
        if isinstance(rgb, dict):
            depth = rgb.get("depth")
            prompt = rgb.get("prompt")
            rgb = rgb.get("rgb")

        if rgb is None:
            raise ValueError("RGB 输入不能为空")

        prompt = self._ensure_single_channel(prompt, rgb)
        depth = self._ensure_single_channel(depth, rgb)

        rgb_in = torch.cat([rgb, prompt], dim=1)

        # RGB 分支
        rgb_p2 = self.rgb_stem(rgb_in)
        rgb_p3 = self.rgb_stage3(rgb_p2)
        rgb_p4 = self.rgb_stage4(rgb_p3)

        # Depth 分支
        depth_p2 = self.depth_stem(depth)
        depth_p3 = self.depth_stage3(depth_p2)
        depth_p4 = self.depth_stage4(depth_p3)

        # 跨模态交互
        rgb_p2, depth_p2 = self.exchange[0](rgb_p2, depth_p2)
        rgb_p3, depth_p3 = self.exchange[1](rgb_p3, depth_p3)
        rgb_p4, depth_p4 = self.exchange[2](rgb_p4, depth_p4)

        # 融合
        f2 = self.fusion[0](torch.cat([rgb_p2, depth_p2], dim=1))
        f3 = self.fusion[1](torch.cat([rgb_p3, depth_p3], dim=1))
        f4 = self.fusion[2](torch.cat([rgb_p4, depth_p4], dim=1))

        return [f2, f3, f4]

    def _ensure_single_channel(
        self, x: Optional[torch.Tensor], ref: torch.Tensor
    ) -> torch.Tensor:
        """保证输入为单通道并与参考尺寸对齐。

        Args:
            x (torch.Tensor | None): 输入张量。
            ref (torch.Tensor): 参考张量，用于尺寸对齐。

        Returns:
            torch.Tensor: 处理后的单通道张量。
        """
        if x is None:
            return torch.zeros(
                (ref.size(0), 1, ref.size(2), ref.size(3)),
                device=ref.device,
                dtype=ref.dtype,
            )
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1] != 1:
            x = x[:, :1, ...]
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="nearest")
        return x


@register_neck("afpn_neck")
class AFPNNeck(nn.Module):
    """渐进式特征金字塔 AFPN。

    先融合浅层特征，再逐步引入高层语义，输出单尺度融合特征。

    Args:
        in_channels (list[int]): 输入特征通道数 [P2, P3, P4]。
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
            in_channels (list[int] | None): 输入通道数。
            out_channels (int): 输出通道数。
            activation (str): 激活函数名称。
        """
        super().__init__()
        if in_channels is None:
            in_channels = [32, 64, 128]

        self.lateral = nn.ModuleList(
            [ConvBNAct(ch, out_channels, k=1, s=1, activation=activation) for ch in in_channels]
        )
        self.fuse_l1 = ConvBNAct(out_channels * 2, out_channels, k=3, s=1, activation=activation)
        self.fuse_l2 = ConvBNAct(out_channels * 2, out_channels, k=3, s=1, activation=activation)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播并输出融合特征。

        Args:
            features (list[torch.Tensor]): 多尺度特征 [P2, P3, P4]。

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


class ScaleAwareAttention(nn.Module):
    """尺度感知注意力。

    通过全局统计生成通道权重，强调关键尺度特征。

    Args:
        channels (int): 输入通道数。
        reduction (int): 通道压缩比。
        activation (str): 激活函数名称。
    """

    def __init__(self, channels: int, reduction: int = 4, activation: str = "silu"):
        """初始化尺度注意力。

        Args:
            channels (int): 输入通道数。
            reduction (int): 通道压缩比。
            activation (str): 激活函数名称。
        """
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            get_activation(activation, ACTIVATION_MAP),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x (torch.Tensor): 输入特征。

        Returns:
            torch.Tensor: 加权后的特征。
        """
        weights = self.mlp(self.pool(x))
        return x * weights


class SpatialAwareAttention(nn.Module):
    """空间感知注意力。

    通过空间权重过滤背景噪声。

    Args:
        kernel_size (int): 空间注意力卷积核大小。
    """

    def __init__(self, kernel_size: int = 7):
        """初始化空间注意力。

        Args:
            kernel_size (int): 卷积核大小。
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x (torch.Tensor): 输入特征。

        Returns:
            torch.Tensor: 加权后的特征。
        """
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class TaskAwareAttention(nn.Module):
    """任务感知注意力。

    生成 bbox 与 mask 两条任务分支特征。

    Args:
        channels (int): 输入通道数。
        activation (str): 激活函数名称。
    """

    def __init__(self, channels: int, activation: str = "silu"):
        """初始化任务感知注意力。

        Args:
            channels (int): 输入通道数。
            activation (str): 激活函数名称。
        """
        super().__init__()
        self.shared = ConvBNAct(channels, channels, k=3, s=1, activation=activation)
        self.bbox_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.mask_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """前向传播并生成任务分支。

        Args:
            x (torch.Tensor): 输入特征。

        Returns:
            dict: {"bbox": bbox_feat, "mask": mask_feat}。
        """
        shared = self.shared(x)
        bbox_feat = shared * self.bbox_gate(shared)
        mask_feat = shared * self.mask_gate(shared)
        return {"bbox": bbox_feat, "mask": mask_feat}


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


@register_head("decoupled_segdet_head")
class DecoupledSegDetHead(nn.Module):
    """解耦预测头。

    同时输出归一化边界框预测与分割 mask logits。

    Args:
        in_channels (int): 输入特征通道数。
        mask_out_ch (int): 分割输出通道数。
        activation (str): 激活函数名称。
        bbox_hidden (int): bbox 分支中间通道数。
    """

    def __init__(
        self,
        in_channels: int = 128,
        mask_out_ch: int = 1,
        activation: str = "silu",
        bbox_hidden: int = 128,
    ):
        """初始化解耦预测头。

        Args:
            in_channels (int): 输入通道数。
            mask_out_ch (int): 分割输出通道数。
            activation (str): 激活函数名称。
            bbox_hidden (int): bbox 分支中间通道数。
        """
        super().__init__()
        self.bbox_branch = nn.Sequential(
            ConvBNAct(in_channels, bbox_hidden, k=3, s=1, activation=activation),
            ConvBNAct(bbox_hidden, bbox_hidden, k=3, s=1, activation=activation),
        )
        self.bbox_pool = nn.AdaptiveAvgPool2d(1)
        self.bbox_fc = nn.Linear(bbox_hidden, 4)

        self.mask_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, s=1, activation=activation),
            ConvBNAct(in_channels, in_channels, k=3, s=1, activation=activation),
            nn.Conv2d(in_channels, mask_out_ch, kernel_size=1),
        )

    def forward(
        self,
        features: Union[torch.Tensor, dict],
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            features (torch.Tensor | dict): 输入特征或任务特征字典。
            input_shape (tuple[int, int] | None): 原图尺寸 (H, W)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                bbox_pred (B, 4) 与 mask_logits (B, C, H, W)。
        """
        if isinstance(features, dict):
            bbox_feat = features.get("bbox")
            mask_feat = features.get("mask")
            if bbox_feat is None:
                bbox_feat = mask_feat
            if mask_feat is None:
                mask_feat = bbox_feat
        else:
            bbox_feat = features
            mask_feat = features

        bbox_feat = self.bbox_branch(bbox_feat)
        bbox_vec = self.bbox_pool(bbox_feat).flatten(1)
        bbox_raw = torch.sigmoid(self.bbox_fc(bbox_vec))
        x1 = torch.min(bbox_raw[:, 0], bbox_raw[:, 2])
        y1 = torch.min(bbox_raw[:, 1], bbox_raw[:, 3])
        x2 = torch.max(bbox_raw[:, 0], bbox_raw[:, 2])
        y2 = torch.max(bbox_raw[:, 1], bbox_raw[:, 3])
        bbox_pred = torch.stack([x1, y1, x2, y2], dim=1)

        mask_logits = self.mask_branch(mask_feat)
        if input_shape is not None:
            mask_logits = F.interpolate(
                mask_logits, size=input_shape, mode="bilinear", align_corners=False
            )

        return bbox_pred, mask_logits
