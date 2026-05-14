"""模型基础模块封装库。

包含常用卷积、注意力、金字塔与 TS-Dual 相关的基础模块。
所有可配置模块使用注册表进行统一管理。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import ACTIVATION_MAP
from utils.common import autopad, get_activation

from .registry import register_block


############################################################
#               基本卷积相关的模块                          #
############################################################
@register_block("maxpool")
class MaxPool(nn.Module):
    """最大池化层封装。

    提供了一个简单的最大池化层封装,继承自 nn.Module。

    Attributes:
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
    """

    def __init__(self, k: int, s: int, p: int, d: int):
        """初始化最大池化层封装类。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.MaxPool2d(kernel_size=k, stride=s, padding=p, dilation=d)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 池化后的输出张量。
        """
        return self.forward_basic(x)


@register_block("adaptive_max_pool")
class AdaptiveMaxPool(nn.Module):
    """自适应最大池化层封装。

    提供自适应大小的最大池化功能,可动态输出指定尺寸特征图。

    Attributes:
        output_size (int | Tuple[int, int]): 输出尺寸。
    """

    def __init__(self, output_size: int):
        """初始化自适应最大池化层。"""
        super(AdaptiveMaxPool, self).__init__()
        self.forward_basic = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=output_size)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 池化后的输出张量。
        """
        return self.forward_basic(x)


@register_block("adaptive_avg_pool")
class AdaptiveAvgPool(nn.Module):
    """自适应平均池化层封装。

    提供自适应大小的平均池化功能。

    Attributes:
        output_size (int | Tuple[int, int]): 输出尺寸。
    """

    def __init__(self, output_size):
        """初始化自适应平均池化层。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=output_size)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 池化后的输出张量。
        """
        return self.forward_basic(x)


@register_block("conv")
class Conv(nn.Module):
    """基础卷积层封装。

    继承 nn.Module 的二维卷积封装模块。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
        g (int): 分组数。
        b (bool): 是否使用偏置。
    """

    def __init__(
        self, in_ch: int, out_ch: int, k: int, s: int, p: int, d: int, g: int, b: bool
    ):
        """初始化卷积层封装类。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=g,
                bias=b,
            )
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.forward_basic(x)


@register_block("basic_conv_block")
class Basic_Conv_Block(nn.Module):
    """卷积 + BN + 激活的基础模块。

    包含二维卷积层、批量归一化层和可指定的激活函数。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
        g (int): 分组数。
        activation (str): 激活函数名称。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        s: int,
        p: int,
        d: int,
        g: int,
        activation: str,
    ):
        """初始化基础卷积模块。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=g,
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 激活后的输出。
        """
        return self.act(self.forward_basic(x))


@register_block("conv_block_nonb")
class Conv_Block_NONB(nn.Module):
    """不含 BN 的卷积 + 激活模块。

    包含二维卷积层和可指定的激活函数。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
        g (int): 分组数。
        activation (str): 激活函数名称。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        s: int,
        p: int,
        d: int,
        g: int,
        activation: str,
    ):
        """初始化不含 BN 的卷积模块。

        Returns:
            None
        """
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=g,
            )
        )
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 激活后的输出。
        """
        return self.act(self.forward_basic(x))


@register_block("depthwise_conv")
class DepthWise_Conv(nn.Module):
    """Depthwise 卷积模块。"""

    def __init__(self, in_ch: int, k: int, s: int, p: int, d: int):
        """初始化 Depthwise 卷积。

        Args:
            in_ch (int): 输入通道数。
            k (int): 卷积核大小。
            s (int): 步幅。
            p (int): 填充。
            d (int): 空洞率。
        """
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=in_ch,
                bias=False,
            )
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.forward_basic(x)


@register_block("pointwise_conv")
class PointWise_Conv(nn.Module):
    """Pointwise (1x1) 卷积模块。"""

    def __init__(self, in_ch: int, out_ch: int, s: int):
        """初始化 Pointwise 卷积。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            s (int): 步幅。
        """
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=s,
                padding=0,
                bias=False,
            )
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.forward_basic(x)


@register_block("depthwise_separable_conv")
class DepthWiseSeparable_Conv(nn.Module):
    """Depthwise + Pointwise 的可分离卷积模块。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        p: int,
        s_D: int,
        s_P: int,
        d_D: int,
        activation: str,
    ):
        """初始化可分离卷积模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): Depthwise 卷积核大小。
            p (int): 填充。
            s_D (int): Depthwise 步幅。
            s_P (int): Pointwise 步幅。
            d_D (int): Depthwise 空洞率。
            activation (str): 激活函数名称。
        """
        super().__init__()
        self.forward_basic = nn.Sequential(
            DepthWise_Conv(in_ch=in_ch, k=k, s=s_D, p=p, d=d_D),
            nn.BatchNorm2d(in_ch),
            get_activation(activation, ACTIVATION_MAP),
            PointWise_Conv(in_ch=in_ch, out_ch=out_ch, s=s_P),
            nn.BatchNorm2d(out_ch),
            get_activation(activation, ACTIVATION_MAP),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.forward_basic(x)


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


@register_block("resnet_block_34")
class ResNetBlock_34(nn.Module):
    """ResNet-34 基本残差块(双 3x3 卷积)。"""

    def __init__(
        self, in_ch: int, out_ch: int, s: int, activation_1="relu", activation_2="relu"
    ):
        """初始化 ResNet-34 残差块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            s (int): 步幅。
            activation_1 (str): 第一层激活函数。
            activation_2 (str): 残差融合后的激活函数。
        """
        super().__init__()
        self.forward_basic_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                stride=s,
                padding=1,
                kernel_size=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_1, ACTIVATION_MAP),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                stride=1,
                padding=1,
                kernel_size=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = None
        if s != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=s,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )
        self.act = get_activation(activation_2, ACTIVATION_MAP)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 残差输出张量。
        """
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.forward_basic_1(x)
        return self.act(out + identity)


@register_block("resnet_block_50")
class ResNetBlock_50(nn.Module):
    """ResNet-50 瓶颈残差块。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        s: int,
        activation_1="relu",
        activation_2="relu",
        activation_3="relu",
        expansion_size=4,
    ):
        """初始化 ResNet-50 瓶颈残差块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 中间通道数。
            s (int): 步幅。
            activation_1 (str): 第一层激活函数。
            activation_2 (str): 第二层激活函数。
            activation_3 (str): 第三层激活函数。
            expansion_size (int): 通道扩展倍数。
        """
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_1, ACTIVATION_MAP),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=s,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_2, ACTIVATION_MAP),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch * expansion_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch * expansion_size),
            get_activation(activation_3, ACTIVATION_MAP),
        )
        self.downsample = None
        if s != 1 or in_ch != out_ch * expansion_size:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch * expansion_size,
                    kernel_size=1,
                    stride=s,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch * expansion_size),
            )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 残差输出张量。
        """
        identity = self.downsample(x) if self.downsample is not None else x
        return self.forward_basic(x) + identity


############################################################
#               注意力机制相关的模块                        #
############################################################
@register_block("cbam_channel_attention")
class CBAM_Channel_Attention(nn.Module):
    """CBAM 通道注意力模块。"""

    def __init__(self, in_ch: int, reduction_ratio: int, activation: str):
        """初始化通道注意力模块。

        Args:
            in_ch (int): 输入通道数。
            reduction_ratio (int): 通道压缩比。
            activation (str): 激活函数名称。
        """
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.MLP_shared = nn.Sequential(
            nn.Linear(
                in_features=in_ch, out_features=in_ch // reduction_ratio, bias=False
            ),
            get_activation(activation, ACTIVATION_MAP),
            nn.Linear(
                in_features=in_ch // reduction_ratio, out_features=in_ch, bias=False
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 加权后的输出特征图。
        """
        avg_weight = self.MLP_shared(self.avgpool(x).view(x.size(0), x.size(1)))
        max_weight = self.MLP_shared(self.maxpool(x).view(x.size(0), x.size(1)))
        attn = self.sigmoid(avg_weight + max_weight).view(x.size(0), x.size(1), 1, 1)
        return attn * x


@register_block("cbam_spatial_attention")
class CBAM_Spatial_Attention(nn.Module):
    """CBAM 空间注意力模块。"""

    def __init__(self, k: int):
        """初始化空间注意力模块。

        Args:
            k (int): 卷积核大小。
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=k, padding=k // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 加权后的输出特征图。
        """
        avg_weight = torch.mean(x, dim=1, keepdim=True)
        max_weight, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_weight, max_weight], dim=1))
        return self.sigmoid(out) * x


@register_block("cbam")
class CBAM(nn.Module):
    """CBAM 通道 + 空间注意力组合模块。"""

    def __init__(self, in_ch: int, reduction_ratio: int, activation: str, k: int):
        """初始化 CBAM 模块。

        Args:
            in_ch (int): 输入通道数。
            reduction_ratio (int): 通道压缩比。
            activation (str): 激活函数名称。
            k (int): 空间注意力卷积核大小。
        """
        super().__init__()
        self.channel_attention = CBAM_Channel_Attention(
            in_ch=in_ch, reduction_ratio=reduction_ratio, activation=activation
        )
        self.spatial_attention = CBAM_Spatial_Attention(k=k)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 加权后的输出特征图。
        """
        return self.spatial_attention(self.channel_attention(x))


############################################################
#               特征金字塔网络 (FPN) 融合模块                #
############################################################
@register_block("fpn_lateral_conv")
class FPNLateralConv(nn.Module):
    """FPN 侧向 1×1 卷积,将 backbone 各级特征通道数统一到 FPN 输出通道。

    侧向连接不改变空间分辨率,仅做通道对齐,
    使 top-down pathway 中的逐元素加法可行。
    """

    def __init__(self, in_ch: int, out_ch: int):
        """初始化侧向卷积。

        Args:
            in_ch (int): backbone 该级特征的输入通道数。
            out_ch (int): FPN 统一输出通道数。
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): backbone 单级特征图 (B, in_ch, H, W)。

        Returns:
            torch.Tensor: 通道对齐后的特征图 (B, out_ch, H, W)。
        """
        return self.conv(x)


@register_block("fpn_output_conv")
class FPNOutputConv(nn.Module):
    """FPN 输出 3×3 卷积,消除上采样 + 逐元素加法带来的混叠伪影。

    在 top-down pathway 融合完成后施加,平滑最终的多尺度输出,
    作用是稳定训练并提升小目标分割精度。
    """

    def __init__(self, ch: int):
        """初始化输出平滑卷积。

        Args:
            ch (int): 输入/输出通道数(保持等通道)。
        """
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 融合后的特征图 (B, ch, H, W)。

        Returns:
            torch.Tensor: 平滑后的特征图 (B, ch, H, W)。
        """
        return self.conv(x)


class FPN(nn.Module):
    """特征金字塔网络(Feature Pyramid Network),自顶向下融合多尺度特征。

    输入 backbone 的四个阶段特征 [c2, c3, c4, c5],
    通过 top-down pathway + lateral connection 融合,
    输出统一通道数的多尺度特征 [p2, p3, p4, p5]。

    典型用法：
        backbone = MultiScaleResNet18()
        fpn = FPN(in_channels_list=[64, 128, 256, 512], out_channels=256)
        features = backbone(x)          # [c2,c3,c4,c5]
        fpn_feats = fpn(features)       # [p2,p3,p4,p5]
    """

    def __init__(self, in_channels_list, out_channels=256):
        """初始化 FPN。

        Args:
            in_channels_list (List[int]): backbone 各级特征通道数,自底向上排列。
            out_channels (int): FPN 输出特征的统一通道数。
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
                for in_ch in in_channels_list
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features):
        """前向传播,自顶向下融合。

        Args:
            features (List[torch.Tensor]): backbone 各阶段特征图,
                                           自底向上排列 [c2, c3, c4, c5]。

        Returns:
            List[torch.Tensor]: FPN 融合后的多尺度特征 [p2, p3, p4, p5],
                                各层通道数 = out_channels。
        """
        # 侧向 1×1 卷积对齐通道
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # 自顶向下路径：从最高层 c5 开始,逐步上采样并融合
        fused = []
        prev = laterals[-1]  # p5 ← lateral(c5)
        fused.append(prev)

        for i in range(len(laterals) - 2, -1, -1):
            up = nn.functional.interpolate(
                prev, size=laterals[i].shape[2:], mode="nearest"
            )
            prev = laterals[i] + up
            fused.insert(0, prev)

        # 3×3 卷积平滑融合结果
        outputs = [conv(f) for conv, f in zip(self.output_convs, fused)]
        return outputs


@register_block("flatten")
class Flatten(nn.Module):
    """展平层封装。"""

    def __init__(self):
        """初始化展平层。"""
        super().__init__()
        self.forward_basic = nn.Sequential(nn.Flatten())

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 展平后的张量。
        """
        return self.forward_basic(x)


@register_block("linear")
class Linear(nn.Module):
    """线性层封装。"""

    def __init__(self, in_feature: int, out_feature: int, bias: bool):
        """初始化线性层。

        Args:
            in_feature (int): 输入特征数。
            out_feature (int): 输出特征数。
            bias (bool): 是否使用偏置。
        """
        super().__init__()
        self.forwar_basic = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=out_feature, bias=bias)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.forwar_basic(x)


@register_block("c3k2")
class C3k2(nn.Module):
    """C3k2 模块 (YOLO11 架构衍生),即自定义跨阶段局部网络(CSP)瓶颈层构建块。

    C3k2 是 YOLO 模型中常用的基于 CSP 的高级特征提取块。它通过一条主分支
    执行连续的瓶颈层操作,并由一个额外捷径(shortcut)通道将特征组合起来,
    能在不大幅度增加算力的同时,较好地保留并整合多尺度、多层级语义细节。

    Attributes:
        in_ch (int): 输入通道数量。
        out_ch (int): 输出通道数量。
        n (int): 内部 Bottleneck 重复次数。
        shortcut (bool): 内部 Bottleneck 是否使用残差连接。
        g (int): 分组卷积组数。
        e (float): 通道扩展比例(压缩率)。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化 C3k2 类。"""
        super().__init__()
        hid_c = int(out_ch * e)  # 中间通道数
        self.conv1 = nn.Conv2d(
            in_ch, 2 * hid_c, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(2 * hid_c)
        self.act1 = nn.SiLU(inplace=True)

        # 构建内部多级联串联的块 (Bottleneck机制)
        # 此处给出最为经典通用的 3x3 + 3x3 串联表达形式,模拟 YOLO 系列中的 Bottleneck
        self.m = nn.ModuleList()
        for _ in range(n):
            bottle_neck = nn.Sequential(
                nn.Conv2d(
                    hid_c,
                    hid_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=g,
                    bias=False,
                ),
                nn.BatchNorm2d(hid_c),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    hid_c,
                    hid_c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=g,
                    bias=False,
                ),
                nn.BatchNorm2d(hid_c),
                nn.SiLU(inplace=True),
            )
            # 是否含有 shortcut 根据用户设置可额外拓展,简化为包含在结构中
            self.m.append(bottle_neck)

        self.conv2 = nn.Conv2d(
            (2 + n) * hid_c, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 第一层卷积
        y = self.act1(self.bn1(self.conv1(x)))
        # 把经过首层卷积提炼后的特征对半拆分(实现局部跨阶段 CSP 效果)
        y1, y2 = torch.chunk(y, 2, dim=1)

        # 收集经过每一层 bottle_neck 萃取后的残差分片
        out = [y1, y2]
        for m in self.m:
            out.append(m(out[-1]))

        # 将所有的分片拼接并由 final conv 统一还原到 out_ch 通道数
        return self.act2(self.bn2(self.conv2(torch.cat(out, dim=1))))


############################################################
#               YOLO 系列专用模块                          #
############################################################
@register_block("bottleneck")
class Bottleneck(nn.Module):
    """YOLO 标准瓶颈模块，由两个 3×3 卷积串联并附加残差连接。

    当 shortcut=True 时，模块的残差边会将输入与经两个卷积变换后的
    特征执行逐元素相加操作，由此辅助梯度流动与特征复用。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        shortcut (bool): 是否使用残差连接。
        g (int): 分组卷积组数。
        e (float): 中间通道相对于输出通道的扩展比。
    """

    def __init__(
        self, in_ch: int, out_ch: int, shortcut: bool = True, g: int = 1, e: float = 0.5
    ):
        """初始化 Bottleneck 模块。"""
        super().__init__()
        hid_ch = int(out_ch * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(
                in_ch, hid_ch, kernel_size=3, stride=1, padding=1, groups=g, bias=False
            ),
            nn.BatchNorm2d(hid_ch),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(
                hid_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=g, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量 (B, in_ch, H, W)。

        Returns:
            torch.Tensor: 输出张量 (B, out_ch, H, W)。
        """
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


@register_block("sppf")
class SPPF(nn.Module):
    """空间金字塔池化快速版(Spatial Pyramid Pooling Fast)。

    通过连续三次同样的最大池化操作代替并行多尺度池化，
    等价于融合 5×5、9×9、13×13 的感受野，增强多尺度目标的特征表达。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 连续池化的卷积核大小。
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        """初始化 SPPF 模块。"""
        super().__init__()
        hid_ch = in_ch // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hid_ch),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(hid_ch * 4, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量 (B, in_ch, H, W)。

        Returns:
            torch.Tensor: 多尺度融合后的输出 (B, out_ch, H, W)。
        """
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
