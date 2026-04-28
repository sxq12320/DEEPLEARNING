import torch.nn as nn
import torch
import torch.nn.functional as F
from configs.config import ACTIVATION_MAP
from utils.common import get_activation, autopad
from .registry import register_block


############################################################
#               基本卷积相关的模块                            #
############################################################
@register_block('maxpool')
class MaxPool(nn.Module):
    def __init__(self, k: int, s: int, p: int, d: int):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.MaxPool2d(kernel_size=k, stride=s, padding=p, dilation=d)
        )

    def forward(self, x):
        return self.forward_basic(x)


@register_block('adaptive_avg_pool')
class AdaptiveAvgPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=output_size)
        )

    def forward(self, x):
        return self.forward_basic(x)


@register_block('conv')
class Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, d: int, g: int, b: bool):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k,
                      stride=s, padding=p, dilation=d, groups=g, bias=b)
        )

    def forward(self, x):
        return self.forward_basic(x)


@register_block('basic_conv_block')
class Basic_Conv_Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, d: int, g: int, activation: str):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k,
                      stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(out_ch),
        )
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x):
        return self.act(self.forward_basic(x))


@register_block('conv_block_nonb')
class Conv_Block_NONB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int, d: int, g: int, activation: str):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k,
                      stride=s, padding=p, dilation=d, groups=g)
        )
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x):
        return self.act(self.forward_basic(x))


@register_block('depthwise_conv')
class DepthWise_Conv(nn.Module):
    def __init__(self, in_ch: int, k: int, s: int, p: int, d: int):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k,
                      stride=s, padding=p, dilation=d, groups=in_ch, bias=False)
        )

    def forward(self, x):
        return self.forward_basic(x)


@register_block('pointwise_conv')
class PointWise_Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s: int):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                      stride=s, padding=0, bias=False)
        )

    def forward(self, x):
        return self.forward_basic(x)


@register_block('depthwise_separable_conv')
class DepthWiseSeparable_Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, p: int, s_D: int, s_P: int, d_D: int, activation: str):
        super().__init__()
        self.forward_basic = nn.Sequential(
            DepthWise_Conv(in_ch=in_ch, k=k, s=s_D, p=p, d=d_D),
            nn.BatchNorm2d(in_ch),
            get_activation(activation, ACTIVATION_MAP),
            PointWise_Conv(in_ch=in_ch, out_ch=out_ch, s=s_P),
            nn.BatchNorm2d(out_ch),
            get_activation(activation, ACTIVATION_MAP)
        )

    def forward(self, x):
        return self.forward_basic(x)


@register_block('resnet_block_34')
class ResNetBlock_34(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s: int, activation_1='relu', activation_2='relu'):
        super().__init__()
        self.forward_basic_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, stride=s,
                      padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_1, ACTIVATION_MAP),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, stride=1,
                      padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = None
        if s != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, stride=s,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = get_activation(activation_2, ACTIVATION_MAP)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.forward_basic_1(x)
        return self.act(out + identity)


@register_block('resnet_block_50')
class ResNetBlock_50(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s: int,
                 activation_1='relu', activation_2='relu', activation_3='relu', expansion_size=4):
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_1, ACTIVATION_MAP),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                      stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_2, ACTIVATION_MAP),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch * expansion_size,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch * expansion_size),
            get_activation(activation_3, ACTIVATION_MAP)
        )
        self.downsample = None
        if s != 1 or in_ch != out_ch * expansion_size:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch * expansion_size,
                          kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_ch * expansion_size)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return self.forward_basic(x) + identity


############################################################
#               注意力机制相关的模块                          #
############################################################
@register_block('cbam_channel_attention')
class CBAM_Channel_Attention(nn.Module):
    def __init__(self, in_ch: int, reduction_ratio: int, activation: str):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.MLP_shared = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=in_ch // reduction_ratio, bias=False),
            get_activation(activation, ACTIVATION_MAP),
            nn.Linear(in_features=in_ch // reduction_ratio, out_features=in_ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_weight = self.MLP_shared(self.avgpool(x).view(x.size(0), x.size(1)))
        max_weight = self.MLP_shared(self.maxpool(x).view(x.size(0), x.size(1)))
        attn = self.sigmoid(avg_weight + max_weight).view(x.size(0), x.size(1), 1, 1)
        return attn * x


@register_block('cbam_spatial_attention')
class CBAM_Spatial_Attention(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_weight = torch.mean(x, dim=1, keepdim=True)
        max_weight, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_weight, max_weight], dim=1))
        return self.sigmoid(out) * x


@register_block('cbam')
class CBAM(nn.Module):
    def __init__(self, in_ch: int, reduction_ratio: int, activation: str, k: int):
        super().__init__()
        self.channel_attention = CBAM_Channel_Attention(in_ch=in_ch, reduction_ratio=reduction_ratio, activation=activation)
        self.spatial_attention = CBAM_Spatial_Attention(k=k)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


@register_block('flatten')
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_basic = nn.Sequential(nn.Flatten())

    def forward(self, x):
        return self.forward_basic(x)


@register_block('linear')
class Linear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int, bias: bool):
        super().__init__()
        self.forwar_basic = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=out_feature, bias=bias)
        )

    def forward(self, x):
        return self.forwar_basic(x)
