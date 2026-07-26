# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Citrus far-field small-object modules (柑橘远距离小目标改进模块库).

针对无锡柑橘幼果数据集的核心痛点设计：远处果实极小（<32px / <16px）、模糊（失焦/分辨率不足导致高频信息弱）、
发黑（欠曝/阴影导致亮度低、对比度低）、标注为估计标注（低质量框）。

模块分组：
1. 下采样改进（防小目标信息丢失）：SPDConv (Sunkara & Luo 2022, arXiv:2208.03641)、
   HWDown (Haar wavelet downsampling, Xu et al. 2023, doi:10.1016/j.patcog.2023.109819)
2. 上采样改进（内容感知，恢复小目标细节）：CARAFE (Wang et al. ICCV 2019, arXiv:1905.02188)、
   DySample (Liu et al. ICCV 2023, arXiv:2308.15085)
3. 注意力（突出弱信号小目标）：EMA (Ouyang et al. ICASSP 2023, arXiv:2305.13563)、
   SimAM (Yang et al. ICML 2021)、CoordAtt (Hou et al. CVPR 2021, arXiv:2103.02907)、
   ELA (Xu & Wan 2024, arXiv:2403.01123)、CAA (Cai et al. CVPR 2024 PKINet, arXiv:2403.06258)
4. 感受野/上下文：RFB (Liu et al. ECCV 2018, arXiv:1711.07767)、SPPF_LSKA (Lau et al. 2024,
   doi:10.1016/j.eswa.2023.121352)、DWR 多尺度膨胀 (Wei et al. 2022, arXiv:2212.01173)
5. 轻量化 block：C3k2_Faster (FasterNet PConv, Chen et al. CVPR 2023, arXiv:2303.03667)、
   C3k2_WT (WTConv 小波大感受野, Finder et al. ECCV 2024, arXiv:2407.05848)
6. 特征融合：BiFPNConcat (Tan et al. CVPR 2020 EfficientDet, arXiv:1911.09070)
7. 原创组合模块（本课题针对性设计）：
   - DFEM  Dual-domain Frequency Enhancement Module — 频域高频增益补偿模糊 + 暗区特征增益补偿欠曝
   - LIAM  Luminance-Invariant Attention Module — 实例归一化亮度对齐 + 无参能量注意力
   - CSFG  Cross-Stage Small-object Feature Guidance — P2 高分辨率细节经无损下采样注入 P3
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .conv import Conv, DWConv
from .lsnet import LSConv

__all__ = (
    "SPDConv",
    "HWDown",
    "CARAFE",
    "DySample",
    "EMA",
    "SimAM",
    "CoordAtt",
    "ELA",
    "CAA",
    "RFB",
    "LSKA",
    "SPPF_LSKA",
    "DWR",
    "C3k2_DWR",
    "PConv",
    "FasterBlock",
    "C3k2_Faster",
    "WTConv",
    "WTBottleneck",
    "C3k2_WT",
    "BiFPNConcat",
    "DFEM",
    "LIAM",
    "CSFG",
    "LRSA",
    "HFBranch",
    "MSDFFFN",
    "EDFFN",
    "FarFormer",
    "LumiFormer",
    "TDAM",
    "LCE",
    "FocusedLinearAttn",
    "MWCA",
    "HCO",
    "HyperACE",
    "LSBottleneck",
    "C3k2_LS",
    "TGP",
    "HSF",
    "PCFA",
    "SXQBottleneck",
    "C3k2_SXQ",
    "MoCEBottleneck",
    "C3k2_MoCE",
    "HyperRes",
    "DyT",
)


# ---------------------------------------------------------------------------------------------------------------------
# Haar 小波工具（正交归一，能量守恒；奇数尺寸自动 pad）
# ---------------------------------------------------------------------------------------------------------------------
def _haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level orthonormal Haar DWT. Returns (LL, LH, HL, HH), each at half resolution."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _haar_idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Inverse of `_haar_dwt`."""
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll + lh - hl - hh) * 0.5
    x10 = (ll - lh + hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    b, c, h, w = ll.shape
    out = ll.new_zeros(b, c, h * 2, w * 2)
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 1. 下采样改进
# ---------------------------------------------------------------------------------------------------------------------
class SPDConv(nn.Module):
    """Space-to-Depth Conv (SPD-Conv):用无损的 space-to-depth 重排替代步长卷积下采样.

    步长卷积/池化会直接丢弃 3/4 的像素，远处 <16px 的柑橘经过两次下采样后特征几乎消失；
    SPD 把 2x2 邻域重排到通道维（信息无损），再用非步长卷积压缩通道。
    Reference: Sunkara & Luo, "No More Strided Convolutions or Pooling" (ECML-PKDD 2022), arXiv:2208.03641.
    """

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
        x = torch.cat([x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)


class HWDown(nn.Module):
    """Haar Wavelet Downsampling (HWD):小波变换下采样，低频保结构、高频保边缘.

    与 SPDConv 同为"信息保持型下采样"，但按频带正交分解：LL 保留整体结构，LH/HL/HH 显式保留
    小目标的边缘高频信息（远处模糊柑橘赖以区分的微弱边缘）。
    Reference: Xu et al., "Haar wavelet downsampling: A simple but effective downsampling module
    for semantic segmentation" (Pattern Recognition 2023), doi:10.1016/j.patcog.2023.109819.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = _haar_dwt(x)
        return self.conv(torch.cat([ll, lh, hl, hh], 1))


# ---------------------------------------------------------------------------------------------------------------------
# 2. 内容感知上采样
# ---------------------------------------------------------------------------------------------------------------------
class CARAFE(nn.Module):
    """CARAFE: Content-Aware ReAssembly of FEatures (轻量重实现).

    最近邻上采样把远处小柑橘的 1 个特征像素复制成 4 份（无新信息）；CARAFE 依据内容预测重组核，
    在上采样时聚合大感受野邻域，小目标边缘更清晰。
    Reference: Wang et al., ICCV 2019, arXiv:1905.02188.
    """

    def __init__(self, c1: int, scale: int = 2, k_enc: int = 3, k_up: int = 5, c_mid: int = 64):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.comp = Conv(c1, c_mid, 1)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_, w_ = h * self.scale, w * self.scale
        w_kernel = self.pix_shf(self.enc(self.comp(x)))  # b, k_up^2, h_, w_
        w_kernel = torch.softmax(w_kernel, dim=1)
        x_up = self.unfold(self.upsmp(x)).view(b, c, -1, h_, w_)  # b, c, k_up^2, h_, w_
        return torch.einsum("bkhw,bckhw->bchw", w_kernel, x_up)


class DySample(nn.Module):
    """DySample: 超轻量动态点采样上采样（'lp' 风格 + dynamic scope）.

    以学习到的亚像素偏移做 grid_sample,代价远低于 CARAFE 而效果相当；对小目标边界恢复友好。
    Reference: Liu et al., "Learning to Upsample by Learning to Sample" (ICCV 2023), arXiv:2308.15085.
    """

    def __init__(self, c1: int, scale: int = 2, groups: int = 4):
        super().__init__()
        self.scale = scale
        self.groups = groups if c1 % groups == 0 else 1
        out_ch = 2 * self.groups * scale**2
        self.offset = nn.Conv2d(c1, out_ch, 1)
        self.scope = nn.Conv2d(c1, out_ch, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        nn.init.zeros_(self.scope.bias)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="xy"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )  # 1, 2, 1, h, w
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        y = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w), coords, mode="bilinear", align_corners=False, padding_mode="border"
        )
        return y.view(b, -1, self.scale * h, self.scale * w)


# ---------------------------------------------------------------------------------------------------------------------
# 3. 注意力模块
# ---------------------------------------------------------------------------------------------------------------------
class EMA(nn.Module):
    """Efficient Multi-scale Attention: 跨空间学习的分组多尺度注意力.

    对与叶片同色、响应微弱的小柑橘，通过 1x1/3x3 双分支跨空间交互放大弱信号通道。
    Reference: Ouyang et al., ICASSP 2023, arXiv:2305.13563.
    """

    def __init__(self, c1: int, factor: int = 8):
        super().__init__()
        self.groups = factor if c1 % factor == 0 else 1
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        cg = c1 // self.groups
        self.gn = nn.GroupNorm(cg, cg)
        self.conv1x1 = nn.Conv2d(cg, cg, 1)
        self.conv3x3 = nn.Conv2d(cg, cg, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(gx)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        cg = c // self.groups
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, cg, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, cg, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)


class SimAM(nn.Module):
    """SimAM: 无参数 3D 能量注意力（不增加任何参数量/几乎不增加延迟）.

    以神经元能量函数突出与邻域差异大的像素——远处小柑橘正是局部低对比的"离群"弱信号，
    能量注意力可放大之；参数为 0，是轻量化约束下的首选。
    Reference: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module" (ICML 2021).
    """

    def __init__(self, c1: int = None, e_lambda: float = 1e-4):  # noqa: ARG002 (c1 kept for YAML parser uniformity)
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e_inv)


class CoordAtt(nn.Module):
    """Coordinate Attention: 把位置信息嵌入通道注意力（H/W 双向条带池化）.

    果园图像中远处小果集中于画面特定条带（树冠远端），坐标注意力能按行/列定位这些区域。
    Reference: Hou et al., CVPR 2021, arXiv:2103.02907.
    """

    def __init__(self, c1: int, reduction: int = 32):
        super().__init__()
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c1, 1)
        self.conv_w = nn.Conv2d(mip, c1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(3, keepdim=True)  # b,c,h,1
        x_w = x.mean(2, keepdim=True).permute(0, 1, 3, 2)  # b,c,w,1
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()
        return x * a_h * a_w


class ELA(nn.Module):
    """Efficient Local Attention: 条带池化 + 分组 1D 卷积的高效局部注意力.

    Reference: Xu & Wan, "ELA: Efficient Local Attention for Deep Convolutional Neural Networks"
    (2024), arXiv:2403.01123.
    """

    def __init__(self, c1: int, k: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        groups = 16 if c1 % 16 == 0 else (4 if c1 % 4 == 0 else 1)
        self.gn = nn.GroupNorm(groups, c1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(3)  # b,c,h
        x_w = x.mean(2)  # b,c,w
        a_h = torch.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
        a_w = torch.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
        return x * a_h * a_w


class CAA(nn.Module):
    """Context Anchor Attention (PKINet): 大条带核捕获远距离上下文的锚点注意力.

    以 1xK/Kx1 深度条带卷积获得大椭圆感受野，用周围枝叶上下文"锚定"远处弱小果实区域。
    Reference: Cai et al., "Poly Kernel Inception Network for Remote Sensing Detection"
    (CVPR 2024), arXiv:2403.06258.
    """

    def __init__(self, c1: int, h_kernel: int = 11, v_kernel: int = 11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(c1, c1, 1)
        self.h_conv = nn.Conv2d(c1, c1, (1, h_kernel), 1, (0, h_kernel // 2), groups=c1)
        self.v_conv = nn.Conv2d(c1, c1, (v_kernel, 1), 1, (v_kernel // 2, 0), groups=c1)
        self.conv2 = Conv(c1, c1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return x * attn


# ---------------------------------------------------------------------------------------------------------------------
# 4. 感受野 / 上下文
# ---------------------------------------------------------------------------------------------------------------------
class RFB(nn.Module):
    """Receptive Field Block (lite): 多膨胀率分支模拟人类视觉离心感受野.

    远处小柑橘需要"小感受野看果、大感受野看枝叶上下文"同时进行；RFB 用并联膨胀卷积实现。
    Reference: Liu et al., "Receptive Field Block Net" (ECCV 2018), arXiv:1711.07767.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_ = max(8, c2 // 4)
        self.b0 = nn.Sequential(Conv(c1, c_, 1), Conv(c_, c_, 3))
        self.b1 = nn.Sequential(Conv(c1, c_, 1), Conv(c_, c_, 3), Conv(c_, c_, 3, d=3))
        self.b2 = nn.Sequential(Conv(c1, c_, 1), Conv(c_, c_, 3), Conv(c_, c_, 3), Conv(c_, c_, 3, d=5))
        self.b3 = Conv(c1, c_, 1)
        self.fuse = Conv(c_ * 4, c2, 1)
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fuse(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1))
        return x + y if self.add else y


class LSKA(nn.Module):
    """Large Separable Kernel Attention: 1D 分离大核注意力（大感受野、参数量小）.

    Reference: Lau et al., "Large Separable Kernel Attention" (Expert Systems with Applications
    2024), doi:10.1016/j.eswa.2023.121352.
    """

    def __init__(self, dim: int, k: int = 11):  # noqa: ARG002 (k kept for interface clarity; 11 decomposition below)
        super().__init__()
        self.conv0h = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv_h = nn.Conv2d(dim, dim, (1, 7), padding=(0, 9), groups=dim, dilation=(1, 3))
        self.conv_v = nn.Conv2d(dim, dim, (7, 1), padding=(9, 0), groups=dim, dilation=(3, 1))
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv1(self.conv_v(self.conv_h(self.conv0v(self.conv0h(x)))))
        return x * attn


class SPPF_LSKA(nn.Module):
    """SPPF + LSKA: 在 SPPF 的多尺度池化聚合特征上施加大核注意力再压缩.

    P5 的全局上下文对判断"远处那一团暗点是不是柑橘"至关重要；LSKA 让 SPPF 聚合特征
    具备空间选择性而非均匀混合。
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, 11)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(self.lska(torch.cat(y, 1)))


class DWR(nn.Module):
    """Dilation-Wise Residual: 先 3x3 收集区域特征，再多膨胀率并联扩展语义感受野.

    Reference: Wei et al., "DWRSeg" (2022), arXiv:2212.01173.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_ = c2 // 2
        self.conv_r = Conv(c1, c_, 3)
        self.d1 = Conv(c_, c_, 3, d=1)
        self.d3 = Conv(c_, c_ // 2, 3, d=3)
        self.d5 = Conv(c_, c_ // 2, 3, d=5)
        self.fuse = Conv(c_ * 2, c2, 1, act=False)
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.conv_r(x)
        y = self.fuse(torch.cat([self.d1(r), self.d3(r), self.d5(r)], 1))
        return x + y if self.add else y


# ---------------------------------------------------------------------------------------------------------------------
# 5. 轻量化 block（C3k2 变体）
# ---------------------------------------------------------------------------------------------------------------------
class PConv(nn.Module):
    """Partial Convolution (FasterNet): 只在 1/4 通道上做 3x3 空间卷积，其余通道恒等直通."""

    def __init__(self, c: int, k: int = 3, ratio: float = 0.25):
        super().__init__()
        self.cp = max(8, int(c * ratio))
        self.conv = nn.Conv2d(self.cp, self.cp, k, 1, k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp, xu = torch.split(x, [self.cp, x.shape[1] - self.cp], dim=1)
        return torch.cat([self.conv(xp), xu], 1)


class FasterBlock(nn.Module):
    """FasterNet block: PConv + 1x1 MLP（残差），FLOPs 显著低于标准 Bottleneck.

    Reference: Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
    (CVPR 2023), arXiv:2303.03667.
    """

    def __init__(self, c1: int, c2: int = None, e: float = 2.0):
        super().__init__()
        c2 = c2 or c1
        c_ = int(c2 * e)
        self.pconv = PConv(c1)
        self.mlp = nn.Sequential(Conv(c1, c_, 1), nn.Conv2d(c_, c2, 1, bias=False))
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(self.pconv(x))
        return x + y if self.add else y


class C3k2_Faster(C3k2):
    """C3k2 with FasterNet blocks (PConv) — 轻量化主干/颈部单元."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(FasterBlock(self.c) for _ in range(n))


class WTConv(nn.Module):
    """简化单层 WTConv: 小波域深度卷积扩大有效感受野（低频路径等效 2x 感受野）.

    模糊 = 高频衰减。WTConv 显式在 LL/LH/HL/HH 四个频带分别卷积，允许网络放大残存高频、
    并用低频大感受野弥补模糊目标的结构缺失。
    Reference: Finder et al., "Wavelet Convolutions for Large Receptive Fields" (ECCV 2024),
    arXiv:2407.05848.
    """

    def __init__(self, c: int, k: int = 5):
        super().__init__()
        self.base = DWConv(c, c, k)
        self.wave = nn.Conv2d(c * 4, c * 4, k, 1, k // 2, groups=c * 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ll, lh, hl, hh = _haar_dwt(x)
        bands = self.wave(torch.cat([ll, lh, hl, hh], 1))
        ll, lh, hl, hh = bands.chunk(4, 1)
        y = _haar_idwt(ll, lh, hl, hh)[..., :h, :w]
        return self.base(x) + y


class WTBottleneck(nn.Module):
    """Bottleneck with wavelet conv: 1x1 降维 → WTConv 频带卷积 → 3x3 升维（残差）."""

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.wt = WTConv(c_, 5)
        self.cv2 = Conv(c_, c2, 3)
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.wt(self.cv1(x)))
        return x + y if self.add else y


class C3k2_WT(C3k2):
    """C3k2 with WTConv bottlenecks — 频域大感受野单元（针对模糊目标）."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(WTBottleneck(self.c, self.c) for _ in range(n))


class C3k2_DWR(C3k2):
    """C3k2 with DWR blocks — 多膨胀率感受野单元（小目标上下文）."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(DWR(self.c, self.c) for _ in range(n))


class SXQBottleneck(nn.Module):
    """SXQ 自研 bottleneck（原创三合一融合）：部分卷积 × 大核上下文 × 卷积门控.

    解决什么问题：标准 Bottleneck 的两个 3x3 全通道卷积既贵（占 C3k2 参数大头）又只有 3x3
    感受野（远处小果需要上下文）且是静态混合（不同区域一视同仁）。
    为什么这么做（每条支路一个机制）：
      - PConv 支：只在 1/4 通道做 3x3 空间混合，其余直通——FasterNet 证明空间冗余集中于少数通道；
      - DW 大核支：7x7 深度卷积全通道补上下文——深度卷积让大核几乎免费（ConvNeXt 结论）；
      - 卷积门控：sigmoid(1x1(x)) 逐元素门控两支之和——TransNeXt 的 Convolutional GLU 思想，
        让"哪里该信细节、哪里该信上下文"由内容决定（远处糊果区自动偏向上下文支）。
    有什么好处：参数约为标准 Bottleneck 的 40%（c_=64 时 ~30K vs 74K），感受野 3→7，动态可选择；
    全部标准算子（conv/dw/sigmoid/mul），端侧直转。
    融合来源：FasterNet (CVPR 2023, arXiv:2303.03667) × ConvNeXt 大核 DW (CVPR 2022,
    arXiv:2201.03545) × TransNeXt Convolutional GLU (CVPR 2024, arXiv:2311.17132)；三合一组合原创。
    """

    def __init__(self, c1: int, c2: int, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.pconv = PConv(c_, 3)
        self.dwk = nn.Conv2d(c_, c_, 7, 1, 3, groups=c_, bias=False)
        self.gate = nn.Conv2d(c_, c_, 1)
        self.cv2 = Conv(c_, c2, 1)
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = (self.pconv(y) + self.dwk(y)) * torch.sigmoid(self.gate(y))
        y = self.cv2(y)
        return x + y if self.add else y


class C3k2_SXQ(C3k2):
    """C3k2 with SXQBottleneck — SXQNet 家族自研块（部分卷积×大核×门控三合一，原创）."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(SXQBottleneck(self.c, self.c) for _ in range(n))


class MoCEBottleneck(nn.Module):
    """Mixture of Conv Experts bottleneck（原创迁移）：LLM MoE 思想的轻量 CNN 落地.

    解决什么问题：同一果园图像里共存四种成像条件（近亮/远暗/模糊/伪装），单一静态卷积核
    要同时拟合全部条件——容量被平均分摊。
    为什么这么做：4 个 5x5 深度卷积"专家核"，router=GAP→fc→softmax 按内容**软组合**专家核
    后做一次卷积（CondConv 核组合式实现：FLOPs≈单个 DW 卷积，参数仅 +3 个 DW 核）；
    软路由（非 top-k 硬路由）保证 ONNX 可导、端侧可部署。
    有什么好处：容量按成像条件条件化（router 可视化=各专家分工，论文可解释性素材），
    FLOPs 几乎不变；DW 专家让参数增量极小。
    融合来源（theme13 已核验）：MoE (Shazeer, ICLR 2017, arXiv:1701.06538) → CondConv (NeurIPS
    2019, arXiv:1904.04971) → Dynamic Conv (CVPR 2020, arXiv:1912.03458)；软路由可导出性引
    Soft MoE (arXiv:2308.00951)。新颖性边界：MoE-in-YOLO 已有 YOLO-Master (arXiv:2512.23273)
    先例，**只可声称组合创新**（nano 实例分割 + 农业成像条件语义路由 + 全软路由 DW 专家保导出
    ——三者交集无先例），写作须引 YOLO-Master 并做差异对比。
    """

    def __init__(self, c1: int, c2: int, experts: int = 4, k: int = 5):
        super().__init__()
        c_ = c2
        self.experts = experts
        self.k = k
        self.cv1 = Conv(c1, c_, 1)
        self.weight = nn.Parameter(torch.randn(experts, c_, 1, k, k) * 0.02)  # E 个 DW 专家核
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_, experts, 1))
        self.cv2 = Conv(c_, c2, 1)
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        b, c, h, w = y.shape
        r = self.router(y).flatten(1).softmax(dim=1)  # (b, E) 软路由
        kernel = torch.einsum("be,ecokk->bcokk".replace("kk", "xy"), r, self.weight.flatten(3).view(
            self.experts, c, 1, self.k, self.k))  # (b, c, 1, k, k)
        y = F.conv2d(y.reshape(1, b * c, h, w), kernel.reshape(b * c, 1, self.k, self.k),
                     padding=self.k // 2, groups=b * c).view(b, c, h, w)
        y = self.cv2(F.silu(y))
        return x + y if self.add else y


class C3k2_MoCE(C3k2):
    """C3k2 with MoCE bottlenecks — 卷积专家混合（MoE 思想 CNN 落地，原创迁移）."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(MoCEBottleneck(self.c, self.c) for _ in range(n))


class HyperRes(nn.Module):
    """双流超连接残差堆叠（"换方向的残差"思想迁移，出处以 theme13 核验为准）.

    解决什么问题：标准残差 y=x+f(x) 是单流单速率——深层小网络里梯度与特征只有一条高速路，
    信息混合速率固定。
    为什么这么做：维护两条残差流 h1/h2，每个块前用可学习系数混合出块输入，块输出按可学习
    2x3 矩阵回写两流（init 精确退化为标准残差：h1 承担原路，h2 恒等旁路）——
    残差"方向/速率"变为可学习，深度加深时梯度路径更丰富。
    有什么好处：参数增量每块仅 8 个标量；init 等价原网络，安全迁移；
    对 Shallow-Heavy 加深后的浅层堆叠（4 个块）尤其有意义。
    文献基础（theme13 已核验）：Hyper-Connections (ByteDance Seed, ICLR 2025, arXiv:2409.19606)
    的 2 流 lite 实现；谱系延伸 mHC (DeepSeek, arXiv:2512.24880) → Attention Residuals
    (Kimi Team, arXiv:2603.15031，深度方向注意力残差——本模块的可升级方向)。块内为自研 SXQBottleneck。
    """

    def __init__(self, c1: int, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(SXQBottleneck(c1, c1) for _ in range(blocks))
        # 每块: in_mix (2,) init [1,0]; out_mix (2,3) init [[1,0,1],[0,1,0]] → 标准残差等价
        self.in_mix = nn.Parameter(torch.tensor([[1.0, 0.0]] * blocks))
        out0 = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]] * blocks)
        self.out_mix = nn.Parameter(out0)
        self.final = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, h2 = x, x
        for i, blk in enumerate(self.blocks):
            inp = self.in_mix[i, 0] * h1 + self.in_mix[i, 1] * h2
            y = blk(inp) - inp if isinstance(blk, SXQBottleneck) else blk(inp)  # 取纯残差项 f(inp)
            m = self.out_mix[i]
            h1, h2 = (m[0, 0] * h1 + m[0, 1] * h2 + m[0, 2] * y,
                      m[1, 0] * h1 + m[1, 1] * h2 + m[1, 2] * y)
        return self.final[0] * h1 + self.final[1] * h2


class LSBottleneck(nn.Module):
    """Bottleneck with LSConv：1x1 降维 → LS 卷积（看大聚小）→ 1x1 升维（残差）.

    LSConv = 大核感知 + 小核聚合的仿生动态卷积（"See Large, Focus Small"）——先用大感受野
    看清枝叶上下文，再以小核动态聚合果实细节，与"远处小果需要上下文佐证"的需求同构。
    Reference: LSNet (Wang et al., CVPR 2025, arXiv:2503.23135)，fork 内置官方结构实现 (lsnet.py)。
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.ls = LSConv(c_, c_)
        self.cv2 = Conv(c_, c2, 1)
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.ls(self.cv1(x)))
        return x + y if self.add else y


class C3k2_LS(C3k2):
    """C3k2 with LSConv bottlenecks — CVPR 2025 '看大聚小' 动态卷积单元."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.m = nn.ModuleList(LSBottleneck(self.c, self.c) for _ in range(n))


# ---------------------------------------------------------------------------------------------------------------------
# 6. 特征融合
# ---------------------------------------------------------------------------------------------------------------------
class BiFPNConcat(nn.Module):
    """BiFPN 风格可学习加权 Concat（快速归一化融合）.

    颈部 Concat 对深浅层特征"一视同仁"；远处小柑橘依赖的浅层细节在融合时容易被深层语义淹没，
    可学习权重让网络自动上调浅层贡献。
    Reference: Tan et al., "EfficientDet: Scalable and Efficient Object Detection" (CVPR 2020),
    arXiv:1911.09070（加权特征融合思想，此处为 concat 维度的轻量适配）.
    """

    def __init__(self, n: int = 2, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w)
        w = w * len(x) / (w.sum() + self.eps)  # 归一化后保持总幅值 ~n，避免训练初期特征幅值骤降
        return torch.cat([w[i] * xi for i, xi in enumerate(x)], self.d)


# ---------------------------------------------------------------------------------------------------------------------
# 7. 原创组合模块（本课题针对性设计）
# ---------------------------------------------------------------------------------------------------------------------
class DFEM(nn.Module):
    """Dual-domain Frequency Enhancement Module（原创）：双域频率增强模块.

    针对痛点"远处柑橘模糊(高频衰减) + 发黑(弱响应)"的联合补偿：
    1) 频域分支：rFFT2 后按归一化半径分成 `bands` 个频带，每通道每频带学习残差增益
       （init=0 → 增益=1，训练前恒等），网络可自适应放大被模糊衰减的高频带；
    2) 暗区分支：以通道均值的 sigmoid 作为"响应亮度图"，对弱响应区域（远处暗果）施加
       可学习的通道增益补偿，等效于特征空间的局部曝光补偿；
    3) 空间分支：DW 3x3 保局部细节，1x1 融合输出（残差式，稳定可迁移预训练权重）。

    文献基础：频带调制思想源于 FreqFusion (Chen et al., TPAMI 2024, arXiv:2408.12879) 与
    WTConv (ECCV 2024, arXiv:2407.05848)；暗区补偿思想源于 PE-YOLO 金字塔增强
    (Yin et al., ICANN 2023, arXiv:2307.10953)；三者在特征空间的联合按元素调制为本课题原创组合。
    """

    def __init__(self, c1: int, bands: int = 4):
        super().__init__()
        self.bands = bands
        self.gains = nn.Parameter(torch.zeros(c1, bands))  # 频带残差增益，init 0 → 恒等
        cm = max(8, c1 // 4)
        self.dark_gain = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c1, cm, 1), nn.SiLU(), nn.Conv2d(cm, c1, 1), nn.Sigmoid()
        )
        self.local = DWConv(c1, c1, 3)
        self.fuse = Conv(c1, c1, 1)
        self._band_cache: dict[tuple, torch.Tensor] = {}

    def _band_index(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        cached = self._band_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(h, device=device).view(-1, 1)  # [-0.5, 0.5)
        fx = torch.fft.rfftfreq(w, device=device).view(1, -1)  # [0, 0.5]
        r = torch.sqrt(fy * fy + fx * fx) / (0.5 * math.sqrt(2.0))  # 归一化半径 [0, 1]
        idx = (r * self.bands).long().clamp_(0, self.bands - 1)  # (h, w//2+1)
        self._band_cache[key] = idx
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # 1) 频域增益
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        idx = self._band_index(h, w, x.device)
        gain = 1.0 + self.gains.float()[:, idx]  # (c, h, w//2+1)
        y = torch.fft.irfft2(xf * gain.unsqueeze(0), s=(h, w), norm="ortho").to(x.dtype)
        # 2) 暗区（弱响应）补偿
        brightness = torch.sigmoid(x.mean(1, keepdim=True))  # (b,1,h,w) 响应亮度图
        y = y + x * (1.0 - brightness) * self.dark_gain(x)
        # 3) 空间细节 + 融合（残差）
        return x + self.fuse(y + self.local(x))


class LIAM(nn.Module):
    """Luminance-Invariant Attention Module（原创）：亮度不变注意力模块.

    针对痛点"同一棵树上近处亮果与远处暗果亮度差异巨大，网络偏向学习亮果统计量"：
    1) 实例归一化分支消除 per-image/per-region 的亮度与对比度偏移（风格/光照不变特征），
       用可学习通道门控 α 与原特征混合，避免过度归一化丢失判别信息；
    2) 在对齐后的特征上计算 SimAM 式无参能量注意力，突出局部离群的弱小目标信号。

    文献基础：IN 的光照/风格不变性来自 IBN-Net (Pan et al., ECCV 2018, arXiv:1807.09441)；
    能量注意力来自 SimAM (Yang et al., ICML 2021)。二者的门控级联为本课题原创组合。
    """

    def __init__(self, c1: int, e_lambda: float = 1e-4):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(c1, affine=True)
        self.alpha = nn.Parameter(torch.zeros(c1))  # sigmoid(0)=0.5 起步，各通道自学习归一化强度
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        xn = a * self.inorm(x) + (1.0 - a) * x
        n = xn.shape[2] * xn.shape[3] - 1
        d = (xn - xn.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return xn * torch.sigmoid(e_inv)


class CSFG(nn.Module):
    """Cross-Stage Small-object Feature Guidance（原创）：跨级小目标特征引导.

    P2 检测头虽对小目标有效但代价大（P2 层特征图 160x160）。CSFG 提供轻量替代：
    1) P2 高分辨率特征经 SPD 无损下采样对齐到 P3 尺度（不丢像素信息）；
    2) 高通滤波（x - avgpool(x)）提取 P2 细节残差（小目标边缘所在）；
    3) 由 P3 内容生成空间门控，仅在"P3 认为可能有目标"的位置注入 P2 细节，抑制背景噪声。

    文献基础：浅层特征注入思想源于 Gold-YOLO 的 gather-distribute (Wang et al., NeurIPS 2023,
    arXiv:2309.11331) 与 ASF-YOLO 的尺度融合 (Kang et al., 2024, doi:10.1016/j.imavis.2024.104957)；
    "SPD 无损对齐 + 高通细节 + 内容门控"的组合为本课题原创设计。
    """

    def __init__(self, c_p2: int, c_p3: int):
        super().__init__()
        self.down = SPDConv(c_p2, c_p3, 3)
        cm = max(8, c_p3 // 4)
        self.gate = nn.Sequential(Conv(c_p3, cm, 1), nn.Conv2d(cm, 1, 1), nn.Sigmoid())
        self.fuse = Conv(c_p3, c_p3, 3)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        p2, p3 = x
        d = self.down(p2)
        if d.shape[-2:] != p3.shape[-2:]:
            d = F.interpolate(d, size=p3.shape[-2:], mode="nearest")
        detail = d - F.avg_pool2d(d, 3, 1, 1)
        return self.fuse(p3 + self.gate(p3) * detail)


# ---------------------------------------------------------------------------------------------------------------------
# 8. XX-Former 范式原创模块（MetaFormer 结构：x + TokenMixer(Norm(x)); x + FFN(Norm(x))）
#    范式出处: MetaFormer (Yu et al., CVPR 2022, arXiv:2111.11418)
# ---------------------------------------------------------------------------------------------------------------------
_BAND_CACHE: dict = {}


def _radial_band_index(h: int, w: int, bands: int, device: torch.device) -> torch.Tensor:
    """rFFT2 频谱的归一化径向频带索引 (h, w//2+1)，全局缓存."""
    key = (h, w, bands, str(device))
    cached = _BAND_CACHE.get(key)
    if cached is not None:
        return cached
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
    r = torch.sqrt(fy * fy + fx * fx) / (0.5 * math.sqrt(2.0))
    idx = (r * bands).long().clamp_(0, bands - 1)
    _BAND_CACHE[key] = idx
    return idx


class LRSA(nn.Module):
    """Low-Resolution Self-Attention: QKV 全部池化到固定小尺寸计算全局注意力再双线性还原.

    以近似线性代价获得图像级感受野——"远处那团暗点是不是柑橘"需要全图上下文佐证。
    Reference: Wu et al., "Low-Resolution Self-Attention for Semantic Segmentation"
    (LRFormer, IEEE TPAMI 2025, IEEE document 11029508).
    """

    def __init__(self, c: int, pool: int = 8, heads: int = 4):
        super().__init__()
        self.heads = heads if c % heads == 0 else 1
        self.scale = (c // self.heads) ** -0.5
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.pool(x)
        m = p.shape[-1]
        q, k, v = self.qkv(p).reshape(b, 3, self.heads, c // self.heads, m * m).unbind(1)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(-1)
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, m, m)
        return F.interpolate(self.proj(out), size=(h, w), mode="bilinear", align_corners=False)


class HFBranch(nn.Module):
    """Haar 高频分支：只处理 LH/HL/HH 三个高频子带并重建（弃 LL），输出纯细节图.

    模糊小果的残存边缘位于高频子带；对其单独卷积可学习放大。
    Reference: WTConv (Finder et al., ECCV 2024, arXiv:2407.05848) 的频带分离思想。
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(c * 3, c * 3, k, 1, k // 2, groups=c * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ll, lh, hl, hh = _haar_dwt(x)
        lh, hl, hh = self.conv(torch.cat([lh, hl, hh], 1)).chunk(3, 1)
        return _haar_idwt(torch.zeros_like(ll), lh, hl, hh)[..., :h, :w]


class MSDFFFN(nn.Module):
    """多尺度动态混合 FFN：1x1 升维 → 通道拆分 5x5/7x7 深度卷积 → 洗牌 → 1x1 降维.

    Reference: SRConvNet 的 Dynamic Mixing Layer (Li et al., IJCV 2025,
    doi:10.1007/s11263-024-02147-y)。
    """

    def __init__(self, c: int, e: float = 2.0):
        super().__init__()
        c_ = int(c * e)
        self.half = c_ // 2
        self.fc1 = Conv(c, c_, 1)
        self.dw5 = nn.Conv2d(self.half, self.half, 5, 1, 2, groups=self.half, bias=False)
        self.dw7 = nn.Conv2d(c_ - self.half, c_ - self.half, 7, 1, 3, groups=c_ - self.half, bias=False)
        self.fc2 = nn.Conv2d(c_, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y1, y2 = torch.split(y, [self.half, y.shape[1] - self.half], dim=1)
        y = torch.cat([self.dw5(y1), self.dw7(y2)], 1)
        b, c_, h, w = y.shape
        y = y.view(b, 2, c_ // 2, h, w).transpose(1, 2).reshape(b, c_, h, w)  # channel shuffle
        return self.fc2(F.silu(y))


class FocusedLinearAttn(nn.Module):
    """Focused Linear Attention：O(N) 线性注意力（ReLU 特征映射 + 聚焦幂 + DWC 秩恢复）.

    选型依据（theme7 调研裁决）：P5 仅 ~400 token，Mamba 的 selective scan 需 CUDA 编译、
    Windows/ONNX 均困难且短序列无优势；MLLA (Han et al., NeurIPS 2024, arXiv:2405.16605)
    证明 Mamba 数学上即带遗忘门的线性注意力——故用纯 PyTorch 线性注意力替代，可导出部署。
    Reference: FLatten Transformer (Han et al., ICCV 2023, arXiv:2308.00442)。
    """

    def __init__(self, c: int, heads: int = 4, p: float = 3.0):
        super().__init__()
        self.heads = heads if c % heads == 0 else 1
        self.p = p
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.dwc = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        d = c // self.heads
        q, k, v = self.qkv(x).reshape(b, 3, self.heads, d, h * w).permute(1, 0, 2, 4, 3).unbind(0)  # (b,H,n,d)
        eps = 1e-6
        q, k = F.relu(q) + eps, F.relu(k) + eps
        # focused mapping：幂运算锐化分布后恢复原范数（保持特征幅值）
        qn, kn = q.norm(dim=-1, keepdim=True), k.norm(dim=-1, keepdim=True)
        q = q.pow(self.p)
        k = k.pow(self.p)
        q = q / (q.norm(dim=-1, keepdim=True) + eps) * qn
        k = k / (k.norm(dim=-1, keepdim=True) + eps) * kn
        kv = torch.einsum("bhnd,bhne->bhde", k, v)  # (b,H,d,d)
        denom = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + eps
        out = torch.einsum("bhnd,bhde->bhne", q, kv) / denom.unsqueeze(-1)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        v_sp = v.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.proj(out + self.dwc(v_sp))


class _FarFormerBlock(nn.Module):
    """FarFormer 基本块（原创，MetaFormer 范式）.

    Token Mixer = LGFM（Low-resolution Global + high-Frequency Mixer）：
        α·全局分支 + (1-α)·HFBranch(Haar 高频细节)，α 为可学习通道门控
        ——远处小果需要"全局上下文佐证 + 高频边缘辨认"两种互补信息。
        全局分支可选：mixer="lrsa"（低分辨率注意力）或 "fla"（focused 线性注意力，消融对照）。
    FFN = MSDFFFN（多尺度动态混合）。
    融合来源：LRFormer (TPAMI 2025) + WTConv (ECCV 2024) + SRConvNet (IJCV 2025)，组合与门控为本课题原创。
    """

    def __init__(self, c: int, pool: int = 8, mixer: str = "lrsa", dyt: bool = False):
        super().__init__()
        norm = (lambda ch: DyT(ch)) if dyt else (lambda ch: nn.GroupNorm(1, ch))  # DyT 仅替换不可折叠的 GN
        self.norm1 = norm(c)
        self.g = LRSA(c, pool) if mixer == "lrsa" else FocusedLinearAttn(c)
        self.f = HFBranch(c)
        self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.norm2 = norm(c)
        self.ffn = MSDFFFN(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        a = torch.sigmoid(self.alpha)
        x = x + a * self.g(y) + (1.0 - a) * self.f(y)
        return x + self.ffn(self.norm2(x))


class FarFormer(nn.Module):
    """FarFormer：远场感知 Former（原创）。blocks 个 _FarFormerBlock 串联，通道不变."""

    def __init__(self, c: int, blocks: int = 1, pool: int = 8, mixer: str = "lrsa", dyt: bool = False):
        super().__init__()
        self.m = nn.Sequential(*(_FarFormerBlock(c, pool, mixer, dyt) for _ in range(blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class _FreqChannelAttn(nn.Module):
    """频域通道注意力：rFFT2 幅谱去直流后按通道聚合 → MLP → 通道权重.

    响应平坦（无结构）的通道谱能量集中于 DC，去 DC 后能量低 → 被抑制；
    携带果实纹理/边缘结构的通道被增强——绿果与平坦叶背景的频域可分性。
    Reference: HS-FPN 的高频感知通道路径 (Chen et al., AAAI 2025, arXiv:2412.10116)。
    """

    def __init__(self, c: int, r: int = 4):
        super().__init__()
        cm = max(8, c // r)
        self.mlp = nn.Sequential(nn.Conv2d(c, cm, 1), nn.SiLU(), nn.Conv2d(cm, c, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.fft.rfft2(x.float(), norm="ortho").abs()
        mag[..., 0, 0] = 0.0  # 去直流：只保留结构/纹理谱能量
        w = self.mlp(mag.mean(dim=(2, 3), keepdim=True).to(x.dtype))
        return x * w


class _LumiSpatialMod(nn.Module):
    """暗区空间调制：由响应亮度图生成暗区门控，放大弱响应（远处暗果）区域的特征.

    Reference: PE-YOLO 金字塔增强 (arXiv:2307.10953) 与 HVI/CIDNet 暗区增强思想
    (CVPR 2025, doi:10.1109/CVPR52734.2025.00533)。
    """

    def __init__(self, c: int):
        super().__init__()
        self.dw = DWConv(c, c, 3)
        self.gate = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.SiLU(), nn.Conv2d(8, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dark = 1.0 - torch.sigmoid(x.mean(1, keepdim=True))
        return self.dw(x) * (1.0 + self.gate(dark))


class EDFFN(nn.Module):
    """末端频域筛选 FFN：1x1 升维 → DW3x3 → 1x1 降维 → 可学习频带增益筛选（init 恒等）.

    与常规"FFN 中间层做频域"不同，在 FFN 末端做频率筛选，代价更低且保高频。
    Reference: EVSSM 的 EDFFN (Kong et al., CVPR 2025, arXiv:2405.14343)。
    """

    def __init__(self, c: int, e: float = 2.0, bands: int = 4):
        super().__init__()
        c_ = int(c * e)
        self.fc1 = Conv(c, c_, 1)
        self.dw = DWConv(c_, c_, 3)
        self.fc2 = nn.Conv2d(c_, c, 1)
        self.bands = bands
        self.gains = nn.Parameter(torch.zeros(c, bands))
        # LayerScale 风格残差缩放（Touvron et al., CaiT, arXiv:2103.17239）：
        # 兼顾训练稳定 + 保证 irfft2 反向拿到实体化梯度（Windows MKL 对 broadcast 梯度报错）
        self.gamma = nn.Parameter(0.01 * torch.ones(c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.dw(self.fc1(x)))
        h, w = y.shape[-2:]
        yf = torch.fft.rfft2(y.float(), norm="ortho")
        gain = self.gains.float()[:, _radial_band_index(h, w, self.bands, y.device)]  # 纯频带残差，init 0
        res = torch.fft.irfft2(yf * gain.unsqueeze(0), s=(h, w), norm="ortho").to(y.dtype)
        return y + self.gamma * res


class _LumiFormerBlock(nn.Module):
    """LumiFormer 基本块（原创，MetaFormer 范式）.

    Token Mixer = 频域通道注意力(_FreqChannelAttn) → 暗区空间调制(_LumiSpatialMod) 串联
        ——通道维选出"有结构"的特征，空间维补偿"发黑"区域，即 HFP 的 CP×SP 双路思想的串联轻量版。
    FFN = EDFFN（末端频域筛选）。
    融合来源：HS-FPN (AAAI 2025) + CIDNet/PE-YOLO 暗区增强 + EVSSM EDFFN (CVPR 2025)，组合原创。
    """

    def __init__(self, c: int, dyt: bool = False):
        super().__init__()
        norm = (lambda ch: DyT(ch)) if dyt else (lambda ch: nn.GroupNorm(1, ch))
        self.norm1 = norm(c)
        self.fca = _FreqChannelAttn(c)
        self.lsm = _LumiSpatialMod(c)
        self.norm2 = norm(c)
        self.ffn = EDFFN(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        x = x + self.lsm(self.fca(y))
        return x + self.ffn(self.norm2(x))


class LumiFormer(nn.Module):
    """LumiFormer：亮度感知 Former（原创）。blocks 个 _LumiFormerBlock 串联，通道不变."""

    def __init__(self, c: int, blocks: int = 1, dyt: bool = False):
        super().__init__()
        self.m = nn.Sequential(*(_LumiFormerBlock(c, dyt) for _ in range(blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


# ---------------------------------------------------------------------------------------------------------------------
# 9. 数据驱动第二轮原创（依据 _dataset_analysis.md 的量化证据设计；全部端侧友好算子）
# ---------------------------------------------------------------------------------------------------------------------
class MWCA(nn.Module):
    """Multi-level Wavelet Cross-band Attention（原创频域模块）：2 级小波分解 + 跨频带注意力 + 高频显著门控.

    数据依据（_dataset_analysis.md）：小果与大果的 Laplacian 模糊度相差 20 倍——判别信息随
    距离在频带间迁移（近果在高频、远果只剩中低频），固定卷积无法按内容自适应选频带。
    机制：
      1) 2 级 Haar DWT 得 7 个子带：{LL2,LH2,HL2,HH2}@1/4 分辨率 + {LH1,HL1,HH1}@1/2 分辨率；
      2) 各子带深度卷积精炼后取 GAP 描述子，经跨频带 MLP（band 间全交互）输出逐子带逐通道门控
         ——网络自己学"当前输入该信哪个频带"；
      3) 一级高频组经 1x1 → sigmoid 生成"伪装痕迹显著图"（绿果与绿叶的差异藏在高频纹理统计），
         下采样后调制 LL2 语义路（频域找伪装的 FEDER 思想 + HS-FPN 高频引导）；
      4) 两级 iDWT 重建，LayerScale 残差输出（init 0.01，训练稳定）。
    全部为 slice/conv/linear 算子（无 FFT），端侧可转。
    融合来源：FEDER 频率分解辨伪装 (He et al., CVPR 2023) + WTConv 小波域卷积 (ECCV 2024,
    arXiv:2407.05848) + HS-FPN 高频感知 (AAAI 2025, arXiv:2412.10116)；跨频带注意力与
    显著门控的组合为本课题原创。
    """

    def __init__(self, c1: int, reduction: int = 8, hidden: int = 64):
        super().__init__()
        self.c = c1
        self.dw1 = nn.Conv2d(c1 * 3, c1 * 3, 3, 1, 1, groups=c1 * 3, bias=False)  # level-1 高频组
        self.dw2 = nn.Conv2d(c1 * 4, c1 * 4, 3, 1, 1, groups=c1 * 4, bias=False)  # level-2 全组
        cs = max(4, c1 // reduction)
        self.squeeze = nn.Linear(c1, cs)
        self.mix = nn.Sequential(nn.Linear(7 * cs, hidden), nn.SiLU(), nn.Linear(hidden, 7 * c1))
        self.sal = nn.Sequential(nn.Conv2d(c1 * 3, 1, 1), nn.Sigmoid())
        self.out = Conv(c1, c1, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones(c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ll1, lh1, hl1, hh1 = _haar_dwt(x)
        ll2, lh2, hl2, hh2 = _haar_dwt(ll1)
        hi1 = self.dw1(torch.cat([lh1, hl1, hh1], 1))
        lo2 = self.dw2(torch.cat([ll2, lh2, hl2, hh2], 1))
        lh1, hl1, hh1 = hi1.chunk(3, 1)
        ll2, lh2, hl2, hh2 = lo2.chunk(4, 1)
        bands = [ll2, lh2, hl2, hh2, lh1, hl1, hh1]
        # 跨频带注意力：7 个子带描述子全交互 → 逐子带逐通道门控
        desc = torch.stack([t.mean(dim=(2, 3)) for t in bands], 1)  # (b, 7, c)
        gates = torch.sigmoid(self.mix(self.squeeze(desc).flatten(1))).view(b, 7, c, 1, 1)
        bands = [t * gates[:, i] for i, t in enumerate(bands)]
        ll2, lh2, hl2, hh2, lh1, hl1, hh1 = bands
        # 高频显著图门控低频语义路（伪装痕迹在高频）
        sal = self.sal(torch.cat([lh1, hl1, hh1], 1))  # (b,1,h/2,w/2)
        ll2 = ll2 * (1.0 + F.avg_pool2d(sal, 2))
        y = _haar_idwt(ll2, lh2, hl2, hh2)[..., : lh1.shape[-2], : lh1.shape[-1]]
        y = _haar_idwt(y, lh1, hl1, hh1)[..., :h, :w]
        return x + self.gamma * self.out(y)


class TDAM(nn.Module):
    """Texture-Difference Amplification Module（原创）：多尺度中心-邻域差分纹理放大.

    数据依据：柑橘幼果与叶片同色（|Δa*| 对比极低），但**纹理模式不同**——果面是光滑球面渐变，
    叶面有叶脉/锯齿边缘。伪装目标检测(COD)的核心手段正是放大这种纹理统计差异。
    机制：d_k = x - avgpool_k(x)（k=3/7/11 三尺度 center-surround 差分，即 DoG 近似）
    → 1x1 融合 → 内容空间门控（只在"疑似有目标"处放大，抑制背景噪声）→ 残差注入。
    纯 pool/conv/sigmoid 算子——单片机部署链路可直转。

    文献基础：SINet 的感受野对比机制 (Fan et al., "Camouflaged Object Detection", CVPR 2020)、
    PFNet distraction mining (Mei et al., CVPR 2021)、Zhai et al. 绿果=COD 立论
    (Comput. Electron. Agric. 2024, doi:10.1016/j.compag.2024.109356)；多尺度差分+门控残差组合为本课题原创。
    """

    def __init__(self, c1: int, ks: tuple = (3, 7, 11)):
        super().__init__()
        self.pools = nn.ModuleList(nn.AvgPool2d(k, 1, k // 2) for k in ks)
        self.fuse = Conv(c1 * len(ks), c1, 1)
        cm = max(8, c1 // 4)
        self.gate = nn.Sequential(Conv(c1, cm, 1), nn.Conv2d(cm, 1, 3, 1, 1), nn.Sigmoid())
        self.gain = nn.Parameter(torch.zeros(1, c1, 1, 1))  # init 0 → 恒等起步

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diffs = torch.cat([x - p(x) for p in self.pools], 1)
        d = self.fuse(diffs)
        return x + torch.sigmoid(self.gain) * self.gate(x) * d


class LCE(nn.Module):
    """Lightweight Curve Enhancement（原创组合）：暗区门控的 Zero-DCE 曲线增强前端（3->3）.

    数据依据：<32px 小果亮度中位数显著低于大果（欠曝），需要输入侧提亮；但全图增强会
    过曝近处亮果。机制：小 CNN 估计逐像素曲线参数 A ∈ (-1,1)，迭代 LE(x)=x+A·x·(1-x)
    提升暗部（Zero-DCE 核心公式）；A 先乘以暗区门控（由亮度图生成），仅作用于暗区域。
    全部为 conv/mul/sigmoid 算子，ONNX→NCNN/RKNN/TFLite 直转（比 HVIEnhance 更端侧友好）。

    文献基础：Zero-DCE (Guo et al., CVPR 2020, doi:10.1109/CVPR42600.2020.00185) 的曲线公式
    + 暗区选择性增强思想 (PE-YOLO, arXiv:2307.10953)；门控曲线组合为本课题原创。
    与 HVIEnhance (010/F23) 互为对照：HVI 换色彩空间做增强，LCE 在 RGB 域做门控曲线。
    """

    def __init__(self, c1: int = 3, c2: int = 3, iters: int = 4, feat: int = 16):
        super().__init__()
        self.iters = iters
        self.net = nn.Sequential(
            Conv(c1, feat, 3), Conv(feat, feat, 3), nn.Conv2d(feat, c2, 3, 1, 1), nn.Tanh()
        )
        self.gate = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1), nn.SiLU(), nn.Conv2d(8, 1, 3, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)  # A init 0 → 恒等起步，安全迁移预训练权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lum = x.mean(1, keepdim=True)
        g = self.gate(1.0 - lum)  # 暗区门控（输入约在 [0,1]）
        a = self.net(x) * g
        y = x
        for _ in range(self.iters):
            y = y + a * y * (1.0 - y)
        return y.clamp(0.0, 1.0)


class PCFA(nn.Module):
    """Partial-Channel Frequency Attention（原创融合）：只对 1/4 通道做频域增强，其余直通.

    解决什么问题：频域模块（DFEM/MWCA）有效但对全通道做变换——而高频判别信息集中在少数
    通道（FasterNet 对空间卷积的同款观察），全通道处理浪费算力、不利端侧。
    为什么这么做：通道拆分 [c/4, 3c/4]，仅前 1/4 走 rFFT2 → 频带残差增益（init 0 恒等）→
    irFFT2，其余 3/4 恒等直通；LayerScale 残差融合。
    有什么好处：频域增强的 FLOPs/参数降约 4 倍；恒等起步安全迁移；1/4 直通对内存带宽友好。
    融合来源：FasterNet 部分通道思想 (CVPR 2023, arXiv:2303.03667) × FreqFusion/DFEM 频带调制
    (TPAMI 2024, arXiv:2408.12879)；partial-channel 频域处理为本课题原创。
    注：含 FFT，用于带 FFT 加速的板子（GPU/NPU）；纯 MCU 线（V2）不用它。
    """

    def __init__(self, c1: int, ratio: float = 0.25, bands: int = 4):
        super().__init__()
        self.cp = max(8, int(c1 * ratio))
        self.bands = bands
        self.gains = nn.Parameter(torch.zeros(self.cp, bands))
        self.fuse = Conv(c1, c1, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones(c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        xp, xr = torch.split(x, [self.cp, x.shape[1] - self.cp], dim=1)
        f = torch.fft.rfft2(xp.float(), norm="ortho")
        gain = 1.0 + self.gains.float()[:, _radial_band_index(h, w, self.bands, x.device)]
        xp2 = torch.fft.irfft2(f * gain.unsqueeze(0), s=(h, w), norm="ortho").to(x.dtype)
        return x + self.gamma * self.fuse(torch.cat([xp2, xr], 1))


class HSF(nn.Module):
    """High-level Screening Fusion（HS-FPN lite）：高层语义筛选低层特征后融合，替代'上采样+Concat'.

    漏检根因之一：低层高分辨率特征背景噪声大，Concat 直接拼接会稀释远处小果的微弱信号，
    且拼接后通道翻倍、后续 C3k2 参数上涨。HSF 改为：高层语义生成通道筛选权重 → 过滤低层特征
    （只保留语义相关响应）→ 与上采样后的高层特征相加融合。输出通道 = 低层通道
    （不翻倍 → 后续 C3k2 输入减半，**参数量反而下降**，符合轻量化硬约束）。
    inputs: [low(高分辨率), high(低分辨率语义)]。
    Reference: HS-FPN (Chen et al., Computers in Biology and Medicine 2024,
    doi:10.1016/j.compbiomed.2024.107917，微小白细胞检测专用的轻量筛选式 FPN)。lite 实现为本课题适配。
    """

    def __init__(self, c_low: int, c_high: int):
        super().__init__()
        self.fc = nn.Conv2d(c_high, c_low, 1)
        self.proj = Conv(c_high, c_low, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        low, high = x
        w = torch.sigmoid(self.fc(F.adaptive_avg_pool2d(high, 1) + F.adaptive_max_pool2d(high, 1)))
        hu = F.interpolate(self.proj(high), size=low.shape[-2:], mode="nearest")
        return low * w + hu


class TGP(nn.Module):
    """Texture-Guided Prior frontend（原创，用户提出的"去颜色、纹理先验"思想的可行化实现）.

    动机（用户观察 + 数据体检互证）：果叶同色（|Δa*|≈2-3），颜色判别力低——判别力在纹理；
    但远处糊果纹理信噪比低（模糊度差 20 倍），硬用纹理先验反而有害，需可靠性感知。
    机制（3->3，参数 ~20 个、FLOPs≈0，纯 pool/元素运算，端侧直转）：
      1) 去颜色：V = max(R,G,B)（HSV 明度定义，丢弃色相/饱和度）；
      2) 多尺度纹理图：局部对比归一化 t_k=(V-μ_k)/(σ_k+ε)，k∈{3,7,15}
         ——LCN 对乘性光照增益不变（远处发黑不污染纹理图），多尺度覆盖近果粗纹理/远果细纹理；
      3) 可靠性门控（解决"远处样本纹理不好"）：局部标准差 σ 即纹理置信度
         （模糊/平坦区 σ 低），c = sigmoid(a·σ+b)，a/b 可学习——糊果处门控趋 0 回退 RGB 主流；
      4) out = clamp(x + γ·c·conv1x1([t_3,t_7,t_15]))，γ init=0 → 恒等起步、预训练权重安全。
    文献基础：局部对比归一化 (Jarrett et al., ICCV 2009) + 绿果=伪装需纹理判别的立论
    (Zhai et al. 2024, doi:10.1016/j.compag.2024.109356)；多尺度纹理金字塔 + 可靠性门控为本课题原创。
    """

    def __init__(self, c1: int = 3, c2: int = 3, ks: tuple = (3, 7, 15)):
        super().__init__()
        self.ks = ks
        self.fuse = nn.Conv2d(len(ks), c2, 1)
        self.a = nn.Parameter(torch.tensor(10.0))  # 置信度斜率
        self.b = nn.Parameter(torch.tensor(-0.5))  # 置信度偏置
        self.gamma = nn.Parameter(torch.zeros(c2, 1, 1))
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.max(dim=1, keepdim=True).values  # HSV 明度（去颜色）
        ts, sigma_mid = [], None
        for k in self.ks:
            mu = F.avg_pool2d(v, k, 1, k // 2)
            var = (F.avg_pool2d(v * v, k, 1, k // 2) - mu * mu).clamp(min=0.0)
            sigma = var.sqrt()
            ts.append(((v - mu) / (sigma + 1e-4)).clamp(-5.0, 5.0))
            if k == self.ks[len(self.ks) // 2]:
                sigma_mid = sigma
        conf = torch.sigmoid(self.a * sigma_mid + self.b)  # 纹理可靠性门控
        tex = self.fuse(torch.cat(ts, 1))
        return (x + self.gamma * conf * tex).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# 10. 顶会新范式（2024-2026）：物理算子 + 超图计算
# ---------------------------------------------------------------------------------------------------------------------
class HCO(nn.Module):
    """Heat Conduction Operator（vHeat 热传导算子，周期边界 FFT 近似实现）.

    新范式：把特征传播建模为物理热传导——热方程的解在频域是逐频率指数衰减
    y = F⁻¹( F(x) · exp(-‖ω‖²·k) )，k 为可学习每通道"热扩散时间"（传播距离）。
    远处小暗果需要大范围上下文佐证；HCO 以 O(N log N) 实现物理可解释的全局信息混合，
    比自注意力便宜、比大核卷积感受野更大，且 k 可视化即"每通道看多远"（论文可解释性卖点）。
    实现注：原论文用 DCT（绝热边界），此处用 rFFT2（周期边界）近似，量级一致、实现更简；
    k=softplus(param) init≈0.05 → 起步近恒等，残差安全。
    Reference: vHeat (Wang et al., 2024, arXiv:2405.16555)。
    """

    def __init__(self, c1: int, blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(_HCOBlock(c1) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class _HCOBlock(nn.Module):
    """单个 HCO 块：norm → 热传导混合(残差) → norm → MSDFFFN(残差)，MetaFormer 结构."""

    def __init__(self, c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.k = nn.Parameter(torch.full((c,), -3.0))  # softplus(-3)≈0.049，起步近恒等
        self.v_proj = Conv(c, c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = MSDFFFN(c)
        self._freq_cache: dict = {}

    def _freq_sq(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, str(device))
        cached = self._freq_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
        fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
        r2 = (fy * fy + fx * fx) * (4.0 * math.pi**2)  # ‖ω‖²
        self._freq_cache[key] = r2
        return r2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.v_proj(self.norm1(x))
        yf = torch.fft.rfft2(y.float(), norm="ortho")
        kernel = torch.exp(-self._freq_sq(h, w, x.device).unsqueeze(0) * F.softplus(self.k.float()).view(-1, 1, 1))
        y = torch.fft.irfft2(yf * kernel.unsqueeze(0), s=(h, w), norm="ortho").to(x.dtype)
        x = x + self.out(y)
        return x + self.ffn(self.norm2(x))


class HyperACE(nn.Module):
    """Hypergraph Adaptive Correlation Enhancement (lite)：超图高阶关联增强.

    新范式：卷积与自注意力都只建模**成对**关联；超图把像素当节点、自适应生成 E 条软超边
    （每条超边聚合一组语义相关节点，如"同一果串的果实"“同一枝条的叶片"），
    经 节点→超边→节点 两跳消息传递实现**多对多高阶**关联建模——密集相邻小果的
    群体证据可互相佐证（单个远处暗果证据不足，但"一串果"的群体模式判别力强）。
    实现：H = softmax_E(1x1conv(x))  (b,E,N) 软关联矩阵；edge = H·Xᵀ 归一化；
    node_update = Hᵀ·edge；1x1 融合 + LayerScale 残差。O(E·N·C)，E=8 时代价极小。
    Reference: Hyper-YOLO (Feng et al., IEEE TPAMI 2025, arXiv:2408.04804)；
    YOLOv13 HyperACE (Lei et al., 2025, arXiv:2506.17733)。lite 版软超边实现为本课题适配。
    """

    def __init__(self, c1: int, edges: int = 8):
        super().__init__()
        self.edges = edges
        self.assign = nn.Conv2d(c1, edges, 1)
        self.edge_proj = nn.Linear(c1, c1)
        self.out = Conv(c1, c1, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones(c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        feats = x.flatten(2).transpose(1, 2)  # (b, n, c)
        hmat = self.assign(x).flatten(2).softmax(dim=1)  # (b, E, n)，softmax over E → 每节点软分配到超边
        deg = hmat.sum(dim=2, keepdim=True) + 1e-6  # (b, E, 1) 超边度
        edge_feat = torch.bmm(hmat, feats) / deg  # (b, E, c) 超边特征 = 成员节点均值
        edge_feat = F.silu(self.edge_proj(edge_feat))
        node_update = torch.bmm(hmat.transpose(1, 2), edge_feat)  # (b, n, c) 两跳回传
        y = node_update.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * self.out(y)


class DyT(nn.Module):
    """Dynamic Tanh：tanh(alpha*x) 逐通道替代归一化层（LLM 界 2025 免归一化思路）.

    适用边界（theme14 核验后的诚实结论）：只用于替换 Former 块内的 GroupNorm（GN 推理时
    不可折叠、有归约开销）；**勿用于替换 BN**（BN 可免费折进卷积，替换是负收益），且 tanh
    与 INT8 量化存在冲突——量化部署线（V2/F44/F45）不使用本模块。
    做法：y = gamma * tanh(alpha * x) + beta，逐通道可学习，无归约。
    Reference: "Transformers without Normalization" (Zhu et al., CVPR 2025, arXiv:2503.10622,
    theme13 已核验)。可作为 Former 块中 GroupNorm 的端侧替换（消融行）。
    """

    def __init__(self, c1: int, alpha0: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(alpha0 * torch.ones(1, c1, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
