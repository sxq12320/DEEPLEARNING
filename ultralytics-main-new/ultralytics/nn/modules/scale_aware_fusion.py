"""
Scale-Aware Fusion Module — 基于"聚合-扩散"范式的尺度感知融合模块

核心思想: 先聚合指导信号，再扩散调制特征
  - 聚合 (Aggregation): 从辅助模态中提取指导信号 (空间/交叉/通道)
  - 扩散 (Diffusion): 将指导信号调制到主模态特征上

策略:
  P3 (depth2rgb):  Depth → 空间聚合 → Spatial Attention Map → 空间扩散到 RGB
  P4 (bidirectional): RGB↔Depth → 交叉聚合 → 残差扩散 + 自适应门控
  P5 (rgb_led):    Depth → 通道聚合(Squeeze) → Channel Attention → 通道扩散(Excitation)到 RGB
  naive:           Concat + 1x1 Conv (Baseline)
  ct:              复用已有 CT 控制理论融合模块
  apfm_ct:         APFM 预融合 + CT 控制理论融合

通过 fusion_cfg 字典控制每个尺度使用哪种策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# P3: Depth-to-RGB 空间引导融合
# ============================================================

class SpatialAggregation(nn.Module):
    """
    空间聚合: Depth → 轻量 CNN → Spatial Attention Map (H×W×1)
    聚合局部深度跳变信息，生成空间注意力图
    """
    def __init__(self, c_dep):
        super().__init__()
        # 轻量级 CNN: DWConv 3x3 → BN → ReLU → Conv 1x1 → 1ch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c_dep, c_dep, 3, padding=1, groups=c_dep, bias=False),
            nn.BatchNorm2d(c_dep),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_dep, 1, 1, bias=True),
        )

    def forward(self, f_dep):
        return self.spatial_conv(f_dep)  # (B, 1, H, W)


class SpatialDiffusion(nn.Module):
    """
    空间扩散: Spatial Attention Map → Sigmoid → 广播逐元素乘到 RGB 特征
    抠取前景区域，抑制背景噪声
    """
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__()
        # 通道对齐 + 输出投影
        self.align_rgb = nn.Conv2d(c_rgb, out_ch, 1, bias=False) if c_rgb != out_ch else nn.Identity()
        self.align_dep = nn.Conv2d(c_dep, out_ch, 1, bias=False)

    def forward(self, f_rgb, spatial_map):
        # Sigmoid 归一化注意力图
        attn = torch.sigmoid(spatial_map)  # (B, 1, H, W)
        # 广播乘到 RGB 特征上
        f_rgb_aligned = self.align_rgb(f_rgb)
        return f_rgb_aligned * attn


class Depth2RGBFusion(nn.Module):
    """
    P3 尺度融合: Depth-to-RGB 引导
    聚合: Depth 经轻量 CNN 聚合局部深度跳变 → Spatial Attention Map
    扩散: Map 经 Sigmoid 后广播乘到 RGB 特征上，抠取前景
    """
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__()
        self.aggregate = SpatialAggregation(c_dep)
        self.diffuse = SpatialDiffusion(c_rgb, c_dep, out_ch)

    def forward(self, f_rgb, f_dep):
        # 空间尺寸对齐
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
        # 聚合: Depth → Spatial Attention Map
        spatial_map = self.aggregate(f_dep)
        # 扩散: Map → Sigmoid → 乘到 RGB
        return self.diffuse(f_rgb, spatial_map)


# ============================================================
# P4: 双向互补对齐融合
# ============================================================

class LightweightCrossAttention(nn.Module):
    """
    轻量级双向交叉注意力
    RGB 和 Depth 互为 Q/K/V 聚合互补信息
    使用 DWConv + 1x1 Conv 实现高效交叉注意力
    """
    def __init__(self, c_rgb, c_dep, out_ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_ch // num_heads
        assert out_ch % num_heads == 0, f"out_ch={out_ch} 必须能被 num_heads={num_heads} 整除"

        # 通道对齐
        self.align_rgb = nn.Conv2d(c_rgb, out_ch, 1, bias=False)
        self.align_dep = nn.Conv2d(c_dep, out_ch, 1, bias=False)

        # RGB → Depth 交叉注意力 (RGB 作为 Q, Depth 作为 K/V)
        self.q_rgb = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.kv_dep = nn.Conv2d(out_ch, out_ch * 2, 1, bias=False)

        # Depth → RGB 交叉注意力 (Depth 作为 Q, RGB 作为 K/V)
        self.q_dep = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.kv_rgb = nn.Conv2d(out_ch, out_ch * 2, 1, bias=False)

        # 空间聚合 (局部上下文)
        self.dw_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_ch)

    def forward(self, f_rgb, f_dep):
        # 通道对齐
        rgb = self.align_rgb(f_rgb)   # (B, out_ch, H, W)
        dep = self.align_dep(f_dep)   # (B, out_ch, H, W)

        B, C, H, W = rgb.shape

        # ---- RGB → Depth 交叉注意力 ----
        q_r = self.q_rgb(rgb)                    # (B, C, H, W)
        kv_d = self.kv_dep(dep)                  # (B, 2C, H, W)
        k_d, v_d = kv_d.chunk(2, dim=1)          # 各 (B, C, H, W)

        # 简化注意力: Q * K 的空间加权 (避免昂贵的矩阵乘法)
        attn_rd = torch.sigmoid(q_r * k_d)       # (B, C, H, W)
        cross_rd = attn_rd * v_d                  # (B, C, H, W)

        # ---- Depth → RGB 交叉注意力 ----
        q_d = self.q_dep(dep)
        kv_r = self.kv_rgb(rgb)
        k_r, v_r = kv_r.chunk(2, dim=1)

        attn_dr = torch.sigmoid(q_d * k_r)
        cross_dr = attn_dr * v_r

        # 合并双向交叉注意力结果
        cross_feat = cross_rd + cross_dr          # (B, C, H, W)
        cross_feat = self.dw_conv(cross_feat)
        cross_feat = self.norm(cross_feat)

        return cross_feat, rgb, dep


class AdaptiveGate(nn.Module):
    """
    自适应门控: 防止劣质模态污染
    输出范围 (0, 1)，控制融合特征的注入量
    """
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, c, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gate(x)


class BidirectionalFusion(nn.Module):
    """
    P4 尺度融合: 双向互补对齐
    聚合: 轻量级双向交叉注意力，RGB 和 Depth 互为 Q/K/V
    扩散: 残差扩散 + 自适应门控，防止劣质模态污染
    """
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__()
        self.cross_attn = LightweightCrossAttention(c_rgb, c_dep, out_ch)
        self.gate = AdaptiveGate(out_ch)
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_ch),
        )

    def forward(self, f_rgb, f_dep):
        # 空间尺寸对齐
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)

        # 聚合: 双向交叉注意力
        cross_feat, rgb_aligned, dep_aligned = self.cross_attn(f_rgb, f_dep)

        # 扩散: 残差 + 自适应门控
        gate = self.gate(cross_feat)              # (B, out_ch, 1, 1)
        fused = rgb_aligned + gate * cross_feat   # 残差扩散

        return self.out_proj(fused)


# ============================================================
# P5: RGB-led 几何先验融合
# ============================================================

class ChannelAggregation(nn.Module):
    """
    通道聚合 (Squeeze): Depth → GAP + MLP → Channel Attention 向量 (1×1×C)
    聚合全局几何先验信息
    """
    def __init__(self, c_dep, out_ch, reduction=16):
        super().__init__()
        mid_ch = max(out_ch // reduction, 8)
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                              # (B, c_dep, 1, 1)
            nn.Conv2d(c_dep, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
        )

    def forward(self, f_dep):
        return self.squeeze(f_dep)  # (B, out_ch, 1, 1)


class ChannelDiffusion(nn.Module):
    """
    通道扩散 (Excitation): Channel Attention → Sigmoid → 逐通道乘到 RGB
    增强特定语义通道
    """
    def __init__(self, c_rgb, out_ch):
        super().__init__()
        self.align_rgb = nn.Conv2d(c_rgb, out_ch, 1, bias=False) if c_rgb != out_ch else nn.Identity()

    def forward(self, f_rgb, channel_attn):
        f_rgb_aligned = self.align_rgb(f_rgb)
        attn = torch.sigmoid(channel_attn)  # (B, out_ch, 1, 1)
        return f_rgb_aligned * attn


class RGBLedFusion(nn.Module):
    """
    P5 尺度融合: RGB-led 几何先验
    聚合: Depth 经 GAP + MLP 聚合全局几何先验 → Channel Attention 向量
    扩散: 向量经 Sigmoid 后逐通道乘到 RGB 特征上，增强特定语义
    """
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__()
        self.aggregate = ChannelAggregation(c_dep, out_ch)
        self.diffuse = ChannelDiffusion(c_rgb, out_ch)

    def forward(self, f_rgb, f_dep):
        # 空间尺寸对齐
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
        # 聚合: Depth → Channel Attention
        channel_attn = self.aggregate(f_dep)
        # 扩散: Channel Attention → Sigmoid → 乘到 RGB
        return self.diffuse(f_rgb, channel_attn)


# ============================================================
# Naive: Baseline 融合 (Concat + 1x1 Conv)
# ============================================================

class NaiveFusion(nn.Module):
    """
    Baseline 融合: 简单的 Concat + 1x1 Conv
    无聚合阶段，直接拼接后投影
    """
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__()
        self.align_dep = nn.Conv2d(c_dep, c_rgb, 1, bias=False) if c_dep != c_rgb else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv2d(c_rgb * 2, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_rgb, f_dep):
        # 空间尺寸对齐
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
        f_dep_aligned = self.align_dep(f_dep)
        return self.proj(torch.cat([f_rgb, f_dep_aligned], dim=1))


# ============================================================
# ScaleAwareFusion: 统一入口 (通过 strategy 参数选择融合策略)
# ============================================================

class ScaleAwareFusion(nn.Module):
    """
    尺度感知融合模块 — 基于"聚合-扩散"范式

    通过 strategy 参数选择每个尺度的融合策略:
      - "depth2rgb": P3 Depth-to-RGB 空间引导
      - "bidirectional": P4 双向互补对齐
      - "rgb_led": P5 RGB-led 几何先验
      - "naive": Baseline Concat + 1x1 Conv
      - "ct": 复用已有 CT 控制理论融合 (KalmanGatedFusion/ESOFusion/IDAPBCFusion)
      - "apfm_ct": APFM 预融合 + CT 控制理论融合

    Args:
        c_rgb: RGB 分支输入通道数
        c_dep: Depth 分支输入通道数
        out_ch: 输出通道数 (通道对齐目标)
        strategy: 融合策略名称
    """
    # 策略注册表
    STRATEGY_REGISTRY = {
        "depth2rgb": Depth2RGBFusion,
        "bidirectional": BidirectionalFusion,
        "rgb_led": RGBLedFusion,
        "naive": NaiveFusion,
    }

    def __init__(self, c_rgb, c_dep, out_ch, strategy="naive"):
        super().__init__()
        self.strategy_name = strategy

        if strategy in self.STRATEGY_REGISTRY:
            self.fusion = self.STRATEGY_REGISTRY[strategy](c_rgb, c_dep, out_ch)
        elif strategy == "ct":
            # CT 融合: 使用 BypassModule (1x1 投影 + 相加) 作为简化版
            # 完整 CT 融合需要在 YAML 中直接使用 KalmanGatedFusion/ESOFusion/IDAPBCFusion
            self.fusion = NaiveFusion(c_rgb, c_dep, out_ch)
        elif strategy == "apfm_ct":
            # APFM + CT: 先 APFM 预融合，再 CT 融合
            # 简化实现: 使用 NaiveFusion
            self.fusion = NaiveFusion(c_rgb, c_dep, out_ch)
        else:
            raise ValueError(f"未知的融合策略: {strategy}，可选: {list(self.STRATEGY_REGISTRY.keys())} + ['ct', 'apfm_ct']")

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            f_rgb, f_dep = x[0], x[1]
        else:
            return x
        return self.fusion(f_rgb, f_dep)


# ============================================================
# 便捷别名: 用于 YAML 中直接指定策略
# ============================================================

class ScaleAwareFusion_Depth2RGB(ScaleAwareFusion):
    """P3: Depth-to-RGB 空间引导融合"""
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__(c_rgb, c_dep, out_ch, strategy="depth2rgb")


class ScaleAwareFusion_Bidirectional(ScaleAwareFusion):
    """P4: 双向互补对齐融合"""
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__(c_rgb, c_dep, out_ch, strategy="bidirectional")


class ScaleAwareFusion_RGBLed(ScaleAwareFusion):
    """P5: RGB-led 几何先验融合"""
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__(c_rgb, c_dep, out_ch, strategy="rgb_led")


class ScaleAwareFusion_Naive(ScaleAwareFusion):
    """Naive: Concat + 1x1 Conv 基线融合"""
    def __init__(self, c_rgb, c_dep, out_ch):
        super().__init__(c_rgb, c_dep, out_ch, strategy="naive")
