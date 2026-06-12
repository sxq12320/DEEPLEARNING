"""
MobileNet V4-Conv RGB Branch — 用于替换 YOLO11 中的 RGB 主干
基于 MobileNetV4 (Google, 2024) 架构，采用 PyTorch 实现

论文: "MobileNetV4: Universal Models for the Mobile Ecosystem"
代码参考: https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py

核心创新:
  1. Universal Inverted Bottleneck (UIB) — 统一倒置瓶颈搜索块
     - IB: 标准 Inverted Bottleneck (1x1 expand → DW → 1x1 project)
     - ExtraDW: 双深度卷积 (DW → 1x1 expand → DW → 1x1 project)
     - ConvNext: ConvNext风格 (DW → 1x1 expand → 1x1 project)
     - FFN: 纯通道混合 (1x1 expand → 1x1 project, 无空间卷积)
  2. FusedIB — 融合倒置瓶颈 (1x1+3x3融合为3x3 → 1x1 project)
  3. Mobile MQA — 移动端优化的多查询注意力 (本实现暂不包含)

MobileNetV4-Conv-M 通道配置 (640输入):
  stem → 32ch (stride 2, P1/2)
  stage1 → 32ch (stride 2, P2/4) — FusedIB
  stage2 → 64ch (stride 2, P3/8) — FusedIB
  stage3 → 96ch (stride 2, P4/16) — ExtraDW+IB+ConvNext+ExtraDW
  extra → Conv s2 + SPPF (P5/32)

MobileNetV4-Conv-L 通道配置 (640输入):
  stem → 32ch (stride 2, P1/2)
  stage1 → 32ch (stride 2, P2/4) — FusedIB
  stage2 → 80ch (stride 2, P3/8) — FusedIB
  stage3 → 160ch (stride 2, P4/16) — ExtraDW+IB+ConvNext+ExtraDW+IB+ExtraDW
  extra → Conv s2 + SPPF (P5/32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    """确保通道数可被 divisor 整除 (MobileNet 标准做法)"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    """MobileNetV4 硬 Swish 激活: x * F.relu6(x + 3) / 6"""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HardSigmoid(nn.Module):
    """MobileNetV4 硬 Sigmoid 激活: F.relu6(x + 3) / 6"""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    """Squeeze-and-Excitation 模块 (使用 HardSigmoid)"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, _make_divisible(channel // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(_make_divisible(channel // reduction, 8), channel, 1, bias=False),
            HardSigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# ============================================================
# UIB (Universal Inverted Bottleneck) Block Implementations
# ============================================================

class UIBBlock(nn.Module):
    """
    Universal Inverted Bottleneck (UIB) 基础块

    UIB 统一了四种微架构:
      - IB (Inverted Bottleneck): expand → DW(k) → project
      - ExtraDW: DW(k1) → expand → DW(k2) → project
      - ConvNext: DW(k) → expand → project
      - FFN: expand → project (无DW)

    通过 start_dw_kernel_size 和 mid_dw_kernel_size 控制具体实例化:
      - IB: start_dw=0, mid_dw>0
      - ExtraDW: start_dw>0, mid_dw>0
      - ConvNext: start_dw>0, mid_dw=0
      - FFN: start_dw=0, mid_dw=0

    Args:
        inp: 输入通道数
        oup: 输出通道数
        stride: 步幅 (1 或 2)
        expand_ratio: 扩展比例
        start_dw_kernel_size: 起始 DW 卷积核大小 (0=不使用)
        mid_dw_kernel_size: 中间 DW 卷积核大小 (0=不使用)
        use_se: 是否使用 SE 模块
        use_hs: 是否使用 HardSwish (否则使用 ReLU)
    """
    def __init__(self, inp, oup, stride=1, expand_ratio=1.0,
                 start_dw_kernel_size=0, mid_dw_kernel_size=3,
                 use_se=False, use_hs=False):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.use_res_connect = self.identity

        mid_oup = _make_divisible(int(inp * expand_ratio), 8)
        act = HardSwish if use_hs else nn.ReLU

        layers = []

        # 1. 起始深度卷积 (ExtraDW / ConvNext)
        if start_dw_kernel_size > 0:
            layers.extend([
                nn.Conv2d(inp, inp, start_dw_kernel_size, stride=1,
                          padding=start_dw_kernel_size // 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                act(inplace=True) if mid_oup > inp else nn.Identity(),
            ])

        # 2. 扩展层 (1x1 Conv)
        if mid_oup != inp:
            layers.extend([
                nn.Conv2d(inp, mid_oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_oup),
                act(inplace=True),
            ])

        # 3. 中间深度卷积 (IB / ExtraDW)
        if mid_dw_kernel_size > 0:
            layers.extend([
                nn.Conv2d(mid_oup, mid_oup, mid_dw_kernel_size, stride,
                          padding=mid_dw_kernel_size // 2, groups=mid_oup, bias=False),
                nn.BatchNorm2d(mid_oup),
                act(inplace=True),
            ])
        else:
            # FFN 或 ConvNext 无中间 DW, 但仍需 stride 处理
            if stride == 2:
                # 无 DW 时通过池化实现下采样
                layers.append(nn.AvgPool2d(2, 2))

        # 4. SE 模块
        if use_se:
            layers.append(SEModule(mid_oup))

        # 5. 投影层 (1x1 Conv, 无激活)
        layers.extend([
            nn.Conv2d(mid_oup, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FusedIB(nn.Module):
    """
    Fused Inverted Bottleneck — 融合倒置瓶颈块
    将 1x1 expand + 3x3 DW 融合为单个 3x3 Conv，减少内存访问

    结构: 3x3 Conv(expanded) → SE(可选) → 1x1 Conv(project) + 残差连接

    Args:
        inp: 输入通道数
        oup: 输出通道数
        stride: 步幅 (1 或 2)
        expand_ratio: 扩展比例
        use_se: 是否使用 SE 模块
        use_hs: 是否使用 HardSwish
    """
    def __init__(self, inp, oup, stride=1, expand_ratio=1.0, use_se=False, use_hs=False):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.use_res_connect = self.identity

        mid_oup = _make_divisible(int(inp * expand_ratio), 8)
        act = HardSwish if use_hs else nn.ReLU

        layers = []
        # 融合的 3x3 Conv (expand + spatial)
        if mid_oup != inp:
            layers.extend([
                nn.Conv2d(inp, mid_oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(mid_oup),
                act(inplace=True),
            ])
        else:
            layers.extend([
                nn.Conv2d(inp, mid_oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(mid_oup),
                act(inplace=True),
            ])

        # SE 模块
        if use_se:
            layers.append(SEModule(mid_oup))

        # 投影层
        layers.extend([
            nn.Conv2d(mid_oup, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ============================================================
# MobileNetV4-Conv-M Block 配置表
# 格式: [block_type, k1, k2, expand_ratio, out_ch, stride, use_se, use_hs]
#   block_type: 'fused_ib', 'extra_dw', 'ib', 'convnext', 'ffn'
#   k1: start_dw_kernel_size (ExtraDW/ConvNext用)
#   k2: mid_dw_kernel_size (IB/ExtraDW用)
# ============================================================

MOBILENETV4_CONV_M_CONFIGS = [
    # Stage 1 (P2/4): FusedIB, stride 2
    ['fused_ib', 0, 0, 1.0, 32, 2, False, False],    # 0: 32→32, s=2

    # Stage 2 (P3/8): FusedIB, stride 2
    ['fused_ib', 0, 0, 3.0, 64, 2, False, False],    # 1: 32→64, s=2

    # Stage 3 (P4/16): ExtraDW + IB + ConvNext + ExtraDW
    ['extra_dw', 5, 5, 3.0, 96, 2, False, True],     # 2: 64→96, s=2, ExtraDW
    ['extra_dw', 5, 5, 6.0, 96, 1, False, True],     # 3: 96→96, s=1, ExtraDW
    ['ib',       0, 3, 6.0, 96, 1, False, True],      # 4: 96→96, s=1, IB
    ['convnext', 3, 0, 6.0, 96, 1, False, True],      # 5: 96→96, s=1, ConvNext
    ['extra_dw', 5, 5, 6.0, 96, 1, False, True],      # 6: 96→96, s=1, ExtraDW

    # Stage 4 (P5/32): ExtraDW + IB + ConvNext + ExtraDW + FFN
    ['extra_dw', 5, 5, 10.67, 128, 2, False, True],   # 7: 96→128, s=2, ExtraDW
    ['extra_dw', 5, 5, 6.0,  128, 1, False, True],    # 8: 128→128, s=1, ExtraDW
    ['ib',       0, 3, 6.0,  128, 1, False, True],     # 9: 128→128, s=1, IB
    ['convnext', 3, 0, 6.0,  128, 1, False, True],     # 10: 128→128, s=1, ConvNext
    ['extra_dw', 5, 5, 6.0,  128, 1, False, True],     # 11: 128→128, s=1, ExtraDW
    ['ffn',      0, 0, 6.0,  128, 1, False, True],     # 12: 128→128, s=1, FFN
]

MOBILENETV4_CONV_L_CONFIGS = [
    # Stage 1 (P2/4): FusedIB, stride 2
    ['fused_ib', 0, 0, 1.0, 32, 2, False, False],    # 0: 32→32, s=2

    # Stage 2 (P3/8): FusedIB, stride 2
    ['fused_ib', 0, 0, 4.0, 80, 2, False, False],    # 1: 32→80, s=2

    # Stage 3 (P4/16): ExtraDW + IB + ConvNext + ExtraDW + IB + ExtraDW
    ['extra_dw', 5, 5, 6.0,  160, 2, False, True],    # 2: 80→160, s=2
    ['extra_dw', 5, 5, 6.0,  160, 1, False, True],     # 3: 160→160, s=1
    ['ib',       0, 3, 6.0,  160, 1, False, True],      # 4: 160→160, s=1
    ['convnext', 3, 0, 6.0,  160, 1, False, True],      # 5: 160→160, s=1
    ['extra_dw', 5, 5, 6.0,  160, 1, False, True],      # 6: 160→160, s=1
    ['ib',       0, 3, 6.0,  160, 1, False, True],       # 7: 160→160, s=1
    ['extra_dw', 5, 5, 6.0,  160, 1, False, True],       # 8: 160→160, s=1

    # Stage 4 (P5/32): ExtraDW + IB + ConvNext + ExtraDW + IB + ExtraDW + FFN
    ['extra_dw', 5, 5, 6.0,  256, 2, False, True],     # 9: 160→256, s=2
    ['extra_dw', 5, 5, 6.0,  256, 1, False, True],      # 10: 256→256, s=1
    ['ib',       0, 3, 6.0,  256, 1, False, True],       # 11: 256→256, s=1
    ['convnext', 3, 0, 6.0,  256, 1, False, True],       # 12: 256→256, s=1
    ['extra_dw', 5, 5, 6.0,  256, 1, False, True],       # 13: 256→256, s=1
    ['ib',       0, 3, 6.0,  256, 1, False, True],        # 14: 256→256, s=1
    ['extra_dw', 5, 5, 6.0,  256, 1, False, True],        # 15: 256→256, s=1
    ['ffn',      0, 0, 6.0,  256, 1, False, True],        # 16: 256→256, s=1
]


def _build_uib_block(block_type, inp, oup, stride, expand_ratio,
                     k1, k2, use_se, use_hs):
    """根据 block_type 构建 UIB 实例化块"""
    if block_type == 'fused_ib':
        return FusedIB(inp, oup, stride, expand_ratio, use_se, use_hs)
    elif block_type == 'extra_dw':
        return UIBBlock(inp, oup, stride, expand_ratio,
                        start_dw_kernel_size=k1, mid_dw_kernel_size=k2,
                        use_se=use_se, use_hs=use_hs)
    elif block_type == 'ib':
        return UIBBlock(inp, oup, stride, expand_ratio,
                        start_dw_kernel_size=0, mid_dw_kernel_size=k2,
                        use_se=use_se, use_hs=use_hs)
    elif block_type == 'convnext':
        return UIBBlock(inp, oup, stride, expand_ratio,
                        start_dw_kernel_size=k1, mid_dw_kernel_size=0,
                        use_se=use_se, use_hs=use_hs)
    elif block_type == 'ffn':
        return UIBBlock(inp, oup, stride, expand_ratio,
                        start_dw_kernel_size=0, mid_dw_kernel_size=0,
                        use_se=use_se, use_hs=use_hs)
    else:
        raise ValueError(f"未知的 block_type: {block_type}")


class MobileNetV4Stem_RGB(nn.Module):
    """
    MobileNetV4 RGB Stem 层
    将输入下采样 2 倍 (stride=2 Conv)

    输入: (B, c_in, H, W) — RGB 或 RGBD 图像
    输出: (B, 32, H/2, W/2) — 例如 640→320
    """
    def __init__(self, c_in, out_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            HardSwish(),
        )

    def forward(self, x):
        return self.stem(x)


class MobileNetV4Stage(nn.Module):
    """
    MobileNetV4 Stage (包含多个 UIB/FusedIB blocks)

    根据 out_ch 自动选择标准配置中的 blocks:
      out_ch=32  → Stage 1 (1 FusedIB, stride=2) — P2/4
      out_ch=64  → Stage 2 (1 FusedIB, stride=2) — P3/8 (Conv-M)
      out_ch=80  → Stage 2 (1 FusedIB, stride=2) — P3/8 (Conv-L)
      out_ch=96  → Stage 3 (5 UIB blocks, stride=2) — P4/16 (Conv-M)
      out_ch=160 → Stage 3 (7 UIB blocks, stride=2) — P4/16 (Conv-L)
      out_ch=128 → Stage 4 (6 UIB blocks, stride=2) — P5/32 (Conv-M)
      out_ch=256 → Stage 4 (8 UIB blocks, stride=2) — P5/32 (Conv-L)

    Args:
        inp_ch: 输入通道数
        out_ch: 输出通道数
    """
    # 各 stage 对应的 block 配置索引范围
    STAGE_BLOCKS_CONV_M = {
        32:  (0, 1),    # Stage 1: 1 FusedIB
        64:  (1, 2),    # Stage 2: 1 FusedIB
        96:  (2, 7),    # Stage 3: 5 blocks (ExtraDW+ExtraDW+IB+ConvNext+ExtraDW)
        128: (7, 13),   # Stage 4: 6 blocks (ExtraDW+ExtraDW+IB+ConvNext+ExtraDW+FFN)
    }

    STAGE_BLOCKS_CONV_L = {
        32:  (0, 1),    # Stage 1: 1 FusedIB
        80:  (1, 2),    # Stage 2: 1 FusedIB
        160: (2, 9),    # Stage 3: 7 blocks
        256: (9, 17),   # Stage 4: 8 blocks
    }

    def __init__(self, inp_ch, out_ch, variant='conv_m'):
        super().__init__()
        if variant == 'conv_m':
            stage_blocks = self.STAGE_BLOCKS_CONV_M
            configs = MOBILENETV4_CONV_M_CONFIGS
        elif variant == 'conv_l':
            stage_blocks = self.STAGE_BLOCKS_CONV_L
            configs = MOBILENETV4_CONV_L_CONFIGS
        else:
            raise ValueError(f"不支持的 variant: {variant}，可选: 'conv_m', 'conv_l'")

        if out_ch in stage_blocks:
            start_idx, end_idx = stage_blocks[out_ch]
            block_configs = configs[start_idx:end_idx]
        else:
            raise ValueError(
                f"MobileNetV4Stage({variant}): 不支持的 out_ch={out_ch}，"
                f"可选: {list(stage_blocks.keys())}"
            )

        layers = []
        current_ch = inp_ch
        for block_type, k1, k2, expand_ratio, oup, stride, use_se, use_hs in block_configs:
            layers.append(_build_uib_block(
                block_type, current_ch, oup, stride, expand_ratio,
                k1, k2, use_se, use_hs
            ))
            current_ch = oup
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)
