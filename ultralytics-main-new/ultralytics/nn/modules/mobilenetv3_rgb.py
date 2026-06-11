"""
MobileNet V3-Large RGB Branch — 用于替换 YOLO11 中的 RGB 主干
基于 MobileNetV3-Large (Google, 2019) 架构，采用标准实现

MobileNetV3-Large 通道配置 (标准):
  stem → 16ch
  stage1 → 24ch (2 blocks, stride 2)
  stage2 → 40ch (3 blocks, stride 2, SE) ← P3/8
  stage3 → 80→112→160ch (7 blocks) → 160ch (stride 2) ← P4/16
  extra → Conv s2 + SPPF ← P5/32

空间尺寸 (640 输入):
  stem → 320×320 (stride 2)
  stage1 → 160×160 (stride 4, P2/4)
  stage2 → 80×80 (stride 8, P3/8)
  stage3 → 40×40 (stride 16, P4/16)
  extra → 20×20 (stride 32, P5/32)
"""

import torch
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    """确保通道数可被 divisor 整除 (MobileNet 标准做法)"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    """MobileNetV3 硬 Swish 激活: x * F.relu6(x + 3) / 6"""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HardSigmoid(nn.Module):
    """MobileNetV3 硬 Sigmoid 激活: F.relu6(x + 3) / 6"""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


import torch.nn.functional as F


class SEModule(nn.Module):
    """MobileNetV3 Squeeze-and-Excitation 模块 (使用 HardSigmoid)"""
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


class MobileNetV3InvertedResidual(nn.Module):
    """
    MobileNetV3 倒残差块 (Inverted Residual Block)
    结构: 1x1 扩展 → DWConv → SE(可选) → 1x1 投影 + 残差连接

    Args:
        inp: 输入通道数
        oup: 输出通道数
        stride: 步幅 (1 或 2)
        expand_size: 扩展层通道数
        kernel_size: DWConv 核大小
        use_se: 是否使用 SE 模块
        use_hs: 是否使用 HardSwish (否则使用 ReLU)
    """
    def __init__(self, inp, oup, stride, expand_size, kernel_size=3, use_se=False, use_hs=False):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.use_res_connect = self.identity

        # 激活函数选择
        act = HardSwish if use_hs else nn.ReLU

        layers = []
        # 扩展层 (1x1 Conv)
        if expand_size != inp:
            layers.extend([
                nn.Conv2d(inp, expand_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand_size),
                act(inplace=True),
            ])
        # 深度可分离卷积
        padding = kernel_size // 2
        layers.extend([
            nn.Conv2d(expand_size, expand_size, kernel_size, stride, padding, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            act(inplace=True),
        ])
        # SE 模块
        if use_se:
            layers.append(SEModule(expand_size))
        # 投影层 (1x1 Conv, 无激活)
        layers.extend([
            nn.Conv2d(expand_size, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# MobileNetV3-Large 标准配置表
# 格式: [kernel, expand_size, out_channels, use_se, use_hs, stride]
MOBILENETV3_LARGE_CONFIGS = [
    # Stage 1 (P2/4): 2 blocks, stride 2
    [3, 16, 16, True, False, 1],    # block 0: 16→16, s=1, SE
    [3, 64, 24, False, False, 2],   # block 1: 16→24, s=2
    [3, 72, 24, False, False, 1],   # block 2: 24→24, s=1
    # Stage 2 (P3/8): 3 blocks, stride 2, SE
    [5, 40, 40, True, False, 2],    # block 3: 24→40, s=2, SE
    [5, 120, 40, True, False, 1],   # block 4: 40→40, s=1, SE
    [5, 240, 40, True, False, 1],   # block 5: 40→40, s=1, SE
    # Stage 3 (P4/16): 7 blocks
    [3, 80, 80, False, True, 1],    # block 6: 40→80, s=1, HS
    [3, 200, 80, False, True, 1],   # block 7: 80→80, s=1, HS
    [3, 184, 80, False, True, 1],   # block 8: 80→80, s=1, HS
    [3, 184, 80, False, True, 1],   # block 9: 80→80, s=1, HS
    [3, 480, 112, True, True, 1],   # block 10: 80→112, s=1, SE, HS
    [3, 672, 112, True, True, 1],   # block 11: 112→112, s=1, SE, HS
    [5, 672, 160, True, True, 2],   # block 12: 112→160, s=2, SE, HS
    [5, 960, 160, True, True, 1],   # block 13: 160→160, s=1, SE, HS
    [5, 960, 160, True, True, 1],   # block 14: 160→160, s=1, SE, HS
]


class MobileNetV3Stem_RGB(nn.Module):
    """
    MobileNetV3-Large RGB Stem 层
    将输入下采样 2 倍 (stride=2 Conv)

    输入: (B, c_in, H, W) — RGB 或 RGBD 图像
    输出: (B, 16, H/2, W/2) — 例如 640→320
    """
    def __init__(self, c_in, out_ch=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            HardSwish(),
        )

    def forward(self, x):
        return self.stem(x)


class MobileNetV3Stage(nn.Module):
    """
    MobileNetV3-Large Stage (包含多个 InvertedResidual blocks)

    根据 out_ch 自动选择标准配置中的 blocks:
      out_ch=24  → Stage 1 (2 blocks, 最后一个 stride=2)
      out_ch=40  → Stage 2 (3 blocks, 最后一个 stride=2) ← P3/8
      out_ch=160 → Stage 3 (9 blocks, stride=2 at block 12) ← P4/16

    Args:
        inp_ch: 输入通道数
        out_ch: 输出通道数 (24/40/160)
        stride: 首个下采样 block 的 stride (通常由配置自动决定)
    """
    # 各 stage 对应的 block 配置索引范围
    STAGE_BLOCKS = {
        24: (1, 3),    # blocks 1-2: 16→24, stride=2 at block 1
        40: (3, 6),    # blocks 3-5: 24→40, stride=2 at block 3
        160: (6, 15),  # blocks 6-14: 40→80→112→160, stride=2 at block 12
    }

    def __init__(self, inp_ch, out_ch, num_blocks=None):
        super().__init__()
        # 获取该 stage 的 block 配置
        if out_ch in self.STAGE_BLOCKS:
            start_idx, end_idx = self.STAGE_BLOCKS[out_ch]
            block_configs = MOBILENETV3_LARGE_CONFIGS[start_idx:end_idx]
        else:
            raise ValueError(f"MobileNetV3Stage: 不支持的 out_ch={out_ch}，可选: 24, 40, 160")

        layers = []
        current_ch = inp_ch
        for k, exp_size, oup, use_se, use_hs, stride in block_configs:
            layers.append(MobileNetV3InvertedResidual(current_ch, oup, stride, exp_size, k, use_se, use_hs))
            current_ch = oup
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)
