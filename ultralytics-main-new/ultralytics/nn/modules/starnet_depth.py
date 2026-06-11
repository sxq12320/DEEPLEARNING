"""
StarNet Depth Branch — 用于替换 YOLO11 中的 Depth 主干
基于 StarNet (Rewrite the Stars, CVPR 2024) 架构

StarNet 核心思想: 星型操作 (Star Operation)
  Star(x) = (W₁·x) ⊙ (W₂·x)
  等价于将特征映射到高维隐空间，无需显式计算高维映射

StarNet 通道配置 (适配 RGB-D 检测):
  stem → 32ch (stride 4: P1/2 → P3/8)
  stage2 → 64ch (stride 2: P3/8 → P4/16)
  stage3 → 128ch (stride 2: P4/16 → P5/32)

空间尺寸 (640 输入, 从 layer 0 的 320×320 开始):
  stem → 80×80 (stride 4, P3/8)
  stage2 → 40×40 (stride 2, P4/16)
  stage3 → 20×20 (stride 2, P5/32)
"""

import torch
import torch.nn as nn


class StarBlock(nn.Module):
    """
    StarNet 基本模块: 星型操作块

    核心公式: Star(x) = (W₁·LayerNorm(x)) ⊙ (W₂·LayerNorm(x))
    等价于隐式高维特征映射，计算效率极高

    结构: LayerNorm → 1x1 Conv(split) → Star Operation → 1x1 Conv → Residual

    Args:
        dim: 特征通道数
        mlp_ratio: MLP 扩展比，默认 2.0
    """
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)

        # Star Operation: 将特征分为两路，分别线性变换后逐元素相乘
        # W₁ 和 W₂ 通过 1x1 Conv 实现
        hidden_dim = int(dim * mlp_ratio)
        self.w1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.w2 = nn.Conv2d(dim, hidden_dim, 1, bias=False)

        # 输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=dim),
        )

        # 可学习的缩放因子 (初始化为较小值，保证训练稳定性)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.01)

    def forward(self, x):
        identity = x
        x = self.norm1(x)

        # Star Operation: (W₁·x) ⊙ (W₂·x)
        x1 = self.w1(x)
        x2 = self.w2(x)
        star_out = x1 * x2

        # 投影 + 残差
        out = self.proj(star_out)
        return identity + self.gamma * out


class StarNetStem_Depth(nn.Module):
    """
    StarNet Depth 流 Stem 层
    将输入下采样 4 倍 (2个 stride=2 Conv)

    输入: (B, c_in, H, W) — 来自 layer 0 的特征图 (如 64ch, 320×320)
    输出: (B, out_ch, H/4, W/4) — 例如 320→80 (P3/8)
    """
    def __init__(self, c_in, out_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class StarNetStage(nn.Module):
    """
    StarNet Stage: 下采样 + 多个 StarBlock

    下采样方式: Patch Embedding (Conv stride=2 + LayerNorm)
    后接 num_blocks 个 StarBlock

    Args:
        inp_ch: 输入通道数
        out_ch: 输出通道数
        num_blocks: StarBlock 数量
        stride: 下采样步幅，默认 2
    """
    def __init__(self, inp_ch, out_ch, num_blocks, stride=2):
        super().__init__()
        # Patch Embedding (下采样)
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Conv2d(inp_ch, out_ch, 1, bias=False) if inp_ch != out_ch else nn.Identity()

        # Star Blocks
        self.blocks = nn.Sequential(*[StarBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.downsample(x)
        return self.blocks(x)
