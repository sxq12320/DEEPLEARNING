"""
ShuffleNetV2 Depth Branch — 用于替换 YOLO11-CT 中的 Depth 流卷积层
基于 ShuffleNetV2 1.0x (Megvii) 架构，采用 torchvision 风格的正确实现

ShuffleNetV2 1.0x 通道配置:
  stem → 24ch
  stage2 → 116ch (4 blocks)
  stage3 → 232ch (8 blocks)
  stage4 → 464ch (4 blocks)

空间尺寸 (640 输入):
  stem → 160×160 (stride 4)
  stage2 → 80×80  (stride 8,  P3/8)
  stage3 → 40×40  (stride 16, P4/16)
  stage4 → 20×20  (stride 32, P5/32)
"""

import torch
import torch.nn as nn


class ShuffleV2Block(nn.Module):
    """
    ShuffleNetV2 基本单元 (torchvision 风格实现)

    stride=1: 通道分割 → 主分支处理一半 → 拼接 → 通道混洗
    stride=2: 双分支并行(下采样) → 拼接 → 通道混洗
    """

    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        branch_features = oup // 2

        if stride == 2:
            # 下采样分支: DWConv(stride=2) → 1×1 Conv
            self.branch_proj = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=2, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch_proj = None

        # 主分支: 1×1 Conv → DWConv → 1×1 Conv
        self.branch_main = nn.Sequential(
            nn.Conv2d(
                inp if stride == 2 else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=branch_features,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch_main(x2)), dim=1)
        else:
            out = torch.cat((self.branch_proj(x), self.branch_main(x)), dim=1)
        out = self._channel_shuffle(out, 2)
        return out

    @staticmethod
    def _channel_shuffle(x, groups):
        """标准通道混洗操作"""
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1, H, W)
        return x


class ShuffleV2Stem_Depth(nn.Module):
    """
    ShuffleNetV2 Depth 流 Stem 层
    将 1 通道深度图下采样 4 倍 (stride=2 Conv + MaxPool)

    输入: (B, 1, H, W) — 单通道深度图
    输出: (B, 24, H/4, W/4) — 例如 640→160
    """

    def __init__(self, c_in, out_ch=24):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(c_in, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        return x


class ShuffleV2Stage(nn.Module):
    """
    ShuffleNetV2 完整 Stage
    第一个 block stride=2 (下采样)，后续 block stride=1

    Args:
        inp_ch: 输入通道数 (来自上一 stage 或 stem)
        out_ch: 输出通道数 (116/232/464 for 1.0x)
        num_blocks: block 数量 (4/8/4 for 1.0x)
    """

    def __init__(self, inp_ch, out_ch, num_blocks):
        super().__init__()
        layers = []
        # 第一个 block: stride=2 下采样
        layers.append(ShuffleV2Block(inp_ch, out_ch, stride=2))
        # 后续 blocks: stride=1 特征提取
        for _ in range(num_blocks - 1):
            layers.append(ShuffleV2Block(out_ch, out_ch, stride=1))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)
