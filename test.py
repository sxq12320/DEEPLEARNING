import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class APFM(nn.Module):
    """
    Attention Parallel Feature Mixer (APFM)

    同时建模 Channel-wise 和 Spatial-wise 注意力，
    对两路互补特征 FA, FB 进行自适应加权融合。

    Args:
        in_channels (int): 输入特征通道数 (FA 和 FB 相同)
        reduction (int): 通道压缩比，默认为 4
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()

        mid_channels = max(in_channels // reduction, 1)

        # ------------------------------------------------------------------
        # Channel Context 分支 (对应论文 Figure 2b 左侧)
        # GAP -> 1x1 Conv -> Norm -> SiLU -> 1x1 Conv -> Norm
        # ------------------------------------------------------------------
        self.channel_branch = nn.Sequential(
            # GAP 在 forward 中手动做，这里只放后续层
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),   # LayerNorm over channels
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
        )

        # ------------------------------------------------------------------
        # Spatial Context 分支 (对应论文 Figure 2b 右侧)
        # GMP -> 1x1 Conv -> Norm -> SiLU -> 1x1 Conv -> Norm
        # ------------------------------------------------------------------
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
        )

        # ------------------------------------------------------------------
        # 融合两路上下文后的 3x3 Conv (论文 Figure 2b 底部)
        # 将 channel + spatial 合并后映射回 in_channels
        # ------------------------------------------------------------------
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.SiLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # 最终 Sigmoid: 生成融合权重 w ∈ (0, 1)
        # Ff = w * FA + (1 - w) * FB
        # ------------------------------------------------------------------
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fa: 特征图 A, shape [B, C, H, W]
            fb: 特征图 B, shape [B, C, H, W]

        Returns:
            ff: 融合后特征图, shape [B, C, H, W]
        """
        assert fa.shape == fb.shape, f"FA {fa.shape} 与 FB {fb.shape} 形状不一致"

        # 拼接两路特征: [B, 2C, H, W]
        f_cat = torch.cat([fa, fb], dim=1)

        # ---- Channel Context ----
        # GAP: [B, 2C, H, W] -> [B, 2C, 1, 1]
        gap = F.adaptive_avg_pool2d(f_cat, 1)
        ch_ctx = self.channel_branch(gap)          # [B, C, 1, 1]

        # ---- Spatial Context ----
        # GMP: [B, 2C, H, W] -> [B, 2C, 1, 1]
        gmp = F.adaptive_max_pool2d(f_cat, 1)
        sp_ctx = self.spatial_branch(gmp)          # [B, C, 1, 1]

        # ---- 合并两路上下文 ----
        # 拼接 channel + spatial context: [B, 2C, 1, 1]
        ctx_cat = torch.cat([ch_ctx, sp_ctx], dim=1)

        # 经 3x3 Conv 融合 (广播到 [B, C, H, W])
        ctx_fused = self.fusion_conv(ctx_cat)       # [B, C, 1, 1]

        # ---- 生成权重并自适应融合 ----
        w = self.sigmoid(ctx_fused)                 # [B, C, 1, 1], 广播到 H×W

        ff = w * fa + (1.0 - w) * fb               # [B, C, H, W]
        return ff
    

model = APFM(3,4)
x = [torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)]
out = model(x[0], x[1])
print(out.shape)
print("====================================================")
print(x[0].shape)