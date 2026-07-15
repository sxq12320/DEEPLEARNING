# Ultralytics LSNet module
# LSNet: See Large, Focus Small (CVPR 2025)
# https://github.com/THU-MIG/lsnet / https://arxiv.org/abs/2503.23135
#
# Integrated as a standalone backbone module for the Ultralytics framework.
# Provides the full LSNet architecture: LSConv, LSBlock, and LSNetBackbone variants (T/S/B).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "LSConv",
    "LSBlock",
    "LSNet_T",
    "LSNet_S",
    "LSNet_B",
    "LSNetBackbone",
)

# ---------------------------------------------------------------------------
#  Core LS Convolution
# ---------------------------------------------------------------------------

class LSConv(nn.Module):
    """Large-Small Convolution (CVPR 2025 LSNet).

    Decouples the perception range from the aggregation range:
      1. Large-Kernel Perception (LKP) — captures broad spatial context via a
         bottleneck with a large depthwise conv (K_L), producing spatial weights.
      2. Small-Kernel Aggregation (SKA) — applies a small depthwise conv (K_S)
         whose output is gated by the LKP weights.
      3. Projection — 1x1 conv (BN + SiLU) to mix channels and resize.

    Args:
        c1: input channels
        c2: output channels
        k_l: large-kernel size for perception (default 7)
        k_s: small-kernel size for aggregation (default 3)
        reduction: channel reduction ratio for the LKP bottleneck (default 4)
    """

    def __init__(self, c1, c2, k_l=7, k_s=3, reduction=4):
        super().__init__()
        mid = max(c1 // reduction, 16)

        # Large-Kernel Perception
        self.lkp_pw1 = nn.Conv2d(c1, mid, 1, bias=False)
        self.lkp_bn = nn.BatchNorm2d(mid)
        self.lkp_dw = nn.Conv2d(mid, mid, k_l, padding=k_l // 2, groups=mid, bias=False)
        self.lkp_pw2 = nn.Conv2d(mid, c1, 1, bias=False)
        self.act = nn.Sigmoid()

        # Small-Kernel Aggregation
        self.ska_dw = nn.Conv2d(c1, c1, k_s, padding=k_s // 2, groups=c1, bias=False)
        self.ska_bn = nn.BatchNorm2d(c1)

        # Projection
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        w = self.lkp_pw1(x)
        w = self.lkp_bn(w)
        w = F.relu(w)
        w = self.lkp_dw(w)
        w = self.lkp_pw2(w)
        w = self.act(w)  # [B, C, H, W]

        x_s = self.ska_dw(x)
        x_s = self.ska_bn(x_s)
        x_s = F.relu(x_s)

        return self.proj(x_s * w)


# ---------------------------------------------------------------------------
#  LS Block  (transformer-style: token-mixer + channel-mixer + residuals)
# ---------------------------------------------------------------------------

class LSBlock(nn.Module):
    """LS Block: LSConv token-mixer + DW-SE local bias + FFN channel-mixer.

    Follows the LSNet paper design:
        x = x + LSConv(x)           # token mixing (large-small)
        x = x + DWConv(x) * SE(x)   # local inductive bias
        x = x + FFN(x)              # channel mixing

    Args:
        dim: input / output channels (same)
        k_l: large-kernel size for LSConv
        k_s: small-kernel size for LSConv
        mlp_ratio: FFN expansion ratio (default 2.0)
        reduction: LSConv bottleneck reduction ratio
        dw_kernel: kernel size for the local DW-SE branch
        se_reduction: reduction ratio for the SE module
    """

    def __init__(
        self,
        c1,
        c2,
        k_l=7,
        k_s=3,
        mlp_ratio=2.0,
        reduction=4,
        dw_kernel=5,
        se_reduction=4,
    ):
        super().__init__()
        # c1, c2 must be equal for residual LSBlock
        # Token mixer
        self.token_mixer = LSConv(c1, c1, k_l=k_l, k_s=k_s, reduction=reduction)

        # Local bias: DWConv + SE-style channel attention
        self.dw = nn.Conv2d(c1, c1, dw_kernel, padding=dw_kernel // 2, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        se_mid = max(c1 // se_reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(c1, se_mid, bias=False),
            nn.ReLU(),
            nn.Linear(se_mid, c1, bias=False),
            nn.Sigmoid(),
        )

        # Channel mixer (FFN)
        mlp_hidden = int(c1 * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(c1, mlp_hidden, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden),
            nn.SiLU(),
            nn.Conv2d(mlp_hidden, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        # Token mixing
        x = x + self.token_mixer(x)

        # Local bias: DW + SE
        residual = x
        x = self.dw(x)
        x = self.dw_bn(x)
        x = F.relu(x)
        b, c, _, _ = x.shape
        se_w = self.se_fc(self.se_pool(x).view(b, c)).view(b, c, 1, 1)
        x = residual + x * se_w

        # Channel mixing
        x = x + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
#  Stem  (overlapping patch embedding)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Overlapping patch embedding (LSNet stem).

    Projects raw pixels to the first-stage feature map (H/4, W/4).

    Args:
        c1: input image channels (default 3)
        c2: output embedding dimension
        patch_size: patch size (only 4 supported — H/4, W/4)
    """

    def __init__(self, c1, c2, patch_size=4):
        super().__init__()
        assert patch_size == 4, "LSNet stem always outputs H/4 feature maps"
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU(),
            nn.Conv2d(c2 // 2, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
#  Downsample  (DW + PW between stages)
# ---------------------------------------------------------------------------

class LSNetDownsample(nn.Module):
    """Downsampling layer used between LSNet stages.

    Depthwise conv (stride 2) → Pointwise conv.

    Args:
        c1: input channels
        c2: output channels
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, stride=2, padding=1, groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.pw = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_bn(x)
        x = F.relu(x)
        return self.pw(x)


# ---------------------------------------------------------------------------
#  Fourth-stage MSA block  (replaces LSBlock at the lowest resolution)
# ---------------------------------------------------------------------------

class MSAStageBlock(nn.Module):
    """Lightweight MSA block for the final (H/32) stage.

    Uses a simplified multi-head self-attention suitable for the lowest
    resolution where token count is small enough for full attention.

    Args:
        c1: input channels
        c2: output channels (must equal c1 for residual)
        num_heads: number of attention heads
        mlp_ratio: FFN expansion ratio
    """

    def __init__(self, c1, c2, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        head_dim = c1 // num_heads
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv = nn.Conv2d(c1, c1 * 3, 1, bias=False)
        self.proj = nn.Conv2d(c1, c1, 1, bias=False)

        mlp_hidden = int(c1 * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(c1, mlp_hidden, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden),
            nn.SiLU(),
            nn.Conv2d(mlp_hidden, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Self-attention
        shortcut = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, nh, hd, N]
        attn = (q.transpose(-2, -1) @ k) * self.scale  # [B, nh, N, N]
        attn = attn.softmax(dim=-1)
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [B, nh, N, hd]
        x = x.reshape(B, C, H, W)
        x = self.proj(x) + shortcut

        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x) + shortcut
        return x


# ---------------------------------------------------------------------------
#  Full LSNet Backbone
# ---------------------------------------------------------------------------

class LSNetBackbone(nn.Module):
    """LSNet backbone — See Large, Focus Small (CVPR 2025).

    A 4-stage hierarchical backbone that decouples perception and aggregation
    ranges. Stages 1–3 use LS blocks; stage 4 uses lightweight MSA blocks.

    Outputs multi-scale feature maps at:
        - P3: H/8  (stage 2 output)
        - P4: H/16 (stage 3 output)
        - P5: H/32 (stage 4 output)

    Args:
        depths: number of LSBlocks in stages 1-3 and MSABlocks in stage 4
                e.g. [2, 2, 4, 2] for LSNet-T
        channels: output channels for each stage
                  e.g. [64, 128, 256, 512] for LSNet-T
        in_chans: input image channels
        k_l: large kernel size for LSConv
        k_s: small kernel size for LSConv
        mlp_ratio: FFN expansion ratio in LSBlock / MSABlock
        num_heads: attention heads for the 4th stage
        drop_path_rate: stochastic depth rate (unused, kept for API compatibility)
        out_indices: which stage outputs to return (default [1,2,3] for P3/P4/P5)
    """

    def __init__(
        self,
        depths=(2, 2, 4, 2),
        channels=(64, 128, 256, 512),
        in_chans=3,
        k_l=7,
        k_s=3,
        mlp_ratio=2.0,
        num_heads=8,
        drop_path_rate=0.0,
        out_indices=(1, 2, 3),
    ):
        super().__init__()
        self.out_indices = out_indices
        self.channels = channels
        self.depths = depths

        # Stem  ->  H/4
        self.patch_embed = PatchEmbed(in_chans, channels[0])

        # Stage 1  ->  H/4,  LS blocks
        self.stage1 = self._make_ls_stage(channels[0], depths[0], k_l, k_s, mlp_ratio)
        self.down1 = LSNetDownsample(channels[0], channels[1])

        # Stage 2  ->  H/8,  LS blocks
        self.stage2 = self._make_ls_stage(channels[1], depths[1], k_l, k_s, mlp_ratio)
        self.down2 = LSNetDownsample(channels[1], channels[2])

        # Stage 3  ->  H/16,  LS blocks
        self.stage3 = self._make_ls_stage(channels[2], depths[2], k_l, k_s, mlp_ratio)
        self.down3 = LSNetDownsample(channels[2], channels[3])

        # Stage 4  ->  H/32,  MSA blocks
        self.stage4 = self._make_msa_stage(channels[3], depths[3], num_heads, mlp_ratio)

        self._init_weights()

    @staticmethod
    def _make_ls_stage(dim, depth, k_l, k_s, mlp_ratio):
        return nn.Sequential(*[
            LSBlock(dim, dim, k_l=k_l, k_s=k_s, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    @staticmethod
    def _make_msa_stage(dim, depth, num_heads, mlp_ratio):
        return nn.Sequential(*[
            MSAStageBlock(dim, dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Returns:
            list of feature maps at out_indices. Default [P3, P4, P5].
        """
        outs = []

        x = self.patch_embed(x)     # H/4
        x = self.stage1(x)          # H/4
        # Stage 1 output is NOT returned by default (too large, not used by YOLO neck)
        if 0 in self.out_indices:
            outs.append(x)

        x = self.down1(x)           # H/8
        x = self.stage2(x)          # H/8  ->  P3
        if 1 in self.out_indices:
            outs.append(x)

        x = self.down2(x)           # H/16
        x = self.stage3(x)          # H/16  ->  P4
        if 2 in self.out_indices:
            outs.append(x)

        x = self.down3(x)           # H/32
        x = self.stage4(x)          # H/32  ->  P5
        if 3 in self.out_indices:
            outs.append(x)

        return outs


# ---------------------------------------------------------------------------
#  Pre-configured variants
# ---------------------------------------------------------------------------

def LSNet_T(in_chans=3, **kwargs):
    """LSNet-Tiny (~11.4M params, 0.3G FLOPs @ 224²)."""
    return LSNetBackbone(
        depths=(2, 2, 4, 2),
        channels=(64, 128, 256, 512),
        in_chans=in_chans,
        **kwargs,
    )


def LSNet_S(in_chans=3, **kwargs):
    """LSNet-Small (~16.1M params, 0.5G FLOPs @ 224²)."""
    return LSNetBackbone(
        depths=(3, 3, 6, 2),
        channels=(64, 128, 256, 512),
        in_chans=in_chans,
        **kwargs,
    )


def LSNet_B(in_chans=3, **kwargs):
    """LSNet-Base (~23.2M params, 1.3G FLOPs @ 224²)."""
    return LSNetBackbone(
        depths=(4, 4, 8, 2),
        channels=(80, 160, 320, 640),
        in_chans=in_chans,
        **kwargs,
    )
