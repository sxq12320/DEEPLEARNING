"""
RGBD Dual-Branch Fusion Neck — Redesigned Architecture
=====================================================

Amodal segmentation using RGBD input with asymmetric dual branches:
  - RGB branch:   MobileNetV2 backbone → P3 [B,24,80,80], P4 [B,32,40,40], P5 [B,96,20,20]
  - Depth branch: StarNet-S4 backbone  → N3 [B,64,80,80], N4 [B,128,40,40], N5 [B,256,20,20]

Three-scale fusion strategy with principled mechanism assignment:
  P3 (80x80): Depth guides RGB, LOCAL mechanism (spatial modulation + deformable alignment)
  P4 (40x40): Bidirectional, soft channel (ECA) + spatial attention gating
  P5 (20x20): RGB guides Depth, CROSS-ATTENTION (correct placement, tractable O(N^2))

FPN top-down decoder merges all six feature maps into F3 at 80x80.

Key design decisions fixing the original architecture's problems:
  Problem 1 — Corrected attention formula: softmax(QK^T / sqrt(d_k)) * V
  Problem 2 — Channel alignment first, spatial alignment second
  Problem 3 — Cross-attention moved from P3 to P5 (256x cheaper, semantically richer)
  Problem 4 — P3 uses local spatial modulation instead of global attention
  Problem 5 — Soft (sigmoid) channel weighting, NOT hard binary removal
  Problem 6 — FPN-style top-down decoder with explicit aggregation topology
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "ScaleAlignBlock",
    "P3FusionModule",
    "P4FusionModule",
    "P5FusionModule",
    "FusionDecoder",
    "RGBDFusionNeck",
)


# ============================================================================
# Part 1 — Channel and Spatial Alignment
# ============================================================================
# Motivation: All six input feature maps have different channel dimensions
# (RGB: 24/32/96, Depth: 64/128/256). Before any fusion, they must be
# projected to a unified channel count C_unified. Channel compression via
# 1x1 conv is applied FIRST (Problem 2 fix), then spatial resampling if needed.
#
# C_unified = 128 justification:
#   - MobileNetV2 outputs: 24, 32, 96  → max 96, 128 is a modest upscale
#   - StarNet-S4 outputs:  64, 128, 256 → 128 is the natural midpoint
#   - 128 balances representational capacity and computational cost
#   - Divisible by 4 (needed for multi-head attention with heads=4 or 8)
#
# For same-scale pairs (P3/N3, P4/N4, P5/N5): channel align only, no spatial change.
# For cross-scale decoder paths: bilinear interpolation (smooth, no learnable params,
# avoids checkerboard artifacts of strided conv transpose).
# AdaptiveAvgPool2d is NOT used for downsampling here because all same-scale pairs
# already share spatial dimensions. Bilinear upsampling is used for the decoder's
# top-down path. If downsampling were needed, AdaptiveAvgPool2d would be preferred
# over strided conv for its parameter-free nature and anti-aliasing properties.


class ScaleAlignBlock(nn.Module):
    """Channel and spatial alignment block.

    Applies 1x1 conv channel projection FIRST (cheaper on high-channel tensors),
    then optional spatial resampling via bilinear interpolation.

    Args:
        c_in (int): Input channel count.
        c_unified (int): Target unified channel count.
        target_size (tuple[int, int] | None): Target spatial size (H, W).
            If None, no spatial resampling is applied.
    """

    def __init__(self, c_in: int, c_unified: int = 128, target_size: tuple | None = None):
        super().__init__()
        # Channel alignment: 1x1 conv + BN + ReLU
        self.channel_proj = nn.Sequential(
            nn.Conv2d(c_in, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, c_in, H, W]
        x = self.channel_proj(x)  # [B, C_unified, H, W]
        if self.target_size is not None and x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        # [B, C_unified, H', W']
        return x


# ============================================================================
# Part 2 — P3 Fusion (fine-scale 80x80, Depth→RGB, LOCAL mechanism)
# ============================================================================
# Motivation: P3 features are spatially fine but semantically weak. Cross-attention
# at 80x80 (N=6400) costs O(N^2)=40,960,000 — 256x more expensive than P5.
# Depth boundary cues are spatially precise and LOCAL, not global. The right
# mechanism is spatial modulation + optional deformable alignment.
#
# Path A — Spatial modulation (primary):
#   N3 → Conv3x3 → BN → ReLU → Conv1x1 → Sigmoid → spatial mask
#   → element-wise multiply onto P3
#
# Path B — Deformable local alignment (secondary, sub-pixel misalignment):
#   P3 → Conv3x3 → offset + modulation fields
#   → F.grid_sample on N3 (fallback from torchvision.ops.deform_conv2d)
#
# Combine: add both paths, split into dual-stream output.


class P3FusionModule(nn.Module):
    """P3 fusion: Depth guides RGB via LOCAL spatial modulation + deformable alignment.

    NO cross-attention here. See Problem 3 justification above.
    At 80x80 (N=6400), cross-attention O(N^2)=40,960,000 is prohibitive.
    Depth boundary cues are local — spatial modulation is the correct mechanism.

    Args:
        c_unified (int): Unified channel count (default 128).
        deform_k (int): Deformable sampling kernel size (default 3).
    """

    def __init__(self, c_unified: int = 128, deform_k: int = 3):
        super().__init__()
        self.c_unified = c_unified
        self.deform_k = deform_k

        # Path A — Spatial modulation: N3 → spatial mask → multiply onto P3
        self.spatial_mask_head = nn.Sequential(
            nn.Conv2d(c_unified, c_unified, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=True),
            nn.Sigmoid(),  # spatial mask in [0, 1]
        )

        # Path B — Deformable local alignment via F.grid_sample
        # P3 predicts offset field for sampling N3
        # offset: [B, 2*k*k, H, W] for kx3 deformable kernel
        k2 = deform_k * deform_k
        self.offset_conv = nn.Sequential(
            nn.Conv2d(c_unified, c_unified, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(c_unified, 2 * k2, kernel_size=1, bias=True)
        # Modulation mask for deformable sampling
        self.modulation_head = nn.Conv2d(c_unified, k2, kernel_size=1, bias=True)

        # Deformable sampling projection (1x1 conv after sampling)
        # NOTE: torchvision.ops.deform_conv2d is preferred but may be unavailable.
        # This fallback uses F.grid_sample with learned offsets.
        self.deform_proj = nn.Sequential(
            nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

        # Combine paths: add + project to dual-stream
        self.combine_proj = nn.Sequential(
            nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

        # Dual-stream split projections
        self.rgb_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)
        self.depth_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)

    def _deformable_sample(
        self, feat: torch.Tensor, offset: torch.Tensor, mod: torch.Tensor
    ) -> torch.Tensor:
        """Deformable sampling via F.grid_sample (fallback implementation).

        NOTE: torchvision.ops.deform_conv2d is preferred but may be unavailable
        in some build configurations. This fallback uses F.grid_sample with
        learned offsets, which is functionally similar but does not use the
        optimized CUDA kernel.

        Args:
            feat: Input feature [B, C, H, W] (N3 aligned).
            offset: Predicted offsets [B, 2*k*k, H, W].
            mod: Modulation mask [B, k*k, H, W].

        Returns:
            Deformably sampled feature [B, C, H, W].
        """
        B, C, H, W = feat.shape
        k = self.deform_k
        k2 = k * k

        # Reshape offset: [B, 2*k*k, H, W] → [B, H, W, k*k, 2]
        offset = offset.permute(0, 2, 3, 1).reshape(B, H, W, k2, 2)
        # Reshape mod: [B, k*k, H, W] → [B, H, W, k*k]
        mod = mod.permute(0, 2, 3, 1).reshape(B, H, W, k2)
        mod = torch.sigmoid(mod)  # modulation in [0, 1]

        # Build base grid for kxk neighborhood
        # Center at (0, 0), offsets in [-1, 1] normalized coordinates
        half = (k - 1) / 2.0
        # dy, dx: [k, k]
        dy = torch.arange(k, device=feat.device, dtype=feat.dtype) - half
        dx = torch.arange(k, device=feat.device, dtype=feat.dtype) - half
        grid_dy, grid_dx = torch.meshgrid(dy, dx, indexing="ij")
        # base_offset: [k*k, 2] (dy, dx)
        base_offset = torch.stack([grid_dx.reshape(-1), grid_dy.reshape(-1)], dim=-1)
        # Normalize to [-1, 1] range: divide by half spatial dim
        base_offset[..., 0] = base_offset[..., 0] / (W / 2.0)
        base_offset[..., 1] = base_offset[..., 1] / (H / 2.0)
        # [1, 1, 1, k*k, 2]
        base_offset = base_offset.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Total offset = base + learned
        total_offset = base_offset + offset  # [B, H, W, k*k, 2]

        # Build sampling grid: [B, H, W, 2] base positions in [-1, 1]
        y_grid = torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype)
        x_grid = torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing="ij")
        # [1, H, W, 1, 2]
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(3)

        # Sample grid: [B, H, W, k*k, 2]
        sample_grid = base_grid + total_offset
        # Clamp to valid range
        sample_grid = torch.clamp(sample_grid, -1.0, 1.0)

        # Apply deformable conv weight via grid_sample
        # For each kernel position, sample feat and apply corresponding weight slice
        # Reshape for grid_sample: [B*H*W, C, 1, 1] <- not efficient
        # Instead, use grouped approach: sample all k*k positions at once
        # Reshape sample_grid: [B, H*W*k*k, 1, 2] for grid_sample
        sample_grid_flat = sample_grid.reshape(B, H * W * k2, 1, 2)
        # grid_sample expects [B, C, H_in, W_in] and grid [B, H_out, W_out, 2]
        # We sample at H*W*k*k points
        sampled = F.grid_sample(
            feat, sample_grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # sampled: [B, C, H*W*k*k, 1]
        sampled = sampled.reshape(B, C, H, W, k2)
        # Apply modulation: [B, C, H, W, k*k] * [1, 1, H, W, k*k]
        mod_expanded = mod.unsqueeze(1)  # [B, 1, H, W, k*k]
        sampled = sampled * mod_expanded
        # Sum over kernel positions: [B, C, H, W]
        result = sampled.sum(dim=-1)

        # Project sampled features
        result = self.deform_proj(result)  # [B, C_u, H, W]

        return result

    def forward(
        self, p3_aligned: torch.Tensor, n3_aligned: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p3_aligned: RGB P3 after ScaleAlignBlock [B, C_u, 80, 80]
            n3_aligned: Depth N3 after ScaleAlignBlock [B, C_u, 80, 80]

        Returns:
            X3_RGB:   [B, C_u, 80, 80]
            X3_Depth: [B, C_u, 80, 80]
        """
        # p3_aligned: [B, C_u, 80, 80]
        # n3_aligned: [B, C_u, 80, 80]

        # Path A — Spatial modulation: N3 generates mask, multiplies onto P3
        spatial_mask = self.spatial_mask_head(n3_aligned)  # [B, C_u, 80, 80]
        path_a = p3_aligned * spatial_mask  # [B, C_u, 80, 80]

        # Path B — Deformable local alignment: P3 predicts offsets, sample N3
        offset_feat = self.offset_conv(p3_aligned)  # [B, C_u, 80, 80]
        offset = self.offset_head(offset_feat)  # [B, 2*k*k, 80, 80]
        mod = self.modulation_head(offset_feat)  # [B, k*k, 80, 80]
        path_b = self._deformable_sample(n3_aligned, offset, mod)  # [B, C_u, 80, 80]

        # Combine: add both paths
        combined = path_a + path_b  # [B, C_u, 80, 80]
        combined = self.combine_proj(combined)  # [B, C_u, 80, 80]

        # Dual-stream split
        x3_rgb = self.rgb_proj(combined)  # [B, C_u, 80, 80]
        x3_depth = self.depth_proj(combined)  # [B, C_u, 80, 80]

        return x3_rgb, x3_depth


# ============================================================================
# Part 3 — P4 Fusion (mid-scale 40x40, bidirectional, soft channel+spatial gating)
# ============================================================================
# Motivation: Mid-scale features benefit from bidirectional cross-modal channel
# selection. ECA (Efficient Channel Attention) is chosen over SE because:
#   - ECA avoids FC dimensionality reduction, has fewer parameters
#   - For C_unified=128, SE with reduction=4 would use 128*32+32*128=8192 params
#   - ECA uses a single 1D conv with adaptive kernel, ~128*k params (k~5)
#   - ECA preserves cross-channel interaction without information bottleneck
#
# Channel attention is SOFT (sigmoid-weighted), NOT hard binary removal.
# After channel attention, spatial attention via 7x7 depthwise conv on
# pooled maps provides spatial selectivity.


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) block.

    Uses 1D adaptive-kernel conv on global average-pooled features.
    SOFT weighting — NOT hard channel removal.

    Args:
        channels (int): Number of input channels.
        gamma (int): Kernel size calculation parameter. Default 2.
        b (int): Kernel size calculation parameter. Default 1.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = |log2(C)/gamma + b/gamma|_odd
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1  # ensure odd kernel size
        k = max(k, 3)  # minimum kernel size of 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k, padding=k // 2, bias=False
        )
        # SOFT weighting — NOT hard channel removal
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        # SOFT weighting — NOT hard channel removal
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y  # [B, C, H, W]


class SpatialAttentionBlock(nn.Module):
    """Spatial attention via 7x7 depthwise conv on pooled maps.

    Concatenates AvgPool and MaxPool along channel dim, applies 7x7 depthwise
    conv, sigmoid to produce spatial mask.

    Args:
        kernel_size (int): Conv kernel size. Default 7.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        spatial_mask = self.sigmoid(self.conv(pooled))  # [B, 1, H, W]
        return x * spatial_mask  # [B, C, H, W]


class P4FusionModule(nn.Module):
    """P4 fusion: bidirectional soft channel (ECA) + spatial attention gating.

    Concat [P4, N4] → ECA channel attention → spatial attention → split dual-stream.
    All channel weighting is SOFT (sigmoid), NOT hard binary removal.

    Args:
        c_unified (int): Unified channel count. Default 128.
    """

    def __init__(self, c_unified: int = 128):
        super().__init__()
        self.c_unified = c_unified

        # Channel attention on concatenated features
        self.channel_attn = ECABlock(channels=c_unified * 2)

        # Spatial attention on channel-attended features
        self.spatial_attn = SpatialAttentionBlock(kernel_size=7)

        # Project concatenated features back to c_unified for each stream
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(c_unified * 2, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv2d(c_unified * 2, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, p4_aligned: torch.Tensor, n4_aligned: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p4_aligned: RGB P4 after ScaleAlignBlock [B, C_u, 40, 40]
            n4_aligned: Depth N4 after ScaleAlignBlock [B, C_u, 40, 40]

        Returns:
            X4_RGB:   [B, C_u, 40, 40]
            X4_Depth: [B, C_u, 40, 40]
        """
        # p4_aligned: [B, C_u, 40, 40]
        # n4_aligned: [B, C_u, 40, 40]

        # Concatenate along channel dim
        fused = torch.cat([p4_aligned, n4_aligned], dim=1)  # [B, 2*C_u, 40, 40]

        # ECA channel attention (SOFT weighting — NOT hard channel removal)
        fused = self.channel_attn(fused)  # [B, 2*C_u, 40, 40]

        # Spatial attention
        fused = self.spatial_attn(fused)  # [B, 2*C_u, 40, 40]

        # Split into dual-stream
        x4_rgb = self.rgb_proj(fused)  # [B, C_u, 40, 40]
        x4_depth = self.depth_proj(fused)  # [B, C_u, 40, 40]

        return x4_rgb, x4_depth


# ============================================================================
# Part 4 — P5 Fusion (coarse-scale 20x20, RGB→Depth, CROSS-ATTENTION)
# ============================================================================
# CORRECT placement for cross-attention. Justification:
#   At 20x20, sequence length N=400, O(N^2)=160,000 — tractable.
#   Compare to P3: N=6400, O(N^2)=40,960,000 — 256x more expensive.
#   P5 features are semantically richest, making global relationship modeling
#   most meaningful here.
#
# Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
#   Q from P5 (RGB), K and V from N5 (Depth)
#   RGB semantic context attends over Depth positions, selecting which
#   Depth regions are semantically relevant.


class P5FusionModule(nn.Module):
    """P5 fusion: RGB guides Depth via cross-attention.

    Cross-attention is correctly placed at P5 (20x20, N=400, O(N^2)=160,000).
    At P3 (80x80, N=6400), O(N^2)=40,960,000 would be 256x more expensive
    and semantically wasteful on fine-scale, low-semantic features.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    Q from P5 (RGB), K and V from N5 (Depth).

    Args:
        c_unified (int): Unified channel count. Default 128.
        num_heads (int): Number of attention heads. Default 8 (C_u=128 / 8 = 16 per head).
    """

    def __init__(self, c_unified: int = 128, num_heads: int = 8):
        super().__init__()
        self.c_unified = c_unified
        self.num_heads = num_heads

        # Q projection from P5 (RGB)
        self.q_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)
        # K projection from N5 (Depth)
        self.k_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)
        # V projection from N5 (Depth)
        self.v_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)

        # Standard multi-head attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        self.attn = nn.MultiheadAttention(
            embed_dim=c_unified,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection after attention
        self.out_proj = nn.Sequential(
            nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

        # Dual-stream projections
        self.rgb_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)
        self.depth_proj = nn.Conv2d(c_unified, c_unified, kernel_size=1, bias=False)

    def forward(
        self, p5_aligned: torch.Tensor, n5_aligned: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p5_aligned: RGB P5 after ScaleAlignBlock [B, C_u, 20, 20]
            n5_aligned: Depth N5 after ScaleAlignBlock [B, C_u, 20, 20]

        Returns:
            X5_RGB:   [B, C_u, 20, 20]  (P5 projected, residual)
            X5_Depth: [B, C_u, 20, 20]  (attention output)
        """
        B, C_u, H, W = p5_aligned.shape  # H=20, W=20
        N = H * W  # N=400

        # Project Q, K, V
        Q = self.q_proj(p5_aligned)  # [B, C_u, 20, 20]
        K = self.k_proj(n5_aligned)  # [B, C_u, 20, 20]
        V = self.v_proj(n5_aligned)  # [B, C_u, 20, 20]

        # Reshape to sequence: [B, C_u, H, W] → [B, N, C_u]
        Q = Q.flatten(2).transpose(1, 2)  # [B, 400, C_u]
        K = K.flatten(2).transpose(1, 2)  # [B, 400, C_u]
        V = V.flatten(2).transpose(1, 2)  # [B, 400, C_u]

        # Standard Scaled Dot-Product Attention:
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        # QK^T: [B, N, N] = [B, 400, 400], O(N^2)=160,000 — tractable
        attn_out, _ = self.attn(Q, K, V)  # [B, 400, C_u]

        # Reshape back to spatial: [B, N, C_u] → [B, C_u, H, W]
        attn_out = attn_out.transpose(1, 2).reshape(B, C_u, H, W)  # [B, C_u, 20, 20]

        # Add residual connection: attention_out + N5_aligned
        attn_out = attn_out + n5_aligned  # [B, C_u, 20, 20]
        attn_out = self.out_proj(attn_out)  # [B, C_u, 20, 20]

        # Dual-stream output
        x5_rgb = self.rgb_proj(p5_aligned)  # [B, C_u, 20, 20] — P5 projected (residual)
        x5_depth = self.depth_proj(attn_out)  # [B, C_u, 20, 20] — attention output

        return x5_rgb, x5_depth


# ============================================================================
# Part 5 — FPN Top-Down Decoder
# ============================================================================
# Motivation: Merge all six feature maps (X3/X4/X5, RGB and Depth streams)
# into F3 at 80x80. Top-down merging propagates global context from P5
# cross-attention downward, compensating for the absence of global modeling
# at P3.
#
# Step 1: Concat(X5_RGB, X5_Depth) → Conv1x1 → F5 → upsample → F5_up
# Step 2: Concat(F5_up, X4_RGB, X4_Depth) → Conv1x1 → F4 → upsample → F4_up
# Step 3: Concat(F4_up, X3_RGB, X3_Depth) → Conv1x1 → F3


class FusionDecoder(nn.Module):
    """FPN top-down decoder merging all six feature maps into F3.

    Global context from P5 cross-attention propagates downward through
    this decoder, providing long-range semantic context to fine-scale
    features at P3.

    Args:
        c_unified (int): Unified channel count. Default 128.
    """

    def __init__(self, c_unified: int = 128):
        super().__init__()
        self.c_unified = c_unified

        # Step 1: Merge X5 → F5
        self.f5_conv = nn.Sequential(
            nn.Conv2d(c_unified * 2, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

        # Step 2: Merge F5_up + X4 → F4
        self.f4_conv = nn.Sequential(
            nn.Conv2d(c_unified * 3, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

        # Step 3: Merge F4_up + X3 → F3
        self.f3_conv = nn.Sequential(
            nn.Conv2d(c_unified * 3, c_unified, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_unified),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x3_rgb: torch.Tensor,
        x3_depth: torch.Tensor,
        x4_rgb: torch.Tensor,
        x4_depth: torch.Tensor,
        x5_rgb: torch.Tensor,
        x5_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x3_rgb:   [B, C_u, 80, 80]
            x3_depth: [B, C_u, 80, 80]
            x4_rgb:   [B, C_u, 40, 40]
            x4_depth: [B, C_u, 40, 40]
            x5_rgb:   [B, C_u, 20, 20]
            x5_depth: [B, C_u, 20, 20]

        Returns:
            F3: [B, C_u, 80, 80]
        """
        # Step 1: Merge X5
        f5 = self.f5_conv(torch.cat([x5_rgb, x5_depth], dim=1))  # [B, C_u, 20, 20]
        f5_up = F.interpolate(f5, scale_factor=2, mode="bilinear", align_corners=False)  # [B, C_u, 40, 40]

        # Step 2: Merge X4
        f4 = self.f4_conv(torch.cat([f5_up, x4_rgb, x4_depth], dim=1))  # [B, C_u, 40, 40]
        f4_up = F.interpolate(f4, scale_factor=2, mode="bilinear", align_corners=False)  # [B, C_u, 80, 80]

        # Step 3: Merge X3
        f3 = self.f3_conv(torch.cat([f4_up, x3_rgb, x3_depth], dim=1))  # [B, C_u, 80, 80]

        return f3


# ============================================================================
# Part 6 — Full Pipeline Assembly
# ============================================================================
# Assemble all modules into a single RGBDFusionNeck:
#   1. Align all six inputs via ScaleAlignBlock
#   2. P3 fusion → X3_RGB, X3_Depth
#   3. P4 fusion → X4_RGB, X4_Depth
#   4. P5 fusion → X5_RGB, X5_Depth
#   5. Decode → F3


class RGBDFusionNeck(nn.Module):
    """RGBD dual-branch fusion neck for amodal segmentation.

    Assembles ScaleAlignBlock (x6), P3FusionModule, P4FusionModule,
    P5FusionModule, and FusionDecoder into a complete pipeline.

    Input feature maps:
        P3: [B, 24,  80, 80]  — MobileNetV2 stride-8
        P4: [B, 32,  40, 40]  — MobileNetV2 stride-16
        P5: [B, 96,  20, 20]  — MobileNetV2 stride-32
        N3: [B, 64,  80, 80]  — StarNet-S4 stride-8
        N4: [B, 128, 40, 40]  — StarNet-S4 stride-16
        N5: [B, 256, 20, 20]  — StarNet-S4 stride-32

    Output:
        F3: [B, C_unified, 80, 80]

    Args:
        C_unified (int): Unified channel count. Default 128.
        rgb_channels (tuple): RGB branch channel counts (P3, P4, P5).
        depth_channels (tuple): Depth branch channel counts (N3, N4, N5).
    """

    def __init__(
        self,
        C_unified: int = 128,
        rgb_channels: tuple = (24, 32, 96),
        depth_channels: tuple = (64, 128, 256),
    ):
        super().__init__()
        self.C_unified = C_unified

        # Scale alignment blocks (one per input feature map)
        # Channel alignment first, then spatial alignment if needed
        # ASSUMPTION: P3/N3, P4/N4, P5/N5 already share spatial dimensions
        self.align_p3 = ScaleAlignBlock(rgb_channels[0], C_unified)
        self.align_p4 = ScaleAlignBlock(rgb_channels[1], C_unified)
        self.align_p5 = ScaleAlignBlock(rgb_channels[2], C_unified)
        self.align_n3 = ScaleAlignBlock(depth_channels[0], C_unified)
        self.align_n4 = ScaleAlignBlock(depth_channels[1], C_unified)
        self.align_n5 = ScaleAlignBlock(depth_channels[2], C_unified)

        # Three-scale fusion modules
        self.p3_fusion = P3FusionModule(c_unified=C_unified)
        self.p4_fusion = P4FusionModule(c_unified=C_unified)
        self.p5_fusion = P5FusionModule(c_unified=C_unified)

        # FPN top-down decoder
        self.decoder = FusionDecoder(c_unified=C_unified)

    def forward(
        self,
        P3: torch.Tensor,
        P4: torch.Tensor,
        P5: torch.Tensor,
        N3: torch.Tensor,
        N4: torch.Tensor,
        N5: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            P3: [B, 24,  80, 80]   — MobileNetV2 stride-8
            P4: [B, 32,  40, 40]   — MobileNetV2 stride-16
            P5: [B, 96,  20, 20]   — MobileNetV2 stride-32
            N3: [B, 64,  80, 80]   — StarNet-S4 stride-8
            N4: [B, 128, 40, 40]   — StarNet-S4 stride-16
            N5: [B, 256, 20, 20]   — StarNet-S4 stride-32

        Returns:
            F3: [B, C_unified, 80, 80]
        """
        # 1. Align all six inputs
        p3_a = self.align_p3(P3)  # [B, C_u, 80, 80]
        p4_a = self.align_p4(P4)  # [B, C_u, 40, 40]
        p5_a = self.align_p5(P5)  # [B, C_u, 20, 20]
        n3_a = self.align_n3(N3)  # [B, C_u, 80, 80]
        n4_a = self.align_n4(N4)  # [B, C_u, 40, 40]
        n5_a = self.align_n5(N5)  # [B, C_u, 20, 20]

        # 2. P3 fusion: Depth guides RGB, LOCAL mechanism
        x3_rgb, x3_depth = self.p3_fusion(p3_a, n3_a)  # [B, C_u, 80, 80] each

        # 3. P4 fusion: Bidirectional, soft channel+spatial gating
        x4_rgb, x4_depth = self.p4_fusion(p4_a, n4_a)  # [B, C_u, 40, 40] each

        # 4. P5 fusion: RGB guides Depth, CROSS-ATTENTION
        x5_rgb, x5_depth = self.p5_fusion(p5_a, n5_a)  # [B, C_u, 20, 20] each

        # 5. Decode: FPN top-down → F3
        f3 = self.decoder(x3_rgb, x3_depth, x4_rgb, x4_depth, x5_rgb, x5_depth)  # [B, C_u, 80, 80]

        return f3


# ============================================================================
# Shape Verification & Parameter Count
# ============================================================================
if __name__ == "__main__":
    import torch

    B, C_u = 2, 128

    P3 = torch.randn(B, 24, 80, 80)   # MobileNetV2 stride-8
    P4 = torch.randn(B, 32, 40, 40)   # MobileNetV2 stride-16
    P5 = torch.randn(B, 96, 20, 20)   # MobileNetV2 stride-32
    N3 = torch.randn(B, 64, 80, 80)   # StarNet-S4 stride-8
    N4 = torch.randn(B, 128, 40, 40)  # StarNet-S4 stride-16
    N5 = torch.randn(B, 256, 20, 20)  # StarNet-S4 stride-32

    neck = RGBDFusionNeck(C_unified=C_u)
    F3 = neck(P3, P4, P5, N3, N4, N5)

    print(f"Output shape : {F3.shape}")
    assert F3.shape == (B, C_u, 80, 80), f"Shape mismatch: {F3.shape}"
    print("All shape checks passed.")

    total = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    print(f"Neck parameters: {total:,}")

    # Per-module parameter breakdown
    print("\n--- Parameter Breakdown ---")
    for name, module in neck.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {params:,}")

    # Individual module tests
    print("\n--- Individual Module Tests ---")

    # ScaleAlignBlock
    sab = ScaleAlignBlock(c_in=24, c_unified=128)
    out = sab(P3)
    print(f"ScaleAlignBlock(24→128): {P3.shape} → {out.shape}")
    assert out.shape == (B, 128, 80, 80)

    # P3FusionModule
    p3_mod = P3FusionModule(c_unified=128)
    p3_a = torch.randn(B, 128, 80, 80)
    n3_a = torch.randn(B, 128, 80, 80)
    x3r, x3d = p3_mod(p3_a, n3_a)
    print(f"P3FusionModule: P3[{p3_a.shape}] + N3[{n3_a.shape}] → X3_RGB[{x3r.shape}], X3_Depth[{x3d.shape}]")
    assert x3r.shape == (B, 128, 80, 80)
    assert x3d.shape == (B, 128, 80, 80)

    # P4FusionModule
    p4_mod = P4FusionModule(c_unified=128)
    p4_a = torch.randn(B, 128, 40, 40)
    n4_a = torch.randn(B, 128, 40, 40)
    x4r, x4d = p4_mod(p4_a, n4_a)
    print(f"P4FusionModule: P4[{p4_a.shape}] + N4[{n4_a.shape}] → X4_RGB[{x4r.shape}], X4_Depth[{x4d.shape}]")
    assert x4r.shape == (B, 128, 40, 40)
    assert x4d.shape == (B, 128, 40, 40)

    # P5FusionModule
    p5_mod = P5FusionModule(c_unified=128)
    p5_a = torch.randn(B, 128, 20, 20)
    n5_a = torch.randn(B, 128, 20, 20)
    x5r, x5d = p5_mod(p5_a, n5_a)
    print(f"P5FusionModule: P5[{p5_a.shape}] + N5[{n5_a.shape}] → X5_RGB[{x5r.shape}], X5_Depth[{x5d.shape}]")
    assert x5r.shape == (B, 128, 20, 20)
    assert x5d.shape == (B, 128, 20, 20)

    # FusionDecoder
    decoder = FusionDecoder(c_unified=128)
    f3 = decoder(x3r, x3d, x4r, x4d, x5r, x5d)
    print(f"FusionDecoder: → F3[{f3.shape}]")
    assert f3.shape == (B, 128, 80, 80)

    print("\nAll individual module tests passed!")

    # ====================================================================
    # Part 7 — Ablation Study Plan
    # ====================================================================
    print("\n" + "=" * 70)
    print("Part 7 — Ablation Study Plan")
    print("=" * 70)
    print("""
| Priority | Component                | What to ablate                                      | Metric               | Expected result if useful                              |
|----------|--------------------------|-----------------------------------------------------|----------------------|--------------------------------------------------------|
| 1        | P5 Cross-Attention       | Replace with simple concat+conv (no attention)      | Mask AP / Amodal AP  | AP drops significantly — global semantic guidance lost  |
| 2        | P3 Deformable Alignment  | Remove Path B (deformable), keep only spatial mask  | Boundary IoU / AP_S  | Small-object AP drops — sub-pixel alignment matters     |
| 3        | P4 ECA Channel Attention | Replace ECA with identity (no channel gating)       | Mask AP              | Moderate AP drop — channel selection is beneficial      |
| 4        | FPN Decoder              | Replace top-down decoder with single-scale F3=concat | Mask AP / AP_M / AP_L| Multi-scale drop, especially AP_L — context propagation |
| 5        | Dual-Stream Output       | Merge RGB/Depth into single stream per scale        | Amodal AP            | Slight drop — dual-stream preserves modality info       |

Honest Limitations:

1. Depth sensor bias: The deformable alignment at P3 assumes small, learnable
   misalignments between RGB and Depth. Large systematic offsets (e.g., from
   structured-light sensors with baseline parallax) may exceed the offset field's
   capacity, requiring explicit geometric calibration as a preprocessing step.

2. Cross-attention at P5 is still O(N^2): While N=400 is tractable, this
   architecture does not scale to higher-resolution P5 maps. If the backbone
   changes to produce larger P5 feature maps (e.g., 40x40), the attention cost
   increases 16x, potentially requiring a switch to linear attention or
   deformable attention.

3. FPN decoder uses only 1x1 convs: The decoder's top-down path relies on
   bilinear upsampling and 1x1 convolutions, which may not adequately resolve
   aliasing artifacts. Adding 3x3 convolutions or lateral connections from
   aligned backbone features would improve this at the cost of parameters.

4. No temporal consistency: This is a single-frame architecture. Video
   sequences with temporal depth noise will produce flickering predictions.
   Temporal smoothing or recurrent fusion is needed for video applications.

5. Dataset scalability: The dual-branch design requires paired RGBD training
   data. Datasets without depth annotations cannot leverage the Depth branch,
   and the Depth branch's features will be randomly initialized and untrained,
   potentially hurting performance vs. an RGB-only model.

6. P3 deformable fallback uses F.grid_sample, not torchvision.ops.deform_conv2d:
   The grid_sample fallback is less memory-efficient and may produce subtly
   different gradients compared to the optimized CUDA deformable convolution.
   This should be swapped when torchvision is available.
""")
