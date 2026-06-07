# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Custom modules for RGB-D Apple Amodal Detection: SFM, WCAF, DGFFN."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("SFM", "WCAF", "DGFFN", "HaarDWT", "HaarIDWT")


# ---------------------------------------------------------------------------
# Haar DWT / IDWT (pure PyTorch, no external deps, ONNX-friendly)
# ---------------------------------------------------------------------------
class HaarDWT(nn.Module):
    """2D Haar Discrete Wavelet Transform using tensor slicing.

    Decomposes input into 4 subbands: LL, LH, HL, HH each at half spatial resolution.
    """

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): LL, LH, HL, HH subbands.
        """
        # Average pairs along H and W (Haar low-pass / high-pass)
        x_ll = (x[..., 0::2, 0::2] + x[..., 1::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 1::2]) * 0.5
        x_lh = (x[..., 0::2, 0::2] - x[..., 1::2, 0::2] + x[..., 0::2, 1::2] - x[..., 1::2, 1::2]) * 0.5
        x_hl = (x[..., 0::2, 0::2] + x[..., 1::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 1::2]) * 0.5
        x_hh = (x[..., 0::2, 0::2] - x[..., 1::2, 0::2] - x[..., 0::2, 1::2] + x[..., 1::2, 1::2]) * 0.5
        return x_ll, x_lh, x_hl, x_hh


class HaarIDWT(nn.Module):
    """2D Inverse Haar Discrete Wavelet Transform using tensor slicing.

    Reconstructs full-resolution tensor from LL, LH, HL, HH subbands.
    """

    def forward(self, ll, lh, hl, hh):
        """Forward pass.

        Args:
            ll (torch.Tensor): LL subband.
            lh (torch.Tensor): LH subband.
            hl (torch.Tensor): HL subband.
            hh (torch.Tensor): HH subband.

        Returns:
            (torch.Tensor): Reconstructed tensor of shape (B, C, 2*H, 2*W).
        """
        B, C, H, W = ll.shape
        out = ll.new_zeros(B, C, H * 2, W * 2)
        out[..., 0::2, 0::2] = ll + lh + hl + hh
        out[..., 1::2, 0::2] = ll - lh + hl - hh
        out[..., 0::2, 1::2] = ll + lh - hl - hh
        out[..., 1::2, 1::2] = ll - lh - hl + hh
        return out


# ---------------------------------------------------------------------------
# SFM - Strip-Freq Mixer
# ---------------------------------------------------------------------------
class SFM(nn.Module):
    """Strip-Freq Mixer: parallel strip perception + global frequency modeling.

    Replaces C3k2/C2f in the backbone.  Output channels == input channels.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (should equal c1 for residual connection).
        k (int): Strip convolution kernel size. Default 7.
    """

    def __init__(self, c1, c2, k=7):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # ---- Branch A: Strip Perception (orthogonal 1xK / Kx1 DW-Convs) ----
        self.strip_h = nn.Conv2d(c1, c1, kernel_size=(1, k), padding=(0, k // 2), groups=c1, bias=False)
        self.strip_v = nn.Conv2d(c1, c1, kernel_size=(k, 1), padding=(k // 2, 0), groups=c1, bias=False)
        self.strip_bn_h = nn.BatchNorm2d(c1)
        self.strip_bn_v = nn.BatchNorm2d(c1)
        self.strip_act = nn.SiLU()

        # ---- Branch B: Global Frequency (2D FFT) ----
        # Process real and imag parts with shared Conv2d
        self.freq_conv = nn.Conv2d(c1 * 2, c1 * 2, kernel_size=1, bias=False)
        self.freq_bn = nn.BatchNorm2d(c1 * 2)

        # ---- Fusion: concat A+B -> 1x1 Conv ----
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 * 2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # ---- Residual projection (when c1 != c2) ----
        self.shortcut = (
            nn.Sequential(nn.Conv2d(c1, c2, kernel_size=1, bias=False), nn.BatchNorm2d(c2))
            if c1 != c2
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)

        # Branch A: Strip Perception
        a = self.strip_act(self.strip_bn_h(self.strip_h(x)) + self.strip_bn_v(self.strip_v(x)))

        # Branch B: Global Frequency
        x_freq = torch.fft.rfft2(x, norm="ortho")
        x_real = x_freq.real
        x_imag = x_freq.imag
        # Concat real & imag along channel dim, process, then split back
        x_cat = torch.cat([x_real, x_imag], dim=1)
        x_cat = self.freq_bn(self.freq_conv(x_cat))
        x_real_p, x_imag_p = x_cat.chunk(2, dim=1)
        x_freq_new = torch.complex(x_real_p, x_imag_p)
        b = torch.fft.irfft2(x_freq_new, s=x.shape[-2:], norm="ortho")

        # Fusion
        out = self.fusion(torch.cat([a, b], dim=1))
        return out + identity


# ---------------------------------------------------------------------------
# WCAF - Wavelet-Cross-Attention Fusion
# ---------------------------------------------------------------------------
class WCAF(nn.Module):
    """Wavelet-Cross-Attention Fusion for RGB and Depth feature streams.

    Replaces Concat in the Neck.  Uses Depth LL subband to gate RGB HF subbands,
    suppressing lighting/shadow noise with geometric priors.

    Args:
        c_rgb (int): RGB feature channels.
        c_dep (int): Depth feature channels.
    """

    def __init__(self, c_rgb, c_dep):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # Project depth channels to match RGB channels for gating
        self.dep_proj = nn.Conv2d(c_dep, c_rgb, kernel_size=1, bias=False)

        # Spatial attention from Depth LL: 1x1 Conv -> Sigmoid
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Output projection after IDWT reconstruction
        self.out_proj = nn.Sequential(
            nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_rgb),
            nn.SiLU(),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]): [rgb_feat, depth_feat].

        Returns:
            (torch.Tensor): Fused feature tensor.
        """
        if isinstance(x, (list, tuple)):
            f_rgb, f_dep = x[0], x[1]
        else:
            f_rgb = x
            f_dep = x

        # DWT decomposition
        rgb_ll, rgb_lh, rgb_hl, rgb_hh = self.dwt(f_rgb)
        dep_ll, dep_lh, dep_hl, dep_hh = self.dwt(f_dep)

        # Project depth LL to RGB channel dim
        dep_ll_proj = self.dep_proj(dep_ll)

        # Cross-modal gating: Depth LL -> Spatial Attention Map
        attn = self.spatial_attn(dep_ll_proj)

        # Gate RGB high-frequency subbands with depth-derived attention
        rgb_lh_gated = rgb_lh * attn
        rgb_hl_gated = rgb_hl * attn
        rgb_hh_gated = rgb_hh * attn

        # IDWT reconstruction
        fused = self.idwt(rgb_ll, rgb_lh_gated, rgb_hl_gated, rgb_hh_gated)

        return self.out_proj(fused)


# ---------------------------------------------------------------------------
# DGFFN - Dilated-Gated Feed-Forward Network
# ---------------------------------------------------------------------------
class DGFFN(nn.Module):
    """Dilated-Gated FFN: multi-scale dilated DWConv + Channel Attention + GLU.

    Replaces the standard YOLO FFN (two 1x1 Convs).  Output channels == input channels.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (should equal c1 for residual connection).
        e (float): Expansion ratio for hidden channels. Default 2.0.
    """

    def __init__(self, c1, c2, e=2.0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        c_hidden = int(c1 * e)

        # 1. Channel expansion (1x1 Conv)
        self.expand = nn.Sequential(
            nn.Conv2d(c1, c_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU(),
        )

        # 2. Multi-scale Dilated DWConv: split c_hidden into two halves
        c_half = c_hidden // 2
        self.dw_d1 = nn.Conv2d(c_half, c_half, kernel_size=3, padding=1, groups=c_half, bias=False)
        self.dw_d2 = nn.Conv2d(c_half, c_half, kernel_size=5, padding=4, dilation=2, groups=c_half, bias=False)
        self.dw_bn = nn.BatchNorm2d(c_hidden)

        # 3. Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_hidden, c_hidden // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c_hidden // 4, c_hidden, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 4. GLU is applied inline in forward (no extra parameters needed)

        # 5. Channel projection (1x1 Conv)
        self.project = nn.Sequential(
            nn.Conv2d(c_hidden // 2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        )

        # 6. Residual projection (when c1 != c2)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(c1, c2, kernel_size=1, bias=False), nn.BatchNorm2d(c2))
            if c1 != c2
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)

        # 1. Expand
        h = self.expand(x)

        # 2. Multi-scale Dilated DWConv
        h1, h2 = h.chunk(2, dim=1)
        h = torch.cat([self.dw_d1(h1), self.dw_d2(h2)], dim=1)
        h = self.dw_bn(h)

        # 3. Channel Attention
        h = h * self.ca(h)

        # 4. GLU: split into value and gate, multiply
        v, g = h.chunk(2, dim=1)
        h = v * torch.sigmoid(g)

        # 5. Project + Residual
        return self.project(h) + identity
