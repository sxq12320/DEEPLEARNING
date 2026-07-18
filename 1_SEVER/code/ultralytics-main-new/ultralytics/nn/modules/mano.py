"""Multipole Attention (MANO) as a plug-and-play Ultralytics module.

Adapts the Multipole Attention Neural Operator (MANO, "Linear Attention with
Global Context: A Multipole Attention Mechanism for Vision and Physics",
ICCV-W 2025) into an NCHW, channel-preserving block that drops into YOLO exactly
like ``C2PSA``. Pure PyTorch (no einops), robust to arbitrary feature-map sizes.

Core idea (faithful to the paper): a SINGLE windowed self-attention is applied
across a hierarchy of down-sampled copies of the feature map, then aggregated
bottom-up with a V-cycle ``res = out_level + (1/(i+1)) * up(res)``. This yields a
global receptive field at near-linear cost, dominated by the finest scale.

Three integration fixes versus the reference implementation
(``MANO-main/models/MANO.py::Multipole_Attention2D``):
  1. Layout - reference is NHWC; we operate directly in NCHW.
  2. Dynamic depth - the number of levels is computed per-forward from the actual
     H, W instead of ``int(log(image_size, rate))`` baked in at ``__init__``.
  3. Non-power-of-two sizes - pad before down-sampling and interpolate-align after
     up-sampling, so 80/40/20-type YOLO maps never break (20 -> 10 -> 5 ...).

Exposed module: ``C2MANO(c1, c2, n, e)`` - same channel contract as ``C2PSA``,
so it needs no dedicated channel branch in ``parse_model`` (just membership in
``base_modules`` / ``repeat_modules``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("C2MANO", "MANOBlock", "MultipoleAttention")


def _pad_to_multiple(x: torch.Tensor, m: int) -> tuple[torch.Tensor, int, int]:
    """Zero-pad the bottom/right of an NCHW tensor so H, W become multiples of ``m``.

    Returns the padded tensor plus the ORIGINAL (h, w) so callers can crop back.
    """
    _, _, h, w = x.shape
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, h, w


class _WindowMHSA(nn.Module):
    """Plain multi-head self-attention over the tokens of a single window ([B_, N, C])."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend within each window. Input/-output shape: [B_, N, C]."""
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B_, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class MultipoleAttention(nn.Module):
    """Hierarchical windowed attention with a V-cycle. NCHW in/out, channels preserved.

    Args:
        dim (int): Input/output channels.
        window_size (int): Local attention window (non-overlapping). Default 4 so that
            small YOLO maps (P5=20) still reach ``max_levels`` (20 -> 10 -> 5).
        num_heads (int | None): Attention heads; defaults to ``max(dim // 32, 1)``.
        sampling_rate (int): Down-/up-sampling factor between levels.
        max_levels (int): Upper bound on hierarchy depth (actual depth is size-adaptive).
        downsample (str): ``"conv"`` (learned, per the paper) or ``"avg"`` (parameter-free).
    """

    def __init__(self, dim, window_size=4, num_heads=None, sampling_rate=2, max_levels=3, downsample="conv"):
        super().__init__()
        if num_heads is None:
            num_heads = max(dim // 32, 1)
        while dim % num_heads != 0 and num_heads > 1:  # keep dim divisible by heads
            num_heads -= 1
        self.window_size = int(window_size)
        self.sampling_rate = int(sampling_rate)
        self.max_levels = int(max_levels)

        self.norm = nn.LayerNorm(dim)
        self.attn = _WindowMHSA(dim, num_heads)  # SHARED across every level

        r = self.sampling_rate
        if downsample == "conv":  # shared learned sampling kernels
            self.down = nn.Conv2d(dim, dim, r, r, bias=False)
            self.up = nn.ConvTranspose2d(dim, dim, r, r, bias=False)
        else:  # parameter-free
            self.down = nn.AvgPool2d(r, r)
            self.up = nn.Upsample(scale_factor=r, mode="nearest")

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Non-overlapping windowed self-attention on an NCHW map (pads, then crops back)."""
        ws = self.window_size
        x, h0, w0 = _pad_to_multiple(x, ws)
        b, c, h, w = x.shape
        nh, nw = h // ws, w // ws
        # [B, C, H, W] -> [B*nH*nW, ws*ws, C]
        x = x.view(b, c, nh, ws, nw, ws).permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, c)
        x = self.attn(self.norm(x))
        # [B*nH*nW, ws*ws, C] -> [B, C, H, W]
        x = x.view(b, nh, nw, ws, ws, c).permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, h, w)
        return x[:, :, :h0, :w0]  # crop padding

    def _num_levels(self, h: int, w: int) -> int:
        """Size-adaptive hierarchy depth: keep down-sampling while >= one window."""
        r, ws, levels = self.sampling_rate, self.window_size, 1
        while levels < self.max_levels:
            h, w = math.ceil(h / r), math.ceil(w / r)
            if min(h, w) < ws:  # coarser than a single window -> stop
                break
            levels += 1
        return levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multipole attention. Input/-output shape: [B, C, H, W]."""
        _, _, h, w = x.shape
        levels = self._num_levels(h, w)

        # build the pyramid (pad before each conv down-sample so nothing is dropped)
        feats = [x]
        for _ in range(1, levels):
            padded, _, _ = _pad_to_multiple(feats[-1], self.sampling_rate)
            feats.append(self.down(padded))

        outs = [self._window_attention(f) for f in feats]  # one shared attention, every scale

        # V-cycle aggregation, coarse -> fine: res = out_l + (1/(i+1)) * up(res)
        res = outs[-1]
        for i, level in enumerate(range(len(outs) - 2, -1, -1)):
            up = self.up(res)
            tgt = outs[level]
            if up.shape[-2:] != tgt.shape[-2:]:  # align to exact finer-scale size
                up = F.interpolate(up, size=tgt.shape[-2:], mode="nearest")
            res = tgt + (1.0 / (i + 1)) * up
        return res


class MANOBlock(nn.Module):
    """Multipole attention + feed-forward with residual connections (mirrors ``PSABlock``)."""

    def __init__(self, c, window_size=4, num_heads=None, sampling_rate=2, max_levels=3, shortcut=True):
        super().__init__()
        self.attn = MultipoleAttention(c, window_size, num_heads, sampling_rate, max_levels)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual multipole-attention then residual FFN."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2MANO(nn.Module):
    """C2PSA-style wrapper that exposes Multipole Attention to YOLO YAMLs.

    Channel-preserving (``c1 == c2``); shares C2PSA's parse contract, so a YAML entry
    is simply ``[-1, n, C2MANO, [c2]]``.

    Examples:
        >>> m = C2MANO(c1=256, c2=256, n=1)
        >>> y = m(torch.randn(1, 256, 20, 20))
        >>> tuple(y.shape)
        (1, 256, 20, 20)
    """

    def __init__(self, c1, c2, n=1, e=0.5, window_size=4, num_heads=None, sampling_rate=2, max_levels=3):
        super().__init__()
        assert c1 == c2, "C2MANO preserves channels (c1 must equal c2)."
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(MANOBlock(self.c, window_size, num_heads, sampling_rate, max_levels) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split, run the multipole blocks on one half, concat, and project back."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
