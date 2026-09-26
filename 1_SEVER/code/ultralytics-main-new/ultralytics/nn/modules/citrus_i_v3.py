# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Narrow achromatic branch for controlled RGB/structure experiments.

Both branches receive the same augmented RGB tensor. Luminance, contrast and
gradient are deterministic views, not independent modalities or new evidence.
The RGB trunk is retained; only P2/P3 receive bounded, spatially gated updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("IV3InputTwin", "IV3AsymGrayFuse")


class IV3InputTwin(nn.Module):
    """Return RGB and one achromatic view without a disk preprocessing pipeline.

    Modes: 0 luminance, 1 luminance plus local contrast, 2 luminance plus
    gradient magnitude. Channel order is fixed and follows torchvision RGB.
    """

    def __init__(self, mode: int = 1):
        super().__init__()
        if mode not in (0, 1, 2):
            raise ValueError("IV3InputTwin mode must be 0, 1 or 2")
        self.mode = mode
        self.register_buffer("rgb_to_luma", torch.tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1))
        self.register_buffer("sobel_x", torch.tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 8)
        self.register_buffer("sobel_y", self.sobel_x.transpose(-1, -2).contiguous())

    def forward(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        if rgb.shape[1] != 3:
            raise ValueError("IV3InputTwin expects 3-channel RGB input")
        luma = (rgb * self.rgb_to_luma.to(dtype=rgb.dtype)).sum(1, keepdim=True)
        if self.mode == 0:
            return [rgb, luma]
        if self.mode == 1:
            local = F.avg_pool2d(luma, 5, stride=1, padding=2, count_include_pad=False)
            return [rgb, torch.cat((luma, luma - local), dim=1)]
        dx = F.conv2d(luma, self.sobel_x.to(dtype=luma.dtype), padding=1)
        dy = F.conv2d(luma, self.sobel_y.to(dtype=luma.dtype), padding=1)
        magnitude = torch.sqrt(dx.square() + dy.square() + 1e-8)
        return [rgb, torch.cat((luma, magnitude), dim=1)]


class IV3AsymGrayFuse(nn.Module):
    """Inject a narrow structural stream only where RGB/structure agree.

    `mode=0` is a direct projected residual control. `mode=1` uses a spatial
    agreement gate. Zero gain keeps the pretrained RGB function intact at the
    first step; the bounded gain limits the scale of later corrections.
    """

    def __init__(self, channels: list[int], mode: int = 1):
        super().__init__()
        if len(channels) != 2 or mode not in (0, 1):
            raise ValueError("IV3AsymGrayFuse expects [RGB, achromatic] and mode 0/1")
        rgb_channels, gray_channels = channels
        self.mode = mode
        self.rgb_key = Conv(rgb_channels, 8, 1)
        self.gray_key = Conv(gray_channels, 8, 1)
        self.gray_value = Conv(gray_channels, rgb_channels, 1, act=False)
        self.gate = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.38629436)  # sigmoid -> 0.2
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        rgb, gray = features
        if rgb.shape[-2:] != gray.shape[-2:]:
            raise ValueError("IV3 RGB and achromatic features must be spatially aligned")
        gray_value = self.gray_value(gray)
        if self.mode == 0:
            return rgb + self.gain.tanh() * gray_value
        rgb_key, gray_key = self.rgb_key(rgb), self.gray_key(gray)
        disagreement = (rgb_key - gray_key).abs()
        gate = self.gate(torch.cat((rgb_key, gray_key, disagreement), 1)).sigmoid()
        return rgb + self.gain.tanh() * gate * gray_value
