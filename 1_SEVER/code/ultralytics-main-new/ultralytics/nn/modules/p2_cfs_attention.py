"""Channel-frequency-spatial attention for the YOLO P2 feature map.

The block is designed for immature-citrus instance segmentation, where P2 must
retain small fruit, weak concave boundaries, and thin leaf/branch occluders. It
uses three channel partitions instead of running every operation over the full
feature tensor:

* channel: multi-scale local contrast plus fixed low-frequency DCT descriptors;
* frequency: phase-preserving band modulation and local soft orientation cues;
* spatial: Fourier-routed horizontal, vertical, and diagonal strip filtering.

The three partitions are mixed by a zero-initialized residual projection, so a
new block starts as an exact identity and does not disturb transferred weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("P2CFSAttention",)


def _dct_basis(size: int, frequencies: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Return orthonormal 2-D DCT-II basis maps with shape ``[K, size, size]``."""
    position = torch.arange(size, dtype=torch.float32) + 0.5
    maps = []
    for u, v in frequencies:
        if not (0 <= u < size and 0 <= v < size):
            raise ValueError(f"DCT frequency {(u, v)} is invalid for size={size}")
        norm_u = math.sqrt(1.0 / size) if u == 0 else math.sqrt(2.0 / size)
        norm_v = math.sqrt(1.0 / size) if v == 0 else math.sqrt(2.0 / size)
        basis_u = norm_u * torch.cos(math.pi * u * position / size)
        basis_v = norm_v * torch.cos(math.pi * v * position / size)
        maps.append(torch.outer(basis_u, basis_v))
    return torch.stack(maps)


class _ChannelGate(nn.Module):
    """Channel gate from local contrast statistics and compact DCT descriptors."""

    def __init__(self, pools: tuple[int, ...] = (3, 5, 7), dct_size: int = 8, dct_k: int = 4):
        super().__init__()
        frequency_order = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2))
        if not 1 <= dct_k <= len(frequency_order):
            raise ValueError(f"dct_k must be in [1, {len(frequency_order)}], got {dct_k}")
        if any(k < 3 or k % 2 == 0 for k in pools):
            raise ValueError(f"pool sizes must be odd integers >= 3, got {pools}")

        self.pools = pools
        self.dct_size = int(dct_size)
        self.register_buffer("basis", _dct_basis(self.dct_size, frequency_order[:dct_k]), persistent=False)

        descriptor_dim = 2 + 2 * len(pools) + dct_k
        hidden = max(descriptor_dim // 2, 4)
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.cross_channel = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-adaptive channel reweighting without changing shape."""
        x_float = x.float()
        mean = x_float.mean((-2, -1))
        std = (x_float - mean[..., None, None]).square().mean((-2, -1)).add(1e-6).sqrt()
        descriptors = [mean, std]
        for kernel in self.pools:
            smooth = F.avg_pool2d(x_float, kernel, stride=1, padding=kernel // 2)
            detail = x_float - smooth
            descriptors.extend((detail.abs().mean((-2, -1)), detail.square().mean((-2, -1)).add(1e-6).sqrt()))

        h, w = x.shape[-2:]
        pad_h = (self.dct_size - h % self.dct_size) % self.dct_size
        pad_w = (self.dct_size - w % self.dct_size) % self.dct_size
        dct_input = F.pad(x_float, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x_float
        kernel = (dct_input.shape[-2] // self.dct_size, dct_input.shape[-1] // self.dct_size)
        pooled = F.avg_pool2d(dct_input, kernel_size=kernel, stride=kernel).flatten(2)
        dct = pooled @ self.basis.flatten(1).transpose(0, 1)
        descriptors.extend(dct.unbind(-1))

        descriptor = torch.stack(descriptors, dim=-1)
        descriptor = F.layer_norm(descriptor, (descriptor.shape[-1],))
        score = self.descriptor_mlp(descriptor).squeeze(-1)
        gate = torch.sigmoid(self.cross_channel(score.unsqueeze(1)).squeeze(1)).to(x.dtype)
        scale = torch.tanh(self.gain).to(x.dtype)
        return x * (1.0 + scale * (2.0 * gate.unsqueeze(-1).unsqueeze(-1) - 1.0))


class _SpectralGate(nn.Module):
    """Phase-preserving band modulation plus soft local Fourier orientation."""

    def __init__(self, orientation_window: int = 8):
        super().__init__()
        if orientation_window < 4:
            raise ValueError(f"orientation_window must be >= 4, got {orientation_window}")
        self.orientation_window = int(orientation_window)
        self.band_mlp = nn.Sequential(
            nn.Linear(3, 6),
            nn.SiLU(inplace=True),
            nn.Linear(6, 3),
            nn.Tanh(),
        )
        self.gain = nn.Parameter(torch.tensor(0.1))

        # FAA estimates one hard dominant angle and rotates a feature. Here the
        # angles remain soft and local, because rotating P2 would move visible
        # citrus boundaries. Fourier-line angles are perpendicular to spatial
        # structures, hence the pi/2 shift below.
        size = self.orientation_window
        fy = torch.fft.fftfreq(size)
        fx = torch.fft.rfftfreq(size)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        spatial_angle = torch.remainder(torch.atan2(grid_y, grid_x) + math.pi / 2.0, math.pi)
        centers = torch.arange(4, dtype=torch.float32) * (math.pi / 4.0)
        angular_masks = torch.exp(4.0 * torch.cos(2.0 * (spatial_angle[None] - centers[:, None, None])))
        angular_masks = angular_masks * radius[None]
        angular_masks[:, 0, 0] = 0.0
        angular_masks = angular_masks / angular_masks.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        self.register_buffer("angular_masks", angular_masks, persistent=False)

    @staticmethod
    def _bands(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Build normalized radial low/mid/high masks for an RFFT spectrum."""
        fy = torch.fft.fftfreq(h, device=device).abs() / 0.5
        fx = torch.fft.rfftfreq(w, device=device).abs() / 0.5
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()).clamp(0.0, 1.0)
        low = torch.exp(-((radius / 0.32) ** 2))
        mid = torch.exp(-(((radius - 0.48) / 0.22) ** 2))
        high = 1.0 - torch.exp(-((radius / 0.70) ** 4))
        bands = torch.stack((low, mid, high))
        return bands / bands.sum(0, keepdim=True).clamp_min(1e-6)

    def _local_orientation(self, x: torch.Tensor) -> torch.Tensor:
        """Return differentiable horizontal/diagonal/vertical/anti-diagonal weights."""
        window = self.orientation_window
        b, _, h, w = x.shape
        signal = x.float().mean(1, keepdim=True)
        pad_h = (window - h % window) % window
        pad_w = (window - w % window) % window
        if pad_h or pad_w:
            signal = F.pad(signal, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = signal.shape[-2:]

        patches = F.unfold(signal, kernel_size=window, stride=window)
        patches = patches.transpose(1, 2).reshape(-1, window, window)
        magnitude = torch.fft.rfft2(patches, norm="ortho").abs().add(1e-6).log1p()
        energy = (magnitude[:, None] * self.angular_masks[None]).sum((-2, -1))
        weights = torch.softmax(F.layer_norm(energy, (4,)), dim=-1)
        weights = weights.view(b, hp // window, wp // window, 4).permute(0, 3, 1, 2)
        return F.interpolate(weights, size=(h, w), mode="nearest")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the filtered feature and its local orientation-routing map."""
        dtype = x.dtype
        x_float = x.float()  # CUDA FFT is unreliable for fp16 and non-power-of-two maps.
        h, w = x.shape[-2:]
        spectrum = torch.fft.rfft2(x_float, norm="ortho")
        bands = self._bands(h, w, x.device).to(x_float.dtype)

        magnitude = spectrum.abs().add(1e-6).log1p()
        numerator = (magnitude.unsqueeze(2) * bands[None, None]).sum((-2, -1))
        denominator = bands.sum((-2, -1)).clamp_min(1e-6)
        energy = F.layer_norm(numerator / denominator, (3,))
        weights = self.band_mlp(energy)

        modulation = (weights[..., None, None] * bands[None, None]).sum(2)
        spectral_gain = 1.0 + torch.tanh(self.gain) * modulation
        output = torch.fft.irfft2(spectrum * spectral_gain, s=(h, w), norm="ortho")
        return output.to(dtype), self._local_orientation(x).to(dtype)


class _MaskedDepthwiseConv(nn.Module):
    """Depthwise diagonal line convolution with no off-line trainable weights."""

    def __init__(self, channels: int, kernel_size: int, anti_diagonal: bool = False):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        mask = torch.eye(kernel_size)
        if anti_diagonal:
            mask = mask.flip(1)
        self.register_buffer("mask", mask[None, None], persistent=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the masked depthwise line kernel."""
        return F.conv2d(x, self.weight * self.mask, padding=self.padding, groups=x.shape[1])


class _SpatialStripGate(nn.Module):
    """Fourier-routed spatial gate for thin occlusion structures."""

    def __init__(self, kernel_size: int = 7, hidden: int = 8):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")
        padding = kernel_size // 2
        self.stem = nn.Sequential(nn.Conv2d(3, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        self.horizontal = nn.Conv2d(
            hidden, hidden, (1, kernel_size), padding=(0, padding), groups=hidden, bias=False
        )
        self.vertical = nn.Conv2d(
            hidden, hidden, (kernel_size, 1), padding=(padding, 0), groups=hidden, bias=False
        )
        self.diagonal = _MaskedDepthwiseConv(hidden, kernel_size)
        self.anti_diagonal = _MaskedDepthwiseConv(hidden, kernel_size, anti_diagonal=True)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.project = nn.Conv2d(hidden * 2, 1, 1)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        """Route four strip directions using local Fourier orientation weights."""
        mean = x.mean(1, keepdim=True)
        maximum = x.amax(1, keepdim=True)
        std = (x - mean).square().mean(1, keepdim=True).add(1e-6).sqrt()
        stats = self.stem(torch.cat((mean, maximum, std), dim=1))
        directional = torch.stack(
            (
                self.horizontal(stats),
                self.diagonal(stats),
                self.vertical(stats),
                self.anti_diagonal(stats),
            ),
            dim=1,
        )
        routed = (directional * orientation.unsqueeze(2)).sum(1)
        gate = torch.sigmoid(self.project(torch.cat((routed, self.local(stats)), 1)))
        scale = torch.tanh(self.gain).to(x.dtype)
        return x * (1.0 + scale * (2.0 * gate - 1.0))


class P2CFSAttention(nn.Module):
    """Shape-preserving channel-frequency-spatial attention for a P2 feature map.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels. Must equal ``c1`` because this is a residual P2 block.
        channel_ratio (float): Fraction sent to the channel/DCT branch.
        spatial_ratio (float): Fraction sent to the spatial strip branch. The remainder uses FFT.
        strip_kernel (int): Horizontal/vertical spatial kernel size.
        dct_k (int): Number of fixed low-frequency DCT descriptors per channel.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        channel_ratio: float = 0.375,
        spatial_ratio: float = 0.375,
        strip_kernel: int = 7,
        dct_k: int = 4,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"P2CFSAttention is shape-preserving, but received c1={c1}, c2={c2}")
        if not (0.0 < channel_ratio < 1.0 and 0.0 < spatial_ratio < 1.0):
            raise ValueError("channel_ratio and spatial_ratio must be in (0, 1)")

        channel_channels = max(round(c2 * channel_ratio), 1)
        spatial_channels = max(round(c2 * spatial_ratio), 1)
        frequency_channels = c2 - channel_channels - spatial_channels
        if frequency_channels < 1:
            raise ValueError(
                f"channel partitions leave no FFT channels: C={c2}, Rc={channel_ratio}, Rs={spatial_ratio}"
            )

        self.channels = (channel_channels, frequency_channels, spatial_channels)
        self.pre_norm = nn.BatchNorm2d(c2)
        self.channel_gate = _ChannelGate(dct_k=dct_k)
        self.spectral_gate = _SpectralGate()
        self.spatial_gate = _SpatialStripGate(strip_kernel)
        self.fuse = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2))

        # A small inner residual lets all branches learn once the outer Proto
        # projection opens. Exact model-level identity is enforced in SegmentP2CFS.
        nn.init.constant_(self.fuse[-1].weight, 0.1)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine P2 while preserving ``[B, C, H, W]`` and an identity shortcut."""
        normalized = self.pre_norm(x)
        channel, frequency, spatial = normalized.split(self.channels, dim=1)
        frequency, orientation = self.spectral_gate(frequency)
        refined = torch.cat(
            (self.channel_gate(channel), frequency, self.spatial_gate(spatial, orientation)), dim=1
        )
        return x + self.fuse(refined)
