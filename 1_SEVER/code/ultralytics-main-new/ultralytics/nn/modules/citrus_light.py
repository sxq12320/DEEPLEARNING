# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Latency-aware backbone stages and adaptive pyramid fusion for the Light citrus series.

The implementation follows two published design principles without nesting them
inside the original C3k2/CSP graph:

* FasterNet (CVPR 2023): spatial mixing is applied to only a fraction of the
  channels, reducing both arithmetic and memory traffic.
* AFPN (arXiv 2023, TCSVT 2024): adjacent scales are fused progressively and a
  spatial softmax resolves conflicting information at each pixel.

The resulting architecture keeps the operations regular (Conv/BN/SiLU, partial
3x3 convolution, nearest interpolation and softmax) and avoids custom CUDA
dependencies, deformable sampling, Mamba and persistent full-resolution streams.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("CitrusLightAFPN", "CitrusLightStage")


class _PartialConv3(nn.Module):
    """Apply a 3x3 convolution to one channel partition, as in FasterNet."""

    def __init__(self, channels: int, division: int = 4):
        super().__init__()
        if division < 2:
            raise ValueError(f"division must be at least 2, got {division}")
        self.mixed_channels = max(1, channels // division)
        self.untouched_channels = channels - self.mixed_channels
        self.partial_conv = nn.Conv2d(
            self.mixed_channels,
            self.mixed_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix one contiguous channel partition and preserve the remainder."""
        mixed, untouched = torch.split(x, (self.mixed_channels, self.untouched_channels), dim=1)
        return torch.cat((self.partial_conv(mixed), untouched), dim=1)


class _FasterResidual(nn.Module):
    """Partial spatial mixing followed by a compact channel MLP."""

    def __init__(self, channels: int, expansion: float = 2.0, division: int = 4, layer_scale: float = 0.1):
        super().__init__()
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        hidden = max(8, int(round(channels * expansion)))
        self.spatial = _PartialConv3(channels, division)
        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # A small non-zero scale keeps the initial function close to identity
        # while allowing every new convolution to receive gradients on step one.
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a near-identity residual update with full first-step gradients."""
        return x + self.layer_scale * self.project(self.expand(self.spatial(x)))


class CitrusLightStage(nn.Module):
    """Non-CSP feature-extraction stage used to replace a complete C3k2 stage.

    Args:
        c1: Input channels.
        c2: Output channels.
        blocks: Number of partial-convolution residual blocks.
        expansion: Channel expansion ratio inside each residual block.
        division: Fraction denominator for the spatially mixed channels.
        layer_scale: Initial residual scale. It is deliberately non-zero to
            avoid the cold-branch gradient problem observed in G_0830.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        blocks: int = 1,
        expansion: float = 2.0,
        division: int = 4,
        layer_scale: float = 0.1,
    ):
        super().__init__()
        if blocks < 1:
            raise ValueError(f"blocks must be positive, got {blocks}")
        self.input_project = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(
            *(_FasterResidual(c2, expansion, division, layer_scale) for _ in range(int(blocks)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project once and process the stage without CSP concatenation."""
        return self.blocks(self.input_project(x))


class _AdaptiveFuse2(nn.Module):
    """Inject one adjacent-scale source through a near-identity adaptive residual.

    A plain two-way softmax starts from a 0.5/0.5 average and repeatedly attenuates
    the destination feature as it passes through the pyramid.  That behavior is
    particularly risky for weak tiny-object responses.  The revised fusion keeps
    the destination path close to identity and learns how much source information
    to inject.  A non-zero initial mix keeps gradients flowing to the source and
    gate on the first optimization step.
    """

    def __init__(self, channels: int, gate_channels: int = 8, initial_mix: float = 0.1):
        super().__init__()
        if gate_channels < 1:
            raise ValueError(f"gate_channels must be positive, got {gate_channels}")
        if not 0.0 < initial_mix < 1.0:
            raise ValueError(f"initial_mix must be in (0, 1), got {initial_mix}")
        # A single projection keeps AFPN's per-pixel spatial selection while avoiding the kernel-launch and memory
        # traffic of two compression blocks, a third scoring layer and a post-fusion refinement block.
        self.weight = nn.Conv2d(channels * 2, 2, kernel_size=1)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        initial_logit = torch.logit(torch.tensor(float(initial_mix)))
        self.mix_logit = nn.Parameter(initial_logit.clone())

    def forward(self, destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Preserve ``destination`` and inject spatially selected ``source`` evidence."""
        if destination.shape != source.shape:
            raise ValueError(
                f"Adaptive fusion requires identical shapes, got {destination.shape} and {source.shape}"
            )
        weights = self.weight(torch.cat((destination, source), dim=1)).softmax(dim=1)
        selected = destination * weights[:, 0:1] + source * weights[:, 1:2]
        mix = self.mix_logit.sigmoid()
        return destination + mix * (selected - destination)


class _DownProject(nn.Module):
    """Low-parameter adjacent-level transition using projection plus anti-alias pooling."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.project = Conv(c1, c2, k=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project channels before one regular 2x average downsampling."""
        return self.pool(self.project(x))


class CitrusLightAFPN(nn.Module):
    """Progressive gather-distribute neck for P2/P3/P4/P5 features.

    The gather path introduces P2 only once at P3 resolution, then progressively
    aggregates adjacent levels until P5. The distribute path returns semantics
    from P5 to P3 using spatially adaptive two-way fusion. This replaces the four
    concatenate-plus-C3k2 nodes of PAN with bounded-width additive fusion.
    """

    def __init__(
        self,
        in_channels: list[int] | tuple[int, int, int, int],
        out_channels: list[int] | tuple[int, int, int] = (64, 128, 256),
        gate_channels: int = 8,
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"CitrusLightAFPN expects P2/P3/P4/P5 channels, got {in_channels}")
        if len(out_channels) != 3:
            raise ValueError(f"CitrusLightAFPN expects P3/P4/P5 output channels, got {out_channels}")
        c2, c3, c4, c5 = (int(value) for value in in_channels)
        o3, o4, o5 = (int(value) for value in out_channels)

        # Progressive low-to-high gather. Dense stride convolutions are used
        # deliberately: FasterNet shows they can have better realized throughput
        # than chains of nominally cheaper depthwise operators.
        self.p2_to_p3 = _DownProject(c2, o3)
        self.p3_lateral = Conv(c3, o3, k=1)
        self.gather_p3 = _AdaptiveFuse2(o3, gate_channels)

        self.p3_to_p4 = _DownProject(o3, o4)
        self.p4_lateral = Conv(c4, o4, k=1)
        self.gather_p4 = _AdaptiveFuse2(o4, gate_channels)

        self.p4_to_p5 = _DownProject(o4, o5)
        self.p5_lateral = Conv(c5, o5, k=1)
        self.gather_p5 = _AdaptiveFuse2(o5, gate_channels)

        # High-to-low semantic distribution. Nearest interpolation avoids the
        # content-reassembly overhead that made previous CARAFE variants slow.
        self.p5_to_p4 = Conv(o5, o4, k=1)
        self.distribute_p4 = _AdaptiveFuse2(o4, gate_channels)
        self.p4_to_p3 = Conv(o4, o3, k=1)
        self.distribute_p3 = _AdaptiveFuse2(o3, gate_channels)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Return adaptively fused P3, P4 and P5 tensors."""
        if len(features) != 4:
            raise ValueError(f"Expected four pyramid inputs, received {len(features)}")
        p2, p3, p4, p5 = features

        gathered_p3 = self.gather_p3(self.p3_lateral(p3), self.p2_to_p3(p2))
        gathered_p4 = self.gather_p4(self.p4_lateral(p4), self.p3_to_p4(gathered_p3))
        gathered_p5 = self.gather_p5(self.p5_lateral(p5), self.p4_to_p5(gathered_p4))

        semantic_p4 = F.interpolate(self.p5_to_p4(gathered_p5), size=gathered_p4.shape[-2:], mode="nearest")
        output_p4 = self.distribute_p4(gathered_p4, semantic_p4)
        semantic_p3 = F.interpolate(self.p4_to_p3(output_p4), size=gathered_p3.shape[-2:], mode="nearest")
        output_p3 = self.distribute_p3(gathered_p3, semantic_p3)
        return [output_p3, output_p4, gathered_p5]
