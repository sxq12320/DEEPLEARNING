# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Shape-semantic bilateral backbone components for immature-citrus segmentation.

The design is intentionally one coherent high-resolution path rather than a stack
of generic attention blocks. A stride-4 shape stream keeps local pixel differences,
deep semantic stages gate which structures are relevant, and a reversible spatial
rearrangement injects the selected evidence into P3 without adding a dense P2 head.

The implementation is grounded in three published/open-source designs:

* PiDiNet central pixel-difference convolution for lightweight edge evidence.
* Gated-SCNN semantic gating of a shallow shape stream.
* PIDNet/Lite-HRNet persistent high-resolution representations and selective fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("CitrusShapeFusion", "CitrusShapeStream", "CitrusStructureStem")


class _CenterDifferenceConv(nn.Module):
    """Learn a 3x3 convolution after removing its local DC response.

    This is the central pixel-difference operator used by PiDiNet. Uniform local
    colour produces zero difference, so the branch is biased toward boundaries
    and shape changes instead of absolute green intensity.
    """

    def __init__(self, c1: int, c2: int, stride: int = 1, groups: int = 1, activate: bool = True):
        super().__init__()
        if c1 % groups or c2 % groups:
            raise ValueError(f"Channels ({c1}, {c2}) must be divisible by groups={groups}")
        self.stride = stride
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(c2, c1 // groups, 3, 3))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if activate else nn.Identity()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the regular kernel minus its summed centre response."""
        regular = F.conv2d(x, self.weight, stride=self.stride, padding=1, groups=self.groups)
        centre_weight = self.weight.sum(dim=(2, 3), keepdim=True)
        centre = F.conv2d(x, centre_weight, stride=self.stride, groups=self.groups)
        return self.act(self.bn(regular - centre))


class _ShapeUpdate(nn.Module):
    """Efficient structural update used inside the persistent P2 stream."""

    def __init__(self, channels: int, mode: str = "pdc"):
        super().__init__()
        if mode not in {"pdc", "conv"}:
            raise ValueError(f"Shape update mode must be 'pdc' or 'conv', got {mode!r}")
        self.spatial = (
            _CenterDifferenceConv(channels, channels, groups=channels)
            if mode == "pdc"
            else Conv(channels, channels, k=3, g=channels)
        )
        self.mix = Conv(channels, channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a local structure update without altering resolution."""
        return self.mix(self.spatial(x))


class CitrusStructureStem(nn.Module):
    """RGB plus achromatic pixel-difference stem with the usual YOLO stride.

    The RGB path retains useful appearance cues. In parallel, a fixed luminance
    projection feeds a learnable centre-difference operator. Concatenation occurs
    before a 1x1 mixer, making this an input-level structural bias rather than an
    irreversible conversion of the whole image to grayscale.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, structure_ratio: float = 0.25):
        super().__init__()
        if c1 != 3:
            raise ValueError(f"CitrusStructureStem expects an RGB tensor, got {c1} channels")
        structure_channels = min(max(round(c2 * structure_ratio), 4), c2 - 1)
        rgb_channels = c2 - structure_channels
        self.rgb = Conv(c1, rgb_channels, k=k, s=s)
        self.structure = _CenterDifferenceConv(1, structure_channels, stride=s)
        self.fuse = Conv(c2, c2, k=1)
        self.register_buffer("luminance", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse learned RGB appearance with an explicitly achromatic structure path."""
        luminance = (x * self.luminance.to(dtype=x.dtype)).sum(dim=1, keepdim=True)
        return self.fuse(torch.cat((self.rgb(x), self.structure(luminance)), dim=1))


class CitrusShapeStream(nn.Module):
    """Persistent stride-4 stream gated by progressively deeper semantics.

    Args:
        channels: P2, P3, P4 and P5 channel counts.
        out_channels: Width of the persistent P2 representation.
        mode: ``pdc`` for centre-difference updates or ``conv`` for the causal control.
        semantic_levels: Number of semantic stages to use, from P3 upward (1--3).

    Semantic tensors only determine the gate on structural updates; they are not
    copied into P2. This follows Gated-SCNN's separation of shape and appearance
    and avoids uniformly flooding tiny boundaries with low-frequency canopy context.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int, int],
        out_channels: int,
        mode: str = "pdc",
        semantic_levels: int = 3,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"CitrusShapeStream expects P2/P3/P4/P5 channels, got {channels}")
        if semantic_levels not in {1, 2, 3}:
            raise ValueError(f"semantic_levels must be 1, 2, or 3, got {semantic_levels}")
        self.semantic_levels = semantic_levels
        self.detail_stem = Conv(channels[0], out_channels, k=1)
        mid_channels = max(out_channels // 4, 8)
        selected_channels = list(channels[1 : 1 + semantic_levels])
        self.detail_keys = nn.ModuleList(Conv(out_channels, mid_channels, k=1, act=False) for _ in selected_channels)
        self.semantic_queries = nn.ModuleList(Conv(c, mid_channels, k=1, act=False) for c in selected_channels)
        self.updates = nn.ModuleList(_ShapeUpdate(out_channels, mode) for _ in selected_channels)
        self.stage_scales = nn.Parameter(torch.ones(semantic_levels))
        self.out = Conv(out_channels, out_channels, k=3, g=out_channels)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Filter P2 structure using P3-to-P5 semantic agreement."""
        if len(features) != 4:
            raise ValueError(f"CitrusShapeStream expects four tensors, got {len(features)}")
        detail = self.detail_stem(features[0])
        size = detail.shape[-2:]
        for index, semantic in enumerate(features[1 : 1 + self.semantic_levels]):
            key = self.detail_keys[index](detail)
            query = F.interpolate(
                self.semantic_queries[index](semantic), size=size, mode="bilinear", align_corners=False
            )
            agreement = torch.sigmoid((key * query).sum(dim=1, keepdim=True) / math.sqrt(key.shape[1]))
            scale = torch.tanh(self.stage_scales[index])
            detail = detail + scale * agreement * self.updates[index](detail)
        return self.out(detail)


class CitrusShapeFusion(nn.Module):
    """Selectively inject high-resolution shape evidence into the semantic P3 path.

    PixelUnshuffle retains every P2 sample while aligning it to P3. A PIDNet-style
    feature-agreement gate and a learned boundary gate jointly reject leaf texture.
    The residual is zero at initialization, preserving the pretrained semantic path.
    """

    def __init__(self, channels: list[int] | tuple[int, int], mid_channels: int = 16):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"CitrusShapeFusion expects [P3, shape] channels, got {channels}")
        semantic_channels, shape_channels = channels
        mid_channels = max(int(mid_channels), 4)
        self.shape_to_p3 = nn.Sequential(nn.PixelUnshuffle(2), Conv(shape_channels * 4, semantic_channels, k=1))
        self.semantic_key = Conv(semantic_channels, mid_channels, k=1, act=False)
        self.shape_query = Conv(semantic_channels, mid_channels, k=1, act=False)
        self.boundary = nn.Sequential(
            _CenterDifferenceConv(shape_channels, shape_channels, groups=shape_channels),
            nn.Conv2d(shape_channels, 1, 1),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1, semantic_channels, 1, 1))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return P3 plus semantic-filtered structural evidence."""
        semantic, shape = features
        shape_p3 = self.shape_to_p3(shape)
        if shape_p3.shape[-2:] != semantic.shape[-2:]:
            raise ValueError(f"Shape path produced {shape_p3.shape[-2:]}, expected P3 {semantic.shape[-2:]}")
        key = self.semantic_key(semantic)
        query = self.shape_query(shape_p3)
        agreement = torch.sigmoid((key * query).sum(dim=1, keepdim=True) / math.sqrt(key.shape[1]))
        boundary = F.max_pool2d(torch.sigmoid(self.boundary(shape)), kernel_size=2, stride=2)
        gate = agreement * (0.5 + 0.5 * boundary)
        return semantic + torch.tanh(self.residual_scale) * gate * shape_p3
