# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Task-specific operators for the citrus SAGE-v1, SAGE-v2 and SAGE-v3 series.

The legacy :class:`CitrusSAGEFuse` keeps its narrow mask-only path so completed
SAGE00--04 experiments remain reproducible.  SAGE-v2 is the architecture-level
redesign used by SAGE10--17:

* :class:`C3k2SAGE` reconstructs the low-resolution P4/P5 backbone stages with
  a pretraining-compatible context residual.
* :class:`CitrusSAGEPyramid` progressively corrects P5, P4 and P3 features, so
  the change reaches detection, classification, mask coefficients and masks.
* A shared four-state topology map represents context, fruit interior,
  boundary and inter-instance separator evidence.  It supervises the fusion
  gate instead of introducing another unsupervised attention block.
* The conservative mode preserves the complete pretrained PAN as an identity
  base; the replacement mode is retained as an explicit aggressive ablation.

Both versions avoid deformable sampling, ``grid_sample``, ``unfold``, dynamic
kernels, full-resolution attention matrices and Mamba.  The hot path uses
ordinary Conv/BN/SiLU kernels, nearest-neighbour resizing, PixelUnshuffle,
low-resolution RepVGGDW and elementwise arithmetic.

SAGE-v3 keeps that operator contract but replaces generic context residuals
with two task-specific architectural operations:

* :class:`C3k2SAGEShape` reconstructs P4/P5 with one low-resolution axial
  shape-context path.  It adapts the rectangular context idea of PKINet and
  CGRSeg without copying their multi-kernel stages or decoder dependencies.
* :class:`CitrusSAGEInnovationPyramid` treats top-down semantics as a prediction
  and the aligned backbone tensor as a measurement.  Their innovation, local
  contrast and semantic state are fused progressively before correcting the
  complete pretrained PAN.  The four-state topology map is the only gate.

The optional style-statistics swap follows CrossNorm's training-distribution
idea.  It is parameter-free and exactly inactive during evaluation, making it
an isolated colour-reliance ablation rather than an inference-time module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2, RepVGGDW
from .conv import Conv

__all__ = (
    "C3k2SAGE",
    "C3k2SAGEShape",
    "CitrusSAGEFuse",
    "CitrusSAGEInnovationPyramid",
    "CitrusSAGEPyramid",
)


class C3k2SAGE(C3k2):
    """Pretraining-compatible deep backbone stage with a low-resolution context residual.

    The inherited C3k2 path deliberately keeps the same parameter names as the
    YOLO11 checkpoint.  A RepViT-style re-parameterizable spatial mixer and a
    compact channel mixer are added after that path.  SAGE uses this block only
    at P4/P5: shallow P2/P3 geometry is not repeatedly filtered and the extra
    spatial work is kept away from expensive high-resolution maps.

    This is a backbone intervention, not an attention plug-in.  It changes the
    feature extractor's deep-stage computation while retaining all compatible
    pretrained C3k2 tensors on the small citrus dataset.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        expansion: float = 0.5,
        initial_scale: float = 0.1,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if not 0.0 < initial_scale <= 1.0:
            raise ValueError(f"initial_scale must be in (0, 1], got {initial_scale}")
        hidden = max(16, int(round(c2 * expansion)))
        self.sage_token_mixer = RepVGGDW(c2)
        self.sage_channel_mixer = nn.Sequential(
            Conv(c2, hidden, k=1),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.sage_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the official CSP feature plus a bounded deep-context update."""
        base = super().forward(x)
        update = self.sage_channel_mixer(self.sage_token_mixer(base))
        return base + torch.tanh(self.sage_scale) * update


class _SAGEStyleSwap(nn.Module):
    """Exchange instance statistics during training and remain identity at inference.

    The operation is a conservative, feature-space adaptation of CrossNorm.  A
    convex blend is used instead of an unconditional full replacement because
    immature-fruit colour still contains useful evidence; the experiment asks
    whether reducing *over-reliance* on green appearance improves robustness.
    """

    def __init__(self, probability: float = 0.0, mix: float = 0.5, eps: float = 1e-5):
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"mix must be in [0, 1], got {mix}")
        self.probability = float(probability)
        self.mix = float(mix)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a style-swapped feature only for eligible training batches."""
        if not self.training or self.probability == 0.0 or x.shape[0] < 2:
            return x
        if torch.rand((), device=x.device) >= self.probability:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        standard_deviation = (variance + self.eps).sqrt()
        permutation = torch.randperm(x.shape[0], device=x.device)
        normalized = (x - mean) / standard_deviation
        swapped = normalized * standard_deviation[permutation] + mean[permutation]
        return torch.lerp(x, swapped, self.mix)


class C3k2SAGEShape(C3k2):
    """Pretraining-compatible C3k2 stage with one axial shape-context correction.

    Only low-resolution P4/P5 stages should use this block.  The official C3k2
    path and tensor names remain intact, while a narrow branch combines local
    3x3 evidence with horizontal/vertical context.  This targets long leaf or
    branch occluders and deeply concave visible fruit masks without installing
    PKINet's five-kernel chain or CGRSeg's full decoder.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        context_ratio: float = 0.25,
        axial_kernel: int = 11,
        initial_scale: float = 0.1,
        style_probability: float = 0.0,
        style_mix: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        if not 0.0 < context_ratio <= 1.0:
            raise ValueError(f"context_ratio must be in (0, 1], got {context_ratio}")
        if axial_kernel < 3 or axial_kernel % 2 == 0:
            raise ValueError(f"axial_kernel must be an odd integer >= 3, got {axial_kernel}")
        if not 0.0 < initial_scale <= 1.0:
            raise ValueError(f"initial_scale must be in (0, 1], got {initial_scale}")

        hidden = max(16, int(round(c2 * context_ratio)))
        self.sage_style = _SAGEStyleSwap(style_probability, style_mix)
        self.sage_reduce = Conv(c2, hidden, k=1)
        self.sage_local = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.sage_context = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=(1, axial_kernel),
                padding=(0, axial_kernel // 2),
                groups=hidden,
                bias=False,
            ),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=(axial_kernel, 1),
                padding=(axial_kernel // 2, 0),
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
        )
        self.sage_expand = nn.Sequential(
            nn.Conv2d(hidden, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.sage_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the official feature plus a bounded shape-context correction."""
        base = self.sage_style(super().forward(x))
        reduced = self.sage_reduce(base)
        local = self.sage_local(reduced)
        context_gate = torch.sigmoid(self.sage_context(reduced))
        update = self.sage_expand(local * context_gate)
        return base + torch.tanh(self.sage_scale) * update


class _SAGEInnovationCell(nn.Module):
    """Fuse semantic prediction, cross-scale innovation and local contrast."""

    def __init__(self, channels: int, initial_scale: float = 0.1):
        super().__init__()
        self.fuse = Conv(channels * 3, channels, k=3)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, measurement: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Correct an aligned semantic prediction using measurement residuals."""
        if prediction.shape[-2:] != measurement.shape[-2:]:
            prediction = F.interpolate(prediction, size=measurement.shape[-2:], mode="nearest")
        innovation = measurement - prediction
        contrast = measurement - F.avg_pool2d(measurement, kernel_size=3, stride=1, padding=1)
        correction = self.fuse(torch.cat((prediction, innovation, contrast), dim=1))
        return prediction + torch.tanh(self.scale) * correction


class CitrusSAGEInnovationPyramid(nn.Module):
    """Progressively correct the pretrained PAN with task-supervised innovations.

    Inputs are C2/C3/C4/C5 and native PAN P3/P4/P5.  The native PAN remains the
    identity base.  A compact route estimates a coarse semantic state at P5,
    compares its predictions with aligned P4/P3 measurements, and transports
    only the resulting corrections.  P2 contributes losslessly rearranged local
    contrast at P3, where one four-state topology map separates context, fruit
    interior, visible boundary and neighbouring-instance separator evidence.
    """

    topology_classes = 4

    def __init__(
        self,
        in_channels: list[int] | tuple[int, ...],
        out_channels: list[int] | tuple[int, int, int],
        route_channels: int = 32,
        initial_scale: float = 0.1,
    ):
        super().__init__()
        if len(in_channels) != 7:
            raise ValueError(f"CitrusSAGEInnovationPyramid expects seven channels, got {in_channels}")
        if len(out_channels) != 3:
            raise ValueError(f"Expected P3/P4/P5 output channels, got {out_channels}")
        if route_channels < 8:
            raise ValueError(f"route_channels must be at least 8, got {route_channels}")
        if not 0.0 < initial_scale <= 1.0:
            raise ValueError(f"initial_scale must be in (0, 1], got {initial_scale}")

        c2, c3, c4, c5 = (int(value) for value in in_channels[:4])
        o3, o4, o5 = (int(value) for value in out_channels)
        if tuple(int(value) for value in in_channels[4:]) != (o3, o4, o5):
            raise ValueError(f"PAN channels {in_channels[4:]} must equal outputs {(o3, o4, o5)}")

        width = int(route_channels)
        self.measure5 = Conv(c5, width, k=1)
        self.measure4 = Conv(c4, width, k=1)
        self.measure3 = Conv(c3, width, k=1)
        self.predict4 = Conv(width, width, k=1)
        self.predict3 = Conv(width, width, k=1)
        self.innovation4 = _SAGEInnovationCell(width, initial_scale)
        self.innovation3 = _SAGEInnovationCell(width, initial_scale)

        self.p2_rearrange = nn.PixelUnshuffle(2)
        self.p2_project = Conv(c2 * 4, width, k=1)
        self.topology_predictor = nn.Conv2d(width * 2, self.topology_classes, kernel_size=1)
        nn.init.constant_(self.topology_predictor.bias[0], 1.0)
        nn.init.constant_(self.topology_predictor.bias[1:], -1.0)

        self.output3 = self._small_output(width, o3)
        self.output4 = self._small_output(width, o4)
        self.output5 = self._small_output(width, o5)
        self.residual_scales = nn.Parameter(torch.full((3,), float(initial_scale)))

    @staticmethod
    def _small_output(c1: int, c2: int) -> nn.Sequential:
        """Create a small live residual projection."""
        projection = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        nn.init.normal_(projection.weight, mean=0.0, std=1e-3)
        return nn.Sequential(projection, nn.BatchNorm2d(c2))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Return corrected P3/P4/P5 tensors and the shared topology logits."""
        if len(features) != 7:
            raise ValueError(f"Expected C2/C3/C4/C5/PAN-P3/P4/P5, received {len(features)} tensors")
        c2, c3, c4, c5, base3, base4, base5 = features

        state5 = self.measure5(c5)
        state4 = self.innovation4(self.measure4(c4), self.predict4(state5))
        state3 = self.innovation3(self.measure3(c3), self.predict3(state4))

        p2_contrast = c2 - F.avg_pool2d(c2, kernel_size=3, stride=1, padding=1)
        detail3 = self.p2_project(self.p2_rearrange(p2_contrast))
        if detail3.shape[-2:] != state3.shape[-2:]:
            raise ValueError(f"P2 route produced {detail3.shape[-2:]}, expected {state3.shape[-2:]}")

        topology_logits = self.topology_predictor(torch.cat((state3, detail3), dim=1))
        probabilities = topology_logits.softmax(dim=1)
        fruit_gate = probabilities[:, 1:3].sum(dim=1, keepdim=True)
        structure_gate = probabilities[:, 2:4].sum(dim=1, keepdim=True)
        state3 = (0.5 + 0.5 * fruit_gate) * state3 + structure_gate * detail3
        state4 = (0.5 + 0.5 * F.interpolate(fruit_gate, size=state4.shape[-2:], mode="nearest")) * state4
        state5 = (0.5 + 0.5 * F.interpolate(fruit_gate, size=state5.shape[-2:], mode="nearest")) * state5

        scales = torch.tanh(self.residual_scales)
        output3 = base3 + scales[0] * self.output3(state3)
        output4 = base4 + scales[1] * self.output4(state4)
        output5 = base5 + scales[2] * self.output5(state5)
        return [output3, output4, output5, topology_logits]


class _SAGESimilarityFuse(nn.Module):
    """Align an adjacent semantic source to a local feature with one similarity map.

    This retains PIDNet's PagFM principle but uses nearest resize and ordinary
    1x1 projections.  It intentionally avoids deformable sampling, grid_sample,
    CARAFE and full channel-wise attention.
    """

    def __init__(self, channels: int, embedding_channels: int = 16):
        super().__init__()
        hidden = max(8, min(int(embedding_channels), channels))
        self.local_embedding = Conv(channels, hidden, k=1, act=False)
        self.source_embedding = Conv(channels, hidden, k=1, act=False)

    def forward(self, local: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Inject source evidence only where projected adjacent-scale features agree."""
        if source.shape[-2:] != local.shape[-2:]:
            source = F.interpolate(source, size=local.shape[-2:], mode="nearest")
        similarity = torch.sigmoid(
            (self.local_embedding(local) * self.source_embedding(source)).mean(dim=1, keepdim=True)
        )
        return local + similarity * source


class CitrusSAGEPyramid(nn.Module):
    """Topology-guided residual pyramid for citrus instance segmentation.

    Seven-input mode consumes C2/C3/C4/C5 and the complete pretrained YOLO PAN
    P3/P4/P5.  The PAN tensors form identity bases; a narrow adjacent-scale path
    learns corrections that affect detection, mask coefficients and prototypes.
    This avoids the destructive all-random neck replacement observed in the
    Light experiments while still changing the fusion topology.

    The P2 route is rearranged to P3 with PixelUnshuffle.  A single four-state
    topology map (context/interior/boundary/separator) controls whether P3 takes
    semantic or geometric evidence.  The same logits receive task supervision in
    ``SegmentCitrusSAGEV2``; the gate is therefore explainable rather than a
    generic unsupervised attention map.
    """

    topology_classes = 4

    def __init__(
        self,
        in_channels: list[int] | tuple[int, ...],
        out_channels: list[int] | tuple[int, int, int],
        route_channels: int = 32,
        embedding_channels: int = 16,
        initial_scale: float = 0.1,
    ):
        super().__init__()
        if len(in_channels) not in {4, 7}:
            raise ValueError(f"CitrusSAGEPyramid expects 4 or 7 channels, got {in_channels}")
        if len(out_channels) != 3:
            raise ValueError(f"Expected P3/P4/P5 output channels, got {out_channels}")
        if route_channels < 8:
            raise ValueError(f"route_channels must be at least 8, got {route_channels}")
        if not 0.0 < initial_scale <= 1.0:
            raise ValueError(f"initial_scale must be in (0, 1], got {initial_scale}")

        c2, c3, c4, c5 = (int(value) for value in in_channels[:4])
        o3, o4, o5 = (int(value) for value in out_channels)
        self.residual_mode = len(in_channels) == 7
        if self.residual_mode and tuple(int(value) for value in in_channels[4:]) != (o3, o4, o5):
            raise ValueError(
                f"PAN channels {in_channels[4:]} must equal requested outputs {(o3, o4, o5)}"
            )

        width = int(route_channels)
        self.c5_project = Conv(c5, width, k=1)
        self.c4_project = Conv(c4, width, k=1)
        self.c3_project = Conv(c3, width, k=1)
        self.p5_to_p4 = Conv(width, width, k=1)
        self.p4_fuse = _SAGESimilarityFuse(width, embedding_channels)
        self.p4_refine = Conv(width, width, k=3)
        self.p4_to_p3 = Conv(width, width, k=1)
        self.p3_fuse = _SAGESimilarityFuse(width, embedding_channels)
        self.p3_refine = Conv(width, width, k=3)

        # Space-to-depth preserves every P2 sample while moving all expensive
        # processing to stride 8.  The local-contrast residual is insensitive to
        # absolute green colour and highlights fruit/leaf structural changes.
        self.p2_rearrange = nn.PixelUnshuffle(2)
        self.p2_project = Conv(c2 * 4, width, k=1)
        self.topology_predictor = nn.Conv2d(width * 2, self.topology_classes, kernel_size=1)

        self.base3 = None if self.residual_mode else Conv(c3, o3, k=1)
        self.base4 = None if self.residual_mode else Conv(c4, o4, k=1)
        self.base5 = None if self.residual_mode else Conv(c5, o5, k=1)
        self.output3 = self._small_output(width, o3)
        self.output4 = self._small_output(width, o4)
        self.output5 = self._small_output(width, o5)
        self.residual_scales = nn.Parameter(torch.full((3,), float(initial_scale)))

    @staticmethod
    def _small_output(c1: int, c2: int) -> nn.Sequential:
        """Create a near-zero projection with live first-step gradients."""
        projection = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        nn.init.normal_(projection.weight, mean=0.0, std=1e-3)
        return nn.Sequential(projection, nn.BatchNorm2d(c2))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Return corrected P3/P4/P5 features and shared four-state topology logits."""
        if len(features) not in {4, 7}:
            raise ValueError(f"Expected four backbone or seven backbone/PAN tensors, received {len(features)}")
        c2, c3, c4, c5 = features[:4]
        if self.residual_mode:
            base3, base4, base5 = features[4:]
        else:
            base3, base4, base5 = self.base3(c3), self.base4(c4), self.base5(c5)

        route5 = self.c5_project(c5)
        source4 = self.p5_to_p4(route5)
        route4 = self.p4_refine(self.p4_fuse(self.c4_project(c4), source4))
        source3 = self.p4_to_p3(route4)
        route3 = self.p3_refine(self.p3_fuse(self.c3_project(c3), source3))

        # Local contrast is computed before rearrangement; no colour-space or
        # learned enhancement front-end can hallucinate mask boundaries.
        p2_contrast = c2 - F.avg_pool2d(c2, kernel_size=3, stride=1, padding=1)
        detail3 = self.p2_project(self.p2_rearrange(p2_contrast))
        if detail3.shape[-2:] != route3.shape[-2:]:
            raise ValueError(f"P2 route produced {detail3.shape[-2:]}, expected {route3.shape[-2:]}")

        topology_logits = self.topology_predictor(torch.cat((route3, detail3), dim=1))
        topology_probability = topology_logits.softmax(dim=1)
        structural_gate = topology_probability[:, 2:4].sum(dim=1, keepdim=True)
        route3 = route3 + structural_gate * detail3

        scales = torch.tanh(self.residual_scales)
        output3 = base3 + scales[0] * self.output3(route3)
        output4 = base4 + scales[1] * self.output4(route4)
        output5 = base5 + scales[2] * self.output5(route5)
        return [output3, output4, output5, topology_logits]


class CitrusSAGEFuse(nn.Module):
    """Create a task-decoupled P3 mask feature from C2/C3/C4/C5 and PAN-P3.

    Stages form a causal ablation chain:

    1. Adjacent P5-to-P4-to-P3 semantic relay.
    2. Stage 1 plus one regular-convolution local alignment at P3.
    3. Stage 2 plus lossless P2-to-P3 geometry rearrangement.
    4. Stage 3 plus semantic-geometry agreement gating.

    Args:
        channels: Channels of C2, C3, C4, C5 and the native PAN-P3 tensor.
        stage: Progressive ablation stage in ``[1, 4]``.
        width: Internal route width.  Thirty-two is used for the nano model.
        initial_scale: Initial magnitude of the mask-only residual.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int, int, int],
        stage: int = 4,
        width: int = 32,
        initial_scale: float = 0.1,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"CitrusSAGEFuse expects C2/C3/C4/C5/PAN-P3 channels, got {channels}")
        if stage not in {1, 2, 3, 4}:
            raise ValueError(f"SAGE stage must be in [1, 4], got {stage}")
        if width < 8:
            raise ValueError(f"SAGE width must be at least 8, got {width}")
        if not 0.0 < initial_scale <= 1.0:
            raise ValueError(f"initial_scale must be in (0, 1], got {initial_scale}")

        c2, c3, c4, c5, pan_c3 = (int(value) for value in channels)
        self.stage = int(stage)

        # Two adjacent semantic transitions.  Only P4 and P3 receive a regular
        # 3x3, keeping spatial work off the expensive P2 feature map.
        self.p5_project = Conv(c5, width, k=1)
        self.p4_project = Conv(c4, width, k=1)
        self.p4_refine = Conv(width, width, k=3)
        self.p3_project = Conv(c3, width, k=1)
        if self.stage == 1:
            self.p3_fuse = Conv(width, width, k=3)
        else:
            # A single dense 3x3 locally re-encodes the resized context together
            # with the P3 reference.  Dense kernels are intentional: on common
            # GPUs one regular kernel is faster than a chain of tiny DW/PW ops.
            self.p3_fuse = Conv(width * 2, width, k=3)

        self.detail_project = None
        self.detail_scale = None
        self.agreement = None
        if self.stage >= 3:
            self.detail_project = nn.Sequential(nn.PixelUnshuffle(2), Conv(c2 * 4, width, k=1))
            self.detail_scale = nn.Parameter(torch.tensor(float(initial_scale)))
        if self.stage == 4:
            self.agreement = nn.Conv2d(width, 1, kernel_size=1)
            nn.init.zeros_(self.agreement.weight)
            nn.init.constant_(self.agreement.bias, -1.3862944)  # sigmoid -> 0.2: conservative P2 admission

        # Linear output projection keeps the residual unbiased.  A small scalar
        # makes the initial model close to the pretrained official PAN.
        self.output_project = nn.Sequential(
            nn.Conv2d(width, pan_c3, kernel_size=1, bias=False),
            nn.BatchNorm2d(pan_c3),
        )
        self.route_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    @staticmethod
    def _nearest(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize with one accelerator-friendly nearest-neighbour kernel."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return a refined P3 tensor used only by the mask-prototype branch."""
        if len(features) != 5:
            raise ValueError(f"Expected C2/C3/C4/C5/PAN-P3 tensors, received {len(features)}")
        c2, c3, c4, c5, pan_p3 = features

        p4 = self.p4_project(c4)
        p4 = self.p4_refine(p4 + self._nearest(self.p5_project(c5), p4.shape[-2:]))

        p3_reference = self.p3_project(c3)
        p4_at_p3 = self._nearest(p4, p3_reference.shape[-2:])
        if self.stage == 1:
            evidence = self.p3_fuse(p3_reference + p4_at_p3)
        else:
            evidence = self.p3_fuse(torch.cat((p3_reference, p4_at_p3), dim=1))

        if self.stage >= 3:
            geometry = self.detail_project(c2)
            if geometry.shape[-2:] != evidence.shape[-2:]:
                raise ValueError(
                    f"PixelUnshuffle P2 geometry has shape {geometry.shape[-2:]}, expected {evidence.shape[-2:]}"
                )
            if self.stage == 4:
                # A single spatial value is sufficient: it gates locations, not
                # channels, and avoids a high-resolution channel-attention stack.
                geometry = geometry * torch.sigmoid(self.agreement(evidence * geometry))
            evidence = evidence + torch.tanh(self.detail_scale) * geometry

        residual = self.output_project(evidence)
        return pan_p3 + torch.tanh(self.route_scale) * residual
