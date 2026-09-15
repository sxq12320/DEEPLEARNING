# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V4: control-theoretic neck corrections for immature citrus instance segmentation.

Task adaptations of classical automatic-control structures (PID control law,
Luenberger state observer, phase-lead compensation, cascade control, anti-alias
filtering). These are feature-space analogies, not literal controllers.

Design constraints learned from previous generations:
- E V2 showed that restructuring the backbone or adding a global context hub
  collapses training (-3 to -5 pt). Every module here therefore starts EXACTLY
  at identity through zero-initialised per-channel gates (ReZero-style), so the
  correction term must earn its gain from data. Module identity does NOT imply
  whole-network identity: replacing concat changes widths and initialization.
- The full-loss suite (NWD + Dice + Boundary + Freq) hurt by -2.3 to -3.7 pt,
  so no module touches the training objective; corrections are structural only.
- G03 showed wholesale frequency-domain neck replacement is too aggressive, so
  the filter idea is applied only as a single-point anti-alias fusion.

Pain points addressed (quantified in results/_analysis):
- Recall is the bottleneck (P ~0.91 vs R ~0.75); 53.3% of instances are
  COCO-small and tiny recall stays below 0.21 at global inference.
- 31% of instances have a neighbour within 2 px and 17.6% of masks are
  strongly concave from stripe-like branch/leaf occlusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = (
    "EV4PIDFusion",
    "EV4ObserverGate",
    "EV4PhaseLead",
    "EV4IntegralContext",
    "EV4FilterFuse",
    "EV4CascadeRefine",
    "EV4ChromaFront",
    "EV4LapFront",
    "EV4ReverseRefine",
    "EV4RadialVote",
    "EV4ScaleSpace",
)


class EV4PIDFusion(nn.Module):
    """PID control law as a two-stream feature fusion. Inputs: [fine, semantic].

    The fine (higher-resolution) feature is the plant output; the aligned
    semantic feature is the reference. Three gated correction terms map the
    control law u = Kp*e + Ki*integral(e) + Kd*de/dt:
      - Kp: direct semantic correction (proportional, instant response).
      - Ki: multi-scale low-pass accumulation of the semantic stream
        (integral, removes steady-state error -> missed small fruits).
      - Kd: local high-pass derivative of the fine stream (derivative,
        anticipates edge changes -> separates touching fruits).
    All gates are zero-initialised, so the module outputs `fine` at init and
    starts on the pretrained-compatible path.
    """

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"EV4PIDFusion expects [fine, semantic] channels, got {channels}")
        c_fine, c_sem = channels
        self.align = Conv(c_sem, c_fine, 1)
        self.low3 = nn.AvgPool2d(3, 1, 1)
        self.low5 = nn.AvgPool2d(5, 1, 2)
        self.integral_mix = nn.Conv2d(c_fine * 2, c_fine, 1, bias=False)
        self.derivative = nn.Conv2d(c_fine, c_fine, 3, 1, 1, groups=c_fine, bias=False)
        self.gate_p = nn.Parameter(torch.zeros(1, c_fine, 1, 1))
        self.gate_i = nn.Parameter(torch.zeros(1, c_fine, 1, 1))
        self.gate_d = nn.Parameter(torch.zeros(1, c_fine, 1, 1))

    def forward(self, features) -> torch.Tensor:
        fine, semantic = features
        reference = F.interpolate(self.align(semantic), size=fine.shape[-2:], mode="nearest")
        low = torch.cat((self.low3(reference), self.low5(reference)), dim=1)
        integral = self.integral_mix(low)
        derivative = self.derivative(fine - self.low3(fine))
        return fine + self.gate_p * reference + self.gate_i * integral + self.gate_d * derivative


class EV4ObserverGate(nn.Module):
    """Luenberger state observer as deep-to-shallow correction. Inputs: [state, observation].

    The shallow feature is the state estimate x_hat; the deep semantic feature
    is the observation y. The innovation (y - x_hat) is injected through a
    learned spatial observer gain L, i.e. x_hat+ = x_hat + lambda * L * (y - x_hat).
    The residual scale lambda is zero-initialised, so the module is identity at
    init; sigmoid keeps L in (0, 1) like a bounded observer gain.
    """

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"EV4ObserverGate expects [state, observation] channels, got {channels}")
        c_state, c_obs = channels
        self.observe = Conv(c_obs, c_state, 1)
        self.gain = nn.Sequential(nn.Conv2d(c_state * 2, c_state, 1, bias=False), nn.Sigmoid())
        self.residual = nn.Parameter(torch.zeros(1, c_state, 1, 1))

    def forward(self, features) -> torch.Tensor:
        state, observation = features
        observed = F.interpolate(self.observe(observation), size=state.shape[-2:], mode="nearest")
        innovation = observed - state
        gain = self.gain(torch.cat((state, observed), dim=1))
        return state + self.residual * gain * innovation


class EV4PhaseLead(nn.Module):
    """Phase-lead compensation Gc(s) = (1 + aTs) / (1 + Ts) on a single feature map.

    The compensator boosts the high-frequency component (derivative action,
    faster transient response -> sharper fruit boundaries) while the averaging
    pool acts as the lag pole that damps overshoot (boundary artifacts). The
    per-channel gain is zero-initialised, so the module is identity at init.
    Nearly parameter-free: one depthwise 3x3 convolution plus one gate vector.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.low = nn.AvgPool2d(3, 1, 1)
        self.compensate = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = self.compensate(x - self.low(x))
        return x + self.gain * lead


class EV4IntegralContext(nn.Module):
    """Integral action at the deepest stage: accumulated multi-scale global context.

    Pools the P5 feature at several grid sizes and re-injects the accumulation,
    the feature-space analogue of integral(e) dt that removes steady-state
    error. For this task the steady-state error is the systematic miss of
    tiny/occluded fruits that local features cannot evidence; scene-level
    context (fruit density, canopy layout) supplies the missing prior.
    Zero-initialised gate: identity at init.
    """

    def __init__(self, channels: int, sizes=(1, 2, 4)):
        super().__init__()
        self.pools = nn.ModuleList(nn.AdaptiveAvgPool2d(s) for s in sizes)
        self.mix = nn.Conv2d(channels * len(sizes), channels, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        # A linear 1x1 projection commutes with nearest interpolation. Project
        # tiny pooled grids first; retain the original parameter/state_dict keys.
        weights = self.mix.weight.split(x.shape[1], dim=1)
        context = sum(
            F.interpolate(F.conv2d(pool(x), weight), size=size, mode="nearest")
            for pool, weight in zip(self.pools, weights)
        )
        return x + self.gain * context


class EV4FilterFuse(nn.Module):
    """Anti-alias low-pass alignment with semantic-gated high-frequency injection.

    Inputs: [fine, semantic]. Signal-processing reading of FreqFusion applied
    at a single fusion point (wholesale frequency-neck replacement already
    failed in G03): the semantic stream is low-pass filtered BEFORE upsampling
    (anti-aliasing, a clean transfer function instead of nearest-neighbour
    spectral copies), and the fine stream's own high-pass component is boosted
    only where the aligned semantic low-pass votes for fruit. Both gates are
    zero-initialised: identity at init.
    """

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"EV4FilterFuse expects [fine, semantic] channels, got {channels}")
        c_fine, c_sem = channels
        self.align = Conv(c_sem, c_fine, 1)
        self.low = nn.AvgPool2d(3, 1, 1)
        self.high_gate = nn.Conv2d(c_fine, c_fine, 1, bias=True)
        self.gate_low = nn.Parameter(torch.zeros(1, c_fine, 1, 1))
        self.gate_high = nn.Parameter(torch.zeros(1, c_fine, 1, 1))

    def forward(self, features) -> torch.Tensor:
        fine, semantic = features
        aligned = self.align(semantic)
        semantic_low = F.interpolate(self.low(aligned), size=fine.shape[-2:], mode="bilinear", align_corners=False)
        high = fine - self.low(fine)
        inject = high * torch.sigmoid(self.high_gate(semantic_low))
        return fine + self.gate_low * semantic_low + self.gate_high * inject


class EV4CascadeRefine(nn.Module):
    """Cascade control as two-stage refinement. Inputs: [fine, semantic].

    The outer (slow) loop computes a semantic setpoint and applies a coarse
    correction toward it; the inner (fast) loop then applies a local depthwise
    correction on the remaining error. Splitting one big correction into a
    coarse outer loop plus a fine inner loop is what keeps a cascade stable,
    and here it prevents the semantic stream from washing out edge detail in a
    single aggressive step. Both gains are zero-initialised: identity at init.
    """

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"EV4CascadeRefine expects [fine, semantic] channels, got {channels}")
        c_fine, c_sem = channels
        self.setpoint = Conv(c_sem, c_fine, 1)
        self.inner = nn.Conv2d(c_fine, c_fine, 3, 1, 1, groups=c_fine, bias=False)
        self.outer_gain = nn.Parameter(torch.zeros(1, c_fine, 1, 1))
        self.inner_gain = nn.Parameter(torch.zeros(1, c_fine, 1, 1))

    def forward(self, features) -> torch.Tensor:
        fine, semantic = features
        reference = F.interpolate(self.setpoint(semantic), size=fine.shape[-2:], mode="nearest")
        outer = fine + self.outer_gain * (reference - fine)
        return outer + self.inner_gain * self.inner(reference - outer)


# ---------------------------------------------------------------------------
# Wave 3: paradigm-level restructuring for green-fruit camouflage and the
# extreme intra-image scale span. Task adaptations of camouflaged-object
# detection (SINet-V2 group-reversal attention, PraNet reverse attention,
# FEDER frequency decomposition, ZoomNet mixed-scale) and classical scale
# space / radial symmetry theory. Not complete reproductions of those works.
# ---------------------------------------------------------------------------


class EV4ChromaFront(nn.Module):
    """Learnable ISP-style colour frontend: 3x3 colour-correction matrix plus a
    per-channel piecewise-linear tone curve, both initialised to identity.

    Green-fruit-on-green-foliage is a colour-constancy / camouflage problem:
    the discriminative projection that separates fruit from leaf is NOT the
    identity in RGB space. Instead of hoping the first convolutions discover
    it, the frontend gets an explicit, tiny (3x3 + 3xP parameters) learnable
    colour pipeline - the same role the CCM and gamma stages play in a camera
    ISP. Identity at init, so the pretrained path is preserved.
    """

    def __init__(self, c1, c2=None, lut_points=8):
        super().__init__()
        if c1 != 3 or c2 not in (None, 3) or not isinstance(lut_points, int) or lut_points < 2:
            raise ValueError("EV4ChromaFront requires RGB input/output and at least two LUT points")
        self.lut_points = lut_points
        self.ccm = nn.Parameter(torch.zeros(3, 3))
        self.tone = nn.Parameter(torch.zeros(3, lut_points))
        knots = torch.linspace(0.0, 1.0, lut_points).view(1, 1, lut_points, 1, 1)
        self.register_buffer("knots", knots, persistent=False)

    def forward(self, x):
        mixed = x + torch.einsum("oc,bchw->bohw", self.ccm, x)
        # Only two knots contribute to a linear LUT. Avoid the B*3*P*H*W
        # triangular-basis tensor (600 MiB at batch=16, 640px, P=8, FP32).
        position = mixed.clamp(0.0, 1.0) * (self.lut_points - 1)
        left = position.floor().long().clamp_max(self.lut_points - 2)
        fraction = position - left
        channel = torch.arange(3, device=x.device).view(1, 3, 1, 1)
        low, high = self.tone[channel, left], self.tone[channel, left + 1]
        delta = torch.lerp(low, high, fraction)
        return mixed + delta


class EV4LapFront(nn.Module):
    """Laplacian-pyramid input frontend: explicit band-pass detail injected at
    the stem input (3-channel image in, 3-channel image out).

    Camouflage similarity between unripe fruit and foliage lives in the
    low-frequency chroma band (FEDER, CVPR 2023: frequency decomposition
    separates foreground from background); fruit identity - oil-gland texture,
    rim curvature - lives in the band-pass structure. The stem therefore
    receives the identity image plus a gated mixture of two Laplacian bands.
    Identity at init via the zero-initialised gate.
    """

    def __init__(self, c1, c2=None):
        super().__init__()
        self.low1 = nn.AvgPool2d(3, 1, 1)
        self.low2 = nn.AvgPool2d(7, 1, 3)
        self.mix = nn.Conv2d(c1 * 2, c1, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, c1, 1, 1))

    def forward(self, x):
        low = self.low1(x)
        band_hi = x - low
        band_mid = low - self.low2(x)
        return x + self.gate * self.mix(torch.cat((band_hi, band_mid), dim=1))


class EV4ReverseRefine(nn.Module):
    """Reverse attention refinement. Inputs: [fine (P3), semantic (P5)].

    Task adaptation of SINet-V2 group-reversal attention (TPAMI 2022) and
    PraNet parallel reverse attention (MICCAI 2020): deep semantics vote a
    coarse objectness map; the confident response is ERASED from the detail
    stream and the residual stream is refined. Camouflaged green fruits are
    precisely the low-confidence residual that the confident path misses -
    this is the recall-oriented half of the search-identify paradigm.
    Zero-initialised gate: identity at init.
    """

    def __init__(self, channels):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"EV4ReverseRefine expects [fine, semantic] channels, got {channels}")
        c_fine, c_sem = channels
        # A logit must be signed/unbounded. SiLU before sigmoid limits its
        # negative range, preventing this learned gate from approaching zero.
        # This is NOT a calibrated objectness map (no direct objectness target).
        self.to_prob = Conv(c_sem, 1, 1, act=False)
        self.refine = nn.Sequential(
            nn.Conv2d(c_fine, c_fine, 3, 1, 1, groups=c_fine, bias=False),
            nn.BatchNorm2d(c_fine),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_fine, c_fine, 1, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(1, c_fine, 1, 1))

    def forward(self, features):
        fine, semantic = features
        prob = torch.sigmoid(
            F.interpolate(self.to_prob(semantic), size=fine.shape[-2:], mode="bilinear", align_corners=False)
        )
        residual = self.refine(fine * (1.0 - prob))
        return fine + self.gate * residual


class EV4RadialVote(nn.Module):
    """Blob-versus-strip shape voting on a single feature map.

    Citrus fruits are radially symmetric blobs - strong extent in EVERY
    orientation; leaves and branches are anisotropic strips - one dominant
    orientation. A differentiable analogue of fast radial symmetry (Loy and
    Zelinsky, TPAMI 2003) built from strip pooling: the orientation-wise
    MINIMUM responds only where all orientations agree (fruit-like), while
    the orientation contrast exposes strip-like structures. Both are mixed
    and injected as a gated shape bias. Zero-initialised gate: identity at
    init. Nearly parameter-free.
    """

    def __init__(self, channels, k=9):
        super().__init__()
        self.pool_h = nn.AvgPool2d((1, k), 1, (0, k // 2))
        self.pool_v = nn.AvgPool2d((k, 1), 1, (k // 2, 0))
        self.vote = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        extent_h, extent_v = self.pool_h(x), self.pool_v(x)
        isotropy = torch.minimum(extent_h, extent_v)
        anisotropy = (extent_h - extent_v).abs()
        return x + self.gate * self.vote(torch.cat((isotropy, anisotropy), dim=1))


class EV4ScaleSpace(nn.Module):
    """Multi-window box-filter context at the detail level (not Gaussian scale space).

    Fixed-octave FPN gives each position exactly one analysis scale, but a
    single orchard image spans extreme object scales. Classical scale-space
    theory (Lindeberg; SIFT, IJCV 2004) instead analyses a dense sigma
    ladder; ZoomNet (CVPR 2022) shows mixed scales are what surfaces
    camouflaged objects - but the E9 experiments here already showed that
    INPUT-level scaling raises tiny recall without raising overall AP, so
    the ladder is built at the FEATURE level, densely and cheaply, and
    re-mixed into the detail map. Zero-initialised gate: identity at init.
    """

    def __init__(self, channels):
        super().__init__()
        self.blur1 = nn.AvgPool2d(3, 1, 1)
        self.blur2 = nn.AvgPool2d(5, 1, 2)
        self.blur3 = nn.AvgPool2d(9, 1, 4)
        self.mix = nn.Conv2d(channels * 4, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        ladder = torch.cat((x, self.blur1(x), self.blur2(x), self.blur3(x)), dim=1)
        return x + self.gate * self.mix(ladder)
