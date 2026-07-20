"""Region-gated low-light enhancement front-end for YOLO (HVI color space).

Adapts the HVI color space from the CVPR-2025 paper "HVI: A New Color Space for
Low-light Image Enhancement" (HVI-CIDNet, https://github.com/Fediory/HVI-CIDNet)
into a lightweight, plug-and-play Ultralytics module that runs as **layer 0** on
the raw 3-channel input image and outputs an enhanced 3-channel image at the same
resolution, before the normal backbone.

Why this design (faithful where it matters, pragmatic elsewhere)
----------------------------------------------------------------
* We keep the paper's key contribution -- the **HVI transform** (polarized
  Hue/Saturation + a learnable intensity-collapse ``density_k``) -- because that
  is what decouples colour from brightness and removes the red/black artefacts
  that plague sRGB/HSV enhancement of dark regions. See ``RGBHVI``.
* We DROP the heavy dual-branch CIDNet (cross-attention LCA, 1.88M / 7.57 GFLOPs)
  and replace it with a tiny down-sample -> gated-conv -> up-sample residual
  enhancer (``IELBlock``), so the front-end stays cheap (~1 GFLOP at 640).
* We DO NOT need paired low/normal-light ground truth: the whole module is
  differentiable and is trained end-to-end by the downstream detection/seg loss.
* **Region-aware gate.** A tiny head predicts a per-pixel gate ``g in [0,1]``;
  the HVI residual is applied as ``hvi + g * delta``. Trained only by the task
  loss, the gate learns to fire where enhancement helps detection (distant, dark,
  low-contrast citrus). A darkness prior ``g *= (1 - I)`` biases enhancement to
  low-light regions and leaves already-bright regions untouched. This realises
  "perceive the object region, enhance only there" without any extra labels.

Determinism
-----------
The reference ``RGB_HVI`` fills Hue via boolean-mask scatter (``hue[mask]=...``)
and up-samples with bilinear interpolation -- both have **non-deterministic**
CUDA backward kernels and raise under ``torch.use_deterministic_algorithms(True)``
(the citrus protocol trains with ``deterministic=True``). We re-derive the exact
same maths with ``torch.where`` / ``amax`` / ``amin`` and use nearest-neighbour
up-sampling, so the module is bit-reproducible. ``test_hvi_enhance.py`` checks the
port numerically against the reference formulae.

Registration (4 files, see 模块使用说明.md)
------------------------------------------
1. this file, 2. ``modules/__init__.py`` (export), 3. ``tasks.py`` import,
4. ``tasks.py::parse_model`` -- a **dedicated** ``elif m is HVIEnhance`` branch
   that keeps ``c1 == c2 == 3`` (NOT ``base_modules``, whose width-scaling would
   turn ``c2=3`` into 8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("RGBHVI", "IELBlock", "HVIEnhance")

_PI = 3.141592653589793


class RGBHVI(nn.Module):
    """Differentiable, deterministic sRGB <-> HVI color-space transform.

    ``HVIT`` maps an RGB image in ``[0, 1]`` (channel order R, G, B) to a
    3-channel ``[H, V, I]`` map; ``PHVIT`` maps it back to RGB. ``density_k`` is a
    single learnable scalar (the paper's intensity-collapse density, init 0.2)
    shared by both directions. Everything is expressed with ``torch.where`` and
    ``amax``/``amin`` so it is fully differentiable AND deterministic on CUDA.

    Numerically equivalent to ``net/HVI_transform.py::RGB_HVI`` in the reference
    repo (verified in ``test_hvi_enhance.py``).
    """

    def __init__(self, k_init: float = 0.2):
        super().__init__()
        self.density_k = nn.Parameter(torch.full((1,), float(k_init)))
        self.eps = 1e-8

    def HVIT(self, img: torch.Tensor) -> torch.Tensor:
        """RGB ``[B, 3, H, W]`` in ``[0, 1]`` -> HVI ``[B, 3, H, W]``."""
        eps = self.eps
        r, g, b = img[:, 0], img[:, 1], img[:, 2]  # each [B, H, W]
        value = img.amax(1)  # V = Imax  (deterministic backward)
        img_min = img.amin(1)
        delta = value - img_min
        dc = delta + eps  # guarded denominator

        # Branchless Hue (in [0, 6)); assignment priority mirrors the reference:
        # base = B-branch, overridden by G-branch, then R-branch, then achromatic.
        hue_r = ((g - b) / dc) % 6.0
        hue_g = 2.0 + (b - r) / dc
        hue_b = 4.0 + (r - g) / dc
        hue = hue_b
        hue = torch.where(value == g, hue_g, hue)
        hue = torch.where(value == r, hue_r, hue)
        hue = torch.where(value == img_min, torch.zeros_like(hue), hue)  # achromatic
        hue = hue / 6.0

        saturation = delta / (value + eps)
        saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation)

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        color_sensitive = ((value * 0.5 * _PI).sin() + eps).pow(k)  # Ck
        ch = (2.0 * _PI * hue).cos()
        cv = (2.0 * _PI * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        return torch.cat([H, V, value], dim=1)

    def PHVIT(self, img: torch.Tensor) -> torch.Tensor:
        """HVI ``[B, 3, H, W]`` -> RGB ``[B, 3, H, W]`` in ``[0, 1]``."""
        eps = self.eps
        H, V, I = img[:, 0], img[:, 1], img[:, 2]
        H = H.clamp(-1, 1)
        V = V.clamp(-1, 1)
        v = I.clamp(0, 1)

        # k is a fixed density for the inverse (detached, like the reference's
        # cached ``this_k``) -- keeps the inverse a stable, well-defined mapping.
        k = self.density_k.detach()
        color_sensitive = ((v * 0.5 * _PI).sin() + eps).pow(k)
        H = (H / (color_sensitive + eps)).clamp(-1, 1)
        V = (V / (color_sensitive + eps)).clamp(-1, 1)

        h = torch.atan2(V + eps, H + eps) / (2.0 * _PI)
        h = h % 1.0
        s = torch.sqrt(H * H + V * V + eps).clamp(0, 1)

        # Branchless HSV -> RGB (sextant selection via torch.where on hi in 0..5).
        hi = torch.floor(h * 6.0) % 6.0
        f = h * 6.0 - torch.floor(h * 6.0)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r = torch.where(hi == 0, v, torch.where(hi == 1, q, torch.where(hi == 2, p,
            torch.where(hi == 3, p, torch.where(hi == 4, t, v)))))
        g = torch.where(hi == 0, t, torch.where(hi == 1, v, torch.where(hi == 2, v,
            torch.where(hi == 3, q, torch.where(hi == 4, p, p)))))
        b = torch.where(hi == 0, p, torch.where(hi == 1, p, torch.where(hi == 2, t,
            torch.where(hi == 3, v, torch.where(hi == 4, v, q)))))

        rgb = torch.stack([r, g, b], dim=1)
        return rgb.clamp(0, 1)


class IELBlock(nn.Module):
    """Lightweight gated depth-wise FFN (the paper's IEL/CDL core, no attention).

    ``project_in`` -> depth-wise 3x3 -> split -> tanh-gated depth-wise mix ->
    ``project_out``, with a residual. Pure convolutions (no einops), so it is
    cheap and deterministic. ``expansion`` controls the hidden width (the paper
    uses 2.66; we default to 1.0 to stay light at full/near-full resolution).
    """

    def __init__(self, dim: int, expansion: float = 1.0, bias: bool = True):
        super().__init__()
        hidden = max(int(dim * expansion), 1)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated depth-wise mixing with a residual. Shape-preserving."""
        identity = x
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.act(self.dwconv1(x1)) + x1
        x2 = self.act(self.dwconv2(x2)) + x2
        x = self.project_out(x1 * x2)
        return x + identity


class HVIEnhance(nn.Module):
    """Region-gated low-light enhancement front-end (3-ch image in, 3-ch out).

    Pipeline: ``HVIT`` -> stem -> (down-sample) -> ``blocks`` x ``IELBlock`` ->
    (up-sample, nearest) -> residual head ``delta`` -> gated blend
    ``hvi + g * delta`` -> ``PHVIT`` -> enhanced RGB image. The residual head is
    zero-initialised so the module starts as (near-)identity, which keeps
    transfer-learning of the downstream backbone intact from step 0.

    Args:
        c1 (int): Input channels -- must be 3 (raw RGB image).
        c2 (int): Output channels -- must be 3 (enhanced RGB image).
        base (int): Enhancer width. Default 16.
        blocks (int): Number of ``IELBlock``. Default 2.
        down (int): Internal down-sampling factor for the enhancer (compute
            saver). Default 2 (process at half resolution). 1 = full resolution.
        expansion (float): Hidden-width factor inside ``IELBlock``. Default 1.0.
        gate (bool): Enable the learned region gate. Default True.
        dark_prior (bool): Bias the gate toward dark regions (``g *= 1 - I``).
            Default True.

    Example:
        >>> m = HVIEnhance(3, 3)
        >>> x = torch.rand(2, 3, 640, 640)          # RGB in [0, 1]
        >>> y = m(x)
        >>> tuple(y.shape)
        (2, 3, 640, 640)
    """

    def __init__(self, c1: int = 3, c2: int = 3, base: int = 16, blocks: int = 2,
                 down: int = 2, expansion: float = 1.0, gate: bool = True, dark_prior: bool = True):
        super().__init__()
        assert c1 == 3 and c2 == 3, "HVIEnhance is an image front-end: c1 == c2 == 3 required."
        self.down = int(down)
        self.use_gate = bool(gate)
        self.dark_prior = bool(dark_prior)

        self.trans = RGBHVI()
        self.stem = nn.Sequential(nn.Conv2d(3, base, 3, padding=1), nn.SiLU(inplace=True))
        self.blocks = nn.Sequential(*(IELBlock(base, expansion) for _ in range(blocks)))
        self.res_head = nn.Conv2d(base, 3, 3, padding=1)
        if self.use_gate:
            self.gate_head = nn.Conv2d(base, 1, 3, padding=1)

        # Start as identity: zero the residual head so delta = 0 -> out = PHVIT(HVIT(x)) ~ x.
        nn.init.zeros_(self.res_head.weight)
        nn.init.zeros_(self.res_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RGB ``[B, 3, H, W]`` in ``[0, 1]`` -> enhanced RGB ``[B, 3, H, W]``."""
        hvi = self.trans.HVIT(x)
        intensity = hvi[:, 2:3]  # I channel, [B, 1, H, W], in [0, 1]

        feat = self.stem(hvi)
        if self.down > 1:
            feat = F.avg_pool2d(feat, self.down, self.down)
        feat = self.blocks(feat)
        if self.down > 1:
            feat = F.interpolate(feat, size=hvi.shape[-2:], mode="nearest")

        delta = self.res_head(feat)  # residual in HVI space, [B, 3, H, W]
        if self.use_gate:
            g = torch.sigmoid(self.gate_head(feat))  # learned region gate [B, 1, H, W]
            if self.dark_prior:
                g = g * (1.0 - intensity).clamp(0, 1)  # focus on low-light regions
            out_hvi = hvi + g * delta
        else:
            out_hvi = hvi + delta

        return self.trans.PHVIT(out_hvi)
