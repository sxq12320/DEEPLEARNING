# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Frequency-preserving and state-space blocks for tiny camouflaged citrus.

The blocks in this file deliberately implement one coherent backbone idea instead of another collection of
independent attention plug-ins:

* :class:`FrequencyAwareDown` performs anti-aliased Haar-band downsampling and learns where high-frequency detail
  should survive. Unlike the earlier AAFM experiment, its gate is based on local contrast and spectral energy rather
  than the unsafe assumption that every tiny citrus is dark.
* :class:`CitrusSAVSS` ports the verified GBC, PAF and four-direction SASS design of SCSegamba into a rectangular,
  channel-agnostic YOLO block. The state-space branch operates on a pooled P4 map while the full-resolution local
  branch is retained, which gives global orchard context without applying a long scan to every P4 pixel.

The state recurrence has a dependency-free, chunk-parallel PyTorch implementation. It uses an associative affine
prefix scan inside short chunks, reducing a length-L Python loop to roughly ``L/chunk_size`` iterations while retaining
the exact SCSegamba recurrence and CUDA autograd. ``mamba_ssm`` remains an optional acceleration path, not a dependency.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

try:  # The training server may provide the official fused kernel; Windows development usually does not.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan_fn
except (ImportError, OSError):  # pragma: no cover - availability is environment-specific
    _mamba_selective_scan_fn = None

__all__ = ("FrequencyAwareDown", "CitrusSAVSS")


def _haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orthonormal Haar LL/LH/HL/HH bands with replicate padding for odd feature sizes."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
    x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (
        (x00 + x01 + x10 + x11) * 0.5,
        (-x00 - x01 + x10 + x11) * 0.5,
        (-x00 + x01 - x10 + x11) * 0.5,
        (x00 - x01 - x10 + x11) * 0.5,
    )


class FrequencyAwareDown(nn.Module):
    """Anti-aliased downsampling with a learned, contrast-aware high-frequency bypass.

    Haar decomposition first separates the band-limited LL path from three directional detail bands. The bypass gate
    sees high-frequency energy and local LL contrast, so it can preserve small circular boundaries while suppressing
    broad smooth foliage. The gate intentionally contains no fixed brightness or colour heuristic: green fruit and
    shadowed leaves cannot be separated reliably by a hard-coded darkness rule.
    """

    def __init__(self, c1: int, c2: int, layer_scale: float = 0.1):
        super().__init__()
        self.low = Conv(c1, c2, 1, 1)
        self.high = Conv(c1 * 3, c2, 1, 1)
        self.gate = nn.Conv2d(2, 1, 3, 1, 1, bias=True)
        self.gamma = nn.Parameter(torch.full((1, c2, 1, 1), float(layer_scale)))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)  # sigmoid(0)=0.5: stable but the high-frequency path is not disabled.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by two while preserving gated directional detail."""
        ll, lh, hl, hh = _haar_dwt(x)
        low = self.low(ll)
        high_bands = torch.cat((lh, hl, hh), 1)
        high = self.high(high_bands)
        spectral_energy = high_bands.square().mean(1, keepdim=True).add(1e-6).sqrt()
        local_contrast = (ll - F.avg_pool2d(ll, 5, 1, 2)).abs().mean(1, keepdim=True)
        gate = self.gate(torch.cat((spectral_energy, local_contrast), 1)).sigmoid()
        return low + self.gamma * gate * high


class _BottConv(nn.Module):
    """Bottleneck depthwise convolution copied from the official SCSegamba GBC implementation."""

    def __init__(self, c1: int, c2: int, cm: int, k: int, p: int = 0):
        super().__init__()
        self.pw1 = nn.Conv2d(c1, cm, 1, bias=True)
        self.dw = nn.Conv2d(cm, cm, k, 1, p, groups=cm, bias=False)
        self.pw2 = nn.Conv2d(cm, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw2(self.dw(self.pw1(x)))


def _group_count(channels: int, preferred: int) -> int:
    """Return the largest valid GroupNorm group count no greater than ``preferred``."""
    groups = min(channels, max(1, preferred))
    while channels % groups:
        groups -= 1
    return groups


class _GBC(nn.Module):
    """Channel-safe port of SCSegamba's official gated bottleneck convolution."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(8, channels // 8)

        def block(k: int, groups: int) -> nn.Sequential:
            return nn.Sequential(
                _BottConv(channels, channels, mid, k, k // 2),
                nn.GroupNorm(_group_count(channels, groups), channels),
                nn.ReLU(inplace=True),
            )

        self.block1 = block(3, max(1, channels // 16))
        self.block2 = block(3, max(1, channels // 16))
        self.block3 = block(1, max(1, channels // 16))
        self.block4 = block(1, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block4(self.block2(self.block1(x)) * self.block3(x))


class _PAF(nn.Module):
    """Progressive asymmetric fusion ported from the official SCSegamba PAF implementation."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(16, channels // 2)
        self.feature_transform = nn.Sequential(_BottConv(channels, mid, 16, 1), nn.BatchNorm2d(mid))
        self.channel_adapter = nn.Sequential(_BottConv(mid, channels, 16, 1), nn.BatchNorm2d(channels))

    def forward(self, local: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        global_context = F.interpolate(global_context, size=local.shape[-2:], mode="bilinear", align_corners=False)
        local_key = self.feature_transform(local)
        global_query = self.feature_transform(global_context)
        similarity = torch.sigmoid(self.channel_adapter(local_key * global_query))
        return (1.0 - similarity) * local + similarity * global_context


def _directions(coords: list[tuple[int, int]]) -> tuple[int, ...]:
    """Encode token-to-token movement as 0=start, 1=right, 2=left, 3=down, 4=up."""
    out = [0]
    for (y0, x0), (y1, x1) in zip(coords, coords[1:]):
        dy, dx = y1 - y0, x1 - x0
        if abs(dx) >= abs(dy) and dx:
            out.append(1 if dx > 0 else 2)
        elif dy:
            out.append(3 if dy > 0 else 4)
        else:
            out.append(0)
    return tuple(out)


@lru_cache(maxsize=32)
def _sass_orders(height: int, width: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Build four rectangular SASS paths and exact inverse permutations.

    The official repository assumes square maps in its wrapper and contains a typo in one diagonal inverse index. This
    implementation constructs inverses from each path itself, making non-square YOLO feature maps safe.
    """
    row = [(y, x) for y in range(height) for x in (range(width) if y % 2 == 0 else range(width - 1, -1, -1))]
    col = [(y, x) for x in range(width) for y in (range(height) if x % 2 == 0 else range(height - 1, -1, -1))]
    diagonal: list[tuple[int, int]] = []
    anti_diagonal: list[tuple[int, int]] = []
    for diagonal_id in range(height + width - 1):
        coords = [(y, diagonal_id - y) for y in range(height) if 0 <= diagonal_id - y < width]
        if diagonal_id % 2:
            coords.reverse()
        diagonal.extend(coords)
        anti_diagonal.extend((y, width - 1 - x) for y, x in coords)

    orders, inverses, directions = [], [], []
    for coords in (row, col, diagonal, anti_diagonal):
        order = tuple(y * width + x for y, x in coords)
        inverse = [0] * (height * width)
        for scan_index, raster_index in enumerate(order):
            inverse[raster_index] = scan_index
        orders.append(order)
        inverses.append(tuple(inverse))
        directions.append(_directions(coords))
    return tuple(orders), tuple(inverses), tuple(directions)


def _selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    delta_bias: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for Mamba selective scan, used when the fused extension is unavailable."""
    input_dtype = u.dtype
    work_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    u, delta, a, b, c, d, delta_bias = (
        tensor.to(work_dtype) for tensor in (u, delta, a, b, c, d, delta_bias)
    )
    delta = F.softplus(delta + delta_bias.view(1, -1, 1))
    batch, channels, length = u.shape
    state = u.new_zeros(batch, channels, a.shape[1])
    outputs = []
    for index in range(length):
        dt = delta[:, :, index]
        decay = torch.exp(dt.unsqueeze(-1) * a.unsqueeze(0))
        drive = dt.unsqueeze(-1) * b[:, :, index].unsqueeze(1) * u[:, :, index].unsqueeze(-1)
        state = decay * state + drive
        y = (state * c[:, :, index].unsqueeze(1)).sum(-1) + d.view(1, -1) * u[:, :, index]
        outputs.append(y)
    return torch.stack(outputs, -1).to(input_dtype)


def _selective_scan_chunked(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    delta_bias: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Dependency-free selective scan using an associative affine prefix operator.

    Each recurrence step is an affine map ``state -> decay * state + drive``. Affine-map composition is associative,
    so all states inside a short chunk can be evaluated in log2(chunk_size) parallel prefix stages. Only chunk
    boundaries remain sequential. This avoids the hundreds of Python iterations in the reference implementation while
    preserving its equations, gradients and four-direction SASS paths.
    """
    input_dtype = u.dtype
    work_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    u, delta, a, b, c, d, delta_bias = (
        tensor.to(work_dtype) for tensor in (u, delta, a, b, c, d, delta_bias)
    )
    delta = F.softplus(delta + delta_bias.view(1, -1, 1))
    batch, channels, length = u.shape
    state = u.new_zeros(batch, channels, a.shape[1])
    outputs = []
    chunk_size = max(1, int(chunk_size))

    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        u_chunk = u[..., start:end]
        delta_chunk = delta[..., start:end]
        b_chunk = b[..., start:end]
        c_chunk = c[..., start:end]

        # Inclusive affine prefix scan. ``prefix_decay`` and ``prefix_drive`` represent the composed transform from
        # the beginning of this chunk to every position in it.
        prefix_decay = torch.exp(delta_chunk.unsqueeze(2) * a[None, :, :, None])
        prefix_drive = (
            delta_chunk.unsqueeze(2) * u_chunk.unsqueeze(2) * b_chunk.unsqueeze(1)
        )
        offset = 1
        while offset < end - start:
            left_decay = F.pad(prefix_decay[..., :-offset], (offset, 0), value=1.0)
            left_drive = F.pad(prefix_drive[..., :-offset], (offset, 0), value=0.0)
            prefix_drive = prefix_drive + prefix_decay * left_drive
            prefix_decay = prefix_decay * left_decay
            offset *= 2

        states = prefix_decay * state.unsqueeze(-1) + prefix_drive
        state = states[..., -1]
        y = (states * c_chunk.unsqueeze(1)).sum(2) + d.view(1, -1, 1) * u_chunk
        outputs.append(y)

    return torch.cat(outputs, -1).to(input_dtype)


class _SASS2D(nn.Module):
    """Four-direction saliency-aware state-space scan adapted from official SCSegamba."""

    def __init__(self, channels: int, d_state: int = 8, expand: float = 1.0):
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        self.inner = max(16, int(channels * expand))
        self.dt_rank = max(1, math.ceil(channels / 16))
        self.in_proj = nn.Conv2d(channels, self.inner * 2, 1)
        self.local_conv = _BottConv(self.inner, self.inner, max(8, self.inner // 16), 3, 1)
        self.x_proj = nn.Linear(self.inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner, bias=True)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.inner, 1))
        self.d = nn.Parameter(torch.ones(self.inner))
        self.direction_b = nn.Parameter(torch.zeros(5, d_state))
        self.out_proj = nn.Conv2d(self.inner, channels, 1)
        self.act = nn.SiLU()
        self._init_dt()

    def _init_dt(self) -> None:
        dt_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt = torch.exp(torch.rand(self.inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(1e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)
        nn.init.trunc_normal_(self.direction_b, std=0.02)

    def _scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        a = -torch.exp(self.a_log.float())
        if _mamba_selective_scan_fn is not None and u.is_cuda:
            return _mamba_selective_scan_fn(
                u.contiguous(),
                delta.contiguous(),
                a,
                b.contiguous(),
                c.contiguous(),
                self.d.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
        return _selective_scan_chunked(u, delta, a, b, c, self.d, self.dt_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_part, z = self.in_proj(x).chunk(2, 1)
        x_part = self.act(self.local_conv(x_part))
        tokens = x_part.flatten(2).transpose(1, 2)
        parameters = self.x_proj(tokens)
        dt_low_rank, b_tokens, c_tokens = torch.split(parameters, (self.dt_rank, self.d_state, self.d_state), -1)
        delta_tokens = self.dt_proj(dt_low_rank)
        orders, inverses, directions = _sass_orders(height, width)
        scans = []
        for order_tuple, inverse_tuple, direction_tuple in zip(orders, inverses, directions):
            order = torch.as_tensor(order_tuple, device=x.device, dtype=torch.long)
            inverse = torch.as_tensor(inverse_tuple, device=x.device, dtype=torch.long)
            direction = torch.as_tensor(direction_tuple, device=x.device, dtype=torch.long)
            u = tokens.index_select(1, order).transpose(1, 2)
            delta = delta_tokens.index_select(1, order).transpose(1, 2)
            b_scan = b_tokens.index_select(1, order).transpose(1, 2)
            c_scan = c_tokens.index_select(1, order).transpose(1, 2)
            b_scan = b_scan + self.direction_b.index_select(0, direction).transpose(0, 1).unsqueeze(0)
            y = self._scan(u, delta, b_scan, c_scan).transpose(1, 2).index_select(1, inverse)
            scans.append(y)
        y = torch.stack(scans, 0).mean(0).transpose(1, 2).reshape(batch, self.inner, height, width)
        # Ultralytics propagates ``model.inplace`` into child activations; ``z`` is a chunk view and therefore must
        # use an explicitly out-of-place activation for autograd safety.
        return self.out_proj(y * F.silu(z, inplace=False))


class CitrusSAVSS(nn.Module):
    """Pooled global SASS plus full-resolution GBC/PAF for tiny green citrus in foliage.

    P4 is the intended placement: P3 is expensive for four scans, while P5 has already discarded too much tiny-fruit
    structure. The local GBC path stays at P4 resolution; only the context path is pooled before SASS and then gated by
    the local path through official-style PAF. LayerScale makes the new block start close to an identity mapping.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        d_state: int = 8,
        expand: float = 1.0,
        context_pool: int = 2,
        layer_scale: float = 0.01,
    ):
        super().__init__()
        self.input_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.local = _GBC(c2)
        self.norm = nn.GroupNorm(1, c2)
        self.context_pool = max(1, int(context_pool))
        self.sass = _SASS2D(c2, d_state=d_state, expand=expand)
        self.fuse = _PAF(c2)
        self.post = _GBC(c2)
        self.gamma = nn.Parameter(torch.full((1, c2, 1, 1), float(layer_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input_proj(x)
        local = self.local(base)
        context_input = self.norm(local)
        if self.context_pool > 1 and min(context_input.shape[-2:]) >= self.context_pool:
            context_input = F.avg_pool2d(context_input, self.context_pool, self.context_pool)
        global_context = self.sass(context_input)
        fused = self.fuse(local, global_context)
        return base + self.gamma * self.post(fused)
