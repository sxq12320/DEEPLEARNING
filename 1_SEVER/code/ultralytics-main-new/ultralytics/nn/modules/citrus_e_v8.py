# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V8: selective context backbone and native-evidence P4 reconciliation.

Task adaptations of partial spatial mixing (FasterNet), separable contextual
mixing (LSKA) and discrepancy-gated fusion (PIDNet/semantic correction).
Not a reproduction, a PID controller, a stability proof or proven accuracy gain.
Early high-resolution stages and pretrained P3 refiners are deliberately kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("EV8ContextStage", "EV8P4Reconcile")


class EV8ContextMixer(nn.Module):
    """Quarter local mixing, quarter orthogonal context, half direct channel path."""

    def __init__(self, channels):
        super().__init__()
        self.part = channels // 4
        if self.part < 4 or channels % 4:
            raise ValueError("Context mixer requires channels divisible by four and >=16")
        p = self.part
        self.local = Conv(p, p, 3)
        self.context = nn.Sequential(Conv(p, p, (1, 7), g=p), Conv(p, p, (7, 1), g=p))
        self.channel = nn.Sequential(Conv(channels, channels * 2, 1), Conv(channels * 2, channels, 1, act=False))
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        local, context, direct = torch.split(x, [self.part, self.part, x.shape[1] - 2 * self.part], dim=1)
        mixed = torch.cat((self.local(local), self.context(context), direct), dim=1)
        return x + self.gain.tanh() * self.channel(mixed)


class EV8ContextStage(nn.Module):
    """Replace a deep CSP stage, not the shallow stem; parser supplies repeats."""

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.project = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(EV8ContextMixer(c2) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.project(x))


class EV8P4Reconcile(nn.Module):
    """Replace P3-down/concat/CSP P4 reconstruction with native discrepancy correction.

    Inputs: [top-down P4, native C4, native C3]. P3 still uses the original
    top-down spatial refiner. P4 no longer depends on an already compressed P3.
    Native C3 is pooled BEFORE projection. Width stays equal to inherited P4,
    preserving prediction tower shapes, but not claiming identical semantics.
    Compared with historical EV3NativeFusion this removes a path rather than
    adding a third concat input. Gates act on evidence differences, not RGB.
    """

    def __init__(self, channels, c2):
        super().__init__()
        if len(channels) != 3 or channels[0] != c2:
            raise ValueError("Use [P4td, C4, C3] with P4td width equal to output")
        hidden = max(16, c2 // 4)
        self.anchor = Conv(channels[0], hidden, 1, act=False)
        self.native = Conv(channels[1], hidden, 1, act=False)
        self.fine = Conv(channels[2], hidden, 1, act=False)
        self.gate = nn.Conv2d(3 * hidden, 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.update = nn.Sequential(Conv(hidden, hidden, 3), Conv(hidden, c2, 1, act=False))
        self.gain = nn.Parameter(torch.full((1, c2, 1, 1), 0.1))

    def forward(self, inputs):
        prior, native, fine = inputs
        shape = prior.shape[-2:]
        if native.shape[-2:] != shape:
            raise ValueError("Native C4 and top-down P4 must share their spatial grid")
        estimate = self.anchor(prior)
        native_error = self.native(native) - estimate
        fine_error = self.fine(F.adaptive_avg_pool2d(fine, shape)) - estimate
        gate = self.gate(torch.cat((estimate, native_error.abs(), fine_error.abs()), 1)).softmax(dim=1)
        correction = gate[:, :1] * native_error + gate[:, 1:] * fine_error
        return prior + self.gain.tanh() * self.update(correction)
