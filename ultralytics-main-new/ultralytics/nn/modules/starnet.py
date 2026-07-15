"""Official StarNet backbone modules for Ultralytics models.

This implementation follows Rewrite the Stars (CVPR 2024) and keeps the block
layout compatible with the released ImageNet checkpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, trunc_normal_  # timm < 0.9
    except ImportError:
        from torch.nn.init import trunc_normal_

        class DropPath(nn.Module):
            """Drop paths per sample, matching timm behavior when timm is unavailable."""

            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor


STARNET_VARIANTS: dict[str, dict[str, Any]] = {
    "s050": {"base_dim": 16, "depths": [1, 1, 3, 1], "mlp_ratio": 3, "url": None},
    "s100": {"base_dim": 20, "depths": [1, 2, 4, 1], "mlp_ratio": 4, "url": None},
    "s150": {"base_dim": 24, "depths": [1, 2, 4, 2], "mlp_ratio": 3, "url": None},
    "s1": {
        "base_dim": 24,
        "depths": [2, 2, 8, 3],
        "mlp_ratio": 4,
        "url": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    },
    "s2": {
        "base_dim": 32,
        "depths": [1, 2, 6, 2],
        "mlp_ratio": 4,
        "url": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    },
    "s3": {
        "base_dim": 32,
        "depths": [2, 2, 8, 4],
        "mlp_ratio": 4,
        "url": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    },
    "s4": {
        "base_dim": 32,
        "depths": [3, 3, 12, 5],
        "mlp_ratio": 4,
        "url": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
    },
}


class StarConvBN(nn.Sequential):
    """Convolution followed by optional BatchNorm, matching the official StarNet code."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups),
        )
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class StarNetBlock(nn.Module):
    """Official StarNet block with depthwise context and element-wise star operation."""

    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        hidden_dim = int(mlp_ratio * dim)
        self.dwconv = StarConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = StarConvBN(dim, hidden_dim, 1, with_bn=False)
        self.f2 = StarConvBN(dim, hidden_dim, 1, with_bn=False)
        self.g = StarConvBN(hidden_dim, dim, 1, with_bn=True)
        self.dwconv2 = StarConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the StarNet block."""
        shortcut = x
        x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        x = self.dwconv2(self.g(x))
        return shortcut + self.drop_path(x)


class StarNetBackbone(nn.Module):
    """StarNet feature extractor returning P2, P3, P4, and P5 feature maps."""

    def __init__(
        self,
        variant: str = "s1",
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        weights: str | None = None,
    ):
        super().__init__()
        variant = str(variant).lower().replace("starnet_", "")
        if variant not in STARNET_VARIANTS:
            raise ValueError(f"Unknown StarNet variant '{variant}'. Choices: {sorted(STARNET_VARIANTS)}")

        cfg = STARNET_VARIANTS[variant]
        base_dim = int(cfg["base_dim"])
        depths = list(cfg["depths"])
        mlp_ratio = int(cfg["mlp_ratio"])
        self.variant = variant
        self.out_indices = tuple(int(i) for i in out_indices)
        self.out_channels = [base_dim * 2**i for i in range(len(depths))]
        self.in_channel = 32

        self.stem = nn.Sequential(StarConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i, depth in enumerate(depths):
            embed_dim = base_dim * 2**i
            down_sampler = StarConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [StarNetBlock(self.in_channel, mlp_ratio, dpr[cur + j]) for j in range(depth)]
            cur += depth
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.apply(self._init_weights)
        if pretrained or weights:
            self.load_pretrained(weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize weights like the official StarNet implementation."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, weights: str | None = None) -> None:
        """Load official classification checkpoint weights while ignoring norm/head keys."""
        if weights:
            checkpoint = torch.load(weights, map_location="cpu")
        else:
            url = STARNET_VARIANTS[self.variant]["url"]
            if not url:
                raise ValueError(f"No official pretrained checkpoint is defined for StarNet-{self.variant}.")
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint)
        filtered = OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in state_dict.items()
            if not key.removeprefix("module.").startswith(("head.", "norm."))
        )
        self.load_state_dict(filtered, strict=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps with strides 4, 8, 16, and 32."""
        x = self.stem(x)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in self.out_indices:
                outputs.append(x)
        return outputs
