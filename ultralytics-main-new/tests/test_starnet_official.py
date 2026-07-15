"""Focused tests for the official StarNet backbone integration."""

from __future__ import annotations

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import StarNetBackbone, StarNetBlock


def test_official_starnet_backbone_shapes() -> None:
    """StarNet-S1 should return P2/P3/P4/P5 with official channel counts."""
    model = StarNetBackbone("s1", pretrained=False)
    x = torch.randn(1, 3, 128, 128)

    outputs = model(x)

    assert [tuple(feature.shape) for feature in outputs] == [
        (1, 24, 32, 32),
        (1, 48, 16, 16),
        (1, 96, 8, 8),
        (1, 192, 4, 4),
    ]


def test_official_starnet_block_matches_source_structure() -> None:
    """The block should keep the official layer names and avoid layer-scale gamma."""
    block = StarNetBlock(24, mlp_ratio=4)

    assert hasattr(block, "dwconv")
    assert hasattr(block, "f1")
    assert hasattr(block, "f2")
    assert hasattr(block, "g")
    assert hasattr(block, "dwconv2")
    assert not hasattr(block, "gamma")


def test_yolo_seg_starnet_official_build_forward_backward() -> None:
    """The YOLO segmentation model should build, run forward, and backpropagate."""
    model = YOLO("0_orange_yaml/002_yolo11-seg-starnet-official-s1.yaml").model
    x = torch.randn(1, 3, 64, 64)

    outputs = model(x)
    loss = sum(value.float().mean() for value in outputs.values() if torch.is_tensor(value))
    loss.backward()

    first_parameter = next(model.parameters())
    assert first_parameter.grad is not None
