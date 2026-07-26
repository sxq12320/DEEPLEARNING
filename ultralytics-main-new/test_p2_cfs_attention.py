"""Smoke tests for the P2 channel-frequency-spatial attention path."""

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import P2CFSAttention


def test_p2_cfs_shape_and_finite_output():
    module = P2CFSAttention(64, 64)
    x = torch.randn(1, 64, 81, 79, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y - x).abs().mean() < 1.0


def test_p2_cfs_backward():
    module = P2CFSAttention(64, 64)
    x = torch.randn(1, 64, 40, 40, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_local_orientation_is_a_probability_map():
    module = P2CFSAttention(64, 64)
    frequency = torch.randn(2, module.channels[1], 83, 77)
    orientation = module.spectral_gate._local_orientation(frequency)
    assert orientation.shape == (2, 4, 83, 77)
    assert torch.allclose(orientation.sum(1), torch.ones(2, 83, 77), atol=1e-6)


def test_p2_cfs_model_build_and_train_forward():
    model = YOLO("0_orange_yaml/012_yolo11-seg-p2-cfs.yaml").model
    model.train()
    outputs = model(torch.randn(1, 3, 256, 256))
    assert isinstance(outputs, dict)
    assert "proto" in outputs
    assert outputs["proto"].shape[-2:] == (64, 64)
    assert torch.count_nonzero(model.model[-1].p2_project[-1].weight) == 0
