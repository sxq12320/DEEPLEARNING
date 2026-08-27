"""Compatibility, forward, backward, and complexity tests for the exactly restored G10 architecture."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import SegmentP2Boundary
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
G10_YAML = (
    ROOT
    / "0_orange_yaml"
    / "legacy_pre20260824"
    / "2026_8_20_gpt_test"
    / "10_yolo11n-seg-hybrid-lska-carafe-bifpn-p2b.yaml"
)


def test_g10_builds_through_official_yolo_yaml_api() -> None:
    wrapper = YOLO(str(G10_YAML), task="segment", verbose=False)
    assert isinstance(wrapper.model.model[-1], SegmentP2Boundary)
    output = wrapper.model.eval()(torch.randn(1, 3, 128, 128))
    assert isinstance(output, tuple)


def test_g10_forward_backward_and_complexity_match_archive() -> None:
    model = SegmentationModel(G10_YAML, ch=3, nc=1, verbose=False)
    args = dict(DEFAULT_CFG_DICT)
    args.update(overlap_mask=True)
    model.args = IterableSimpleNamespace(**args)
    masks = torch.zeros(1, 32, 32)
    masks[0, 8:24, 9:23] = 1
    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.50, 0.50, 0.44, 0.50]]),
        "masks": masks,
    }
    total, components = model.loss(batch)
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()
    head = model.model[-1]
    assert head.p2_to_proto.weight.grad is not None
    assert head.p2_to_proto.weight.grad.abs().sum() > 0
    assert get_flops(model.eval(), imgsz=640) == pytest.approx(14.5404544, rel=1e-5)
