"""Focused construction and gradient tests for CitrusTopo-Seg."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "L_series"
FULL_YAML = YAML_DIR / "07_citrus_toposeg_full.yaml"


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_all_citrus_topo_yamls_build_and_forward(yaml_path: Path) -> None:
    model = SegmentationModel(yaml_path, ch=3, nc=1, verbose=False).train()
    output = model(torch.randn(1, 3, 128, 128))
    assert output["proto"].shape == (1, 32, 32, 32)


def test_full_model_forward_backward_and_flops() -> None:
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False).train()
    output = model(torch.randn(1, 3, 128, 128))
    objective = output["proto"].mean() + output["citrus_boundary"].mean() + output["citrus_query"].mean()
    objective.backward()
    assert model.model[-1].topology_fusion.boundary_predictor.weight.grad is not None
    assert get_flops(model, imgsz=128) > 0


def test_full_model_retains_at_least_95_percent_pretrained_state() -> None:
    checkpoint = torch.load(ROOT / "yolo11n-seg.pt", map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    target = model.state_dict()
    matched = intersect_dicts(source, target, exclude=())
    ratio = sum(target[key].numel() for key in matched) / sum(value.numel() for value in target.values())
    assert ratio >= 0.95


def test_all_task_specific_losses_backward() -> None:
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.50,
        citrus_concavity=0.25,
        citrus_query=0.10,
        citrus_exclusive=0.10,
    )
    model.args = IterableSimpleNamespace(**args)
    masks = torch.zeros(1, 64, 64)
    masks[0, 10:33, 9:29] = 1
    masks[0, 14:39, 29:50] = 2
    batch = {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.30, 0.34, 0.31, 0.36], [0.62, 0.41, 0.33, 0.39]]),
        "masks": masks,
    }
    total, _ = model.loss(batch)
    total.sum().backward()
    fusion = model.model[-1].topology_fusion
    assert fusion.boundary_predictor.weight.grad.abs().sum() > 0
    assert fusion.query[-1].weight.grad.abs().sum() > 0
    assert fusion.mask_scale.grad.abs().sum() > 0
