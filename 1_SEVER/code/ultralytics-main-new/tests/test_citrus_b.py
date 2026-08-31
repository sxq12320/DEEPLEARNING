"""Construction, gradients, complexity, and transfer checks for CitrusB."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "B_series"
FINAL_YAML = YAML_DIR / "09_b09_recall_balanced_final.yaml"


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_official_api_build_and_forward(yaml_path: Path) -> None:
    """Every B model must work through the public YOLO YAML entry point."""
    wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
    output = wrapper.model.eval()(torch.randn(1, 3, 128, 128))
    assert isinstance(output, tuple)
    assert get_flops(wrapper.model, imgsz=128) > 0


def test_final_loss_backward_reaches_all_new_paths() -> None:
    """Context, scale-fusion, boundary, and tiny-query paths must receive gradients."""
    model = SegmentationModel(FINAL_YAML, ch=3, nc=1, verbose=False)
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.25,
        citrus_query=0.05,
        citrus_contrast=0.0,
        citrus_exclusive=0.0,
        citrus_quality=0.0,
        citrus_vfl=0.0,
        nwd_ratio=0.0,
    )
    model.args = IterableSimpleNamespace(**args)
    masks = torch.zeros(1, 64, 64)
    masks[0, 9:31, 8:28] = 1
    masks[0, 13:38, 28:50] = 2
    batch = {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.28, 0.31, 0.31, 0.34], [0.61, 0.40, 0.34, 0.39]]),
        "masks": masks,
    }
    total, _ = model.loss(batch)
    assert torch.isfinite(total).all()
    total.sum().backward()
    head = model.model[-1]
    assert model.model[9].context_scale.grad.abs().sum() > 0
    assert model.model[15].gate[-1].weight.grad.abs().sum() > 0
    assert head.citrus_bq_aux.boundary_predictor.weight.grad.abs().sum() > 0
    assert head.citrus_bq_aux.query[-1].weight.grad.abs().sum() > 0


def test_final_is_nano_scale_and_retains_pretrained_core() -> None:
    """Keep the final candidate below 3.1M parameters and 12 GFLOPs at 640."""
    model = SegmentationModel(FINAL_YAML, ch=3, nc=80, verbose=False)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_100_000
    assert get_flops(model, imgsz=640) < 12.0

    checkpoint = torch.load(ROOT / "yolo11n-seg.pt", map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    target = model.state_dict()
    matched = intersect_dicts(source, target, exclude=())
    ratio = sum(target[key].numel() for key in matched) / sum(value.numel() for value in target.values())
    assert ratio >= 0.70
