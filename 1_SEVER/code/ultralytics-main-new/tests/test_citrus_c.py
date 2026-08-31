"""Construction, gradient, topology-target, and complexity checks for CitrusC."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "C_series"
CORE_YAML = YAML_DIR / "03_c03_dualproto_core.yaml"


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_official_api_build_and_forward(yaml_path: Path) -> None:
    """Every C YAML must work through the public YOLO model entry point."""
    wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
    output = wrapper.model.eval()(torch.randn(1, 3, 64, 64))
    assert isinstance(output, tuple)
    assert get_flops(wrapper.model, imgsz=64) > 0


def test_dualproto_loss_backward_reaches_task_specific_paths() -> None:
    """Mask and topology supervision must train the P2 detail and P3 semantic paths."""
    model = SegmentationModel(CORE_YAML, ch=3, nc=1, verbose=False)
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.0,
        citrus_concavity=0.0,
        citrus_query=0.0,
        citrus_contrast=0.0,
        citrus_exclusive=0.0,
        citrus_quality=0.0,
        citrus_topology=0.10,
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
    total, components = model.loss(batch)
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    prototype = model.model[-1].proto
    assert prototype.detail_prototypes.conv.weight.grad.abs().sum() > 0
    assert prototype.topology_predictor.weight.grad.abs().sum() > 0
    assert prototype.p3_residual_scale.grad.abs().sum() > 0
    assert prototype.cv1.conv.weight.grad.abs().sum() > 0


def test_dualproto_is_nano_scale_and_retains_pretrained_core() -> None:
    """The task-specific core must remain deployable at nano scale."""
    model = SegmentationModel(CORE_YAML, ch=3, nc=80, verbose=False)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_100_000
    assert get_flops(model, imgsz=640) < 13.0

    checkpoint_path = ROOT / "yolo11n-seg.pt"
    if not checkpoint_path.is_file():
        pytest.skip("YOLO11n-seg checkpoint is not available for transfer audit")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    target = model.state_dict()
    matched = intersect_dicts(source, target, exclude=())
    ratio = sum(target[key].numel() for key in matched) / sum(value.numel() for value in target.values())
    assert ratio >= 0.75
