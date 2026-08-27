"""Construction, gradient, transfer, and deploy-path tests for CitrusSwift."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "20260824_citrus_swift"
FULL_YAML = YAML_DIR / "08_citrus_swift_full.yaml"


def _collect_tensors(value) -> list[torch.Tensor]:
    """Flatten tensors from nested Ultralytics inference outputs."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _collect_tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _collect_tensors(item)]
    return []


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_all_swift_yamls_build_and_forward(yaml_path: Path) -> None:
    model = SegmentationModel(yaml_path, ch=3, nc=1, verbose=False).eval()
    output = model(torch.randn(1, 3, 128, 128))
    assert isinstance(output, tuple)
    assert get_flops(model, imgsz=128) > 0


def test_auxiliary_branches_exist_only_in_training_output() -> None:
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    train_output = model.train()(torch.randn(1, 3, 128, 128))
    assert {"citrus_boundary", "citrus_query", "citrus_contrast"}.issubset(train_output)
    eval_output = model.eval()(torch.randn(1, 3, 128, 128))
    predictions = eval_output[1]
    assert not any(key.startswith("citrus_") for key in predictions)


def test_full_loss_and_nwd_backward_are_finite() -> None:
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.25,
        citrus_concavity=0.10,
        citrus_query=0.05,
        citrus_contrast=0.10,
        citrus_exclusive=0.05,
        nwd_ratio=0.25,
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
    assert torch.isfinite(total).all()
    total.sum().backward()
    auxiliary = model.model[-1].citrus_aux
    assert auxiliary.boundary_predictor[-1].weight.grad.abs().sum() > 0
    assert auxiliary.query_predictor.weight.grad.abs().sum() > 0
    assert auxiliary.contrast_predictor[-1].weight.grad.abs().sum() > 0


def test_full_candidate_retains_at_least_90_percent_pretrained_state() -> None:
    checkpoint = torch.load(ROOT / "yolo11n-seg.pt", map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    target = model.state_dict()
    matched = intersect_dicts(source, target, exclude=())
    ratio = sum(target[key].numel() for key in matched) / sum(value.numel() for value in target.values())
    assert ratio >= 0.90


def test_repcontext_fuse_is_numerically_equivalent() -> None:
    model = SegmentationModel(YAML_DIR / "01_repcontext_backbone.yaml", ch=3, nc=1, verbose=False).eval()
    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = _collect_tensors(model(sample))
    model.fuse(verbose=False)
    with torch.no_grad():
        after = _collect_tensors(model(sample))
    assert len(before) == len(after)
    for unfused, fused in zip(before, after):
        torch.testing.assert_close(unfused, fused, atol=1e-6, rtol=1e-5)
