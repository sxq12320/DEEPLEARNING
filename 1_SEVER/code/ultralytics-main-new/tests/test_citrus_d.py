"""Construction, gradient, transfer, and complexity checks for CitrusD."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "D_series"
FULL_YAML = YAML_DIR / "06_d06_shape_semantic_full.yaml"
TOPOLOGY_YAML = YAML_DIR / "08_d08_topology_masks.yaml"


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_official_api_build_forward_and_flops(yaml_path: Path) -> None:
    """Every D model must work through the public YOLO YAML entry point."""
    wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
    output = wrapper.model.eval()(torch.randn(1, 3, 64, 64))
    assert isinstance(output, tuple)
    assert get_flops(wrapper.model, imgsz=64) > 0


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Return two nearby, partially concave-looking masks for loss smoke tests."""
    masks = torch.zeros(1, 64, 64)
    masks[0, 9:34, 8:30] = 1
    masks[0, 18:29, 23:30] = 0
    masks[0, 13:40, 30:53] = 2
    masks[0, 24:35, 30:37] = 0
    return {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.30, 0.34, 0.34, 0.39], [0.65, 0.41, 0.36, 0.42]]),
        "masks": masks,
    }


def _model_args(**overrides: float | bool) -> IterableSimpleNamespace:
    """Create a complete loss namespace while only enabling the requested objectives."""
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.0,
        citrus_concavity=0.0,
        citrus_query=0.0,
        citrus_contrast=0.0,
        citrus_exclusive=0.0,
        citrus_quality=0.0,
        citrus_topology=0.0,
        citrus_vfl=0.0,
        nwd_ratio=0.0,
    )
    args.update(overrides)
    return IterableSimpleNamespace(**args)


def test_shape_semantic_full_backward_reaches_every_new_path() -> None:
    """Mask and auxiliary losses must train the structure stem, shape stream, fusion, and head."""
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    model.args = _model_args(citrus_boundary=0.15, citrus_query=0.03)

    # Fusion is intentionally identity-initialized. Nudge only its scalar so this one-step
    # test verifies the complete gradient path that becomes active after the first update.
    model.model[12].residual_scale.data.fill_(0.05)
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    shape_stream = model.model[11]
    fusion = model.model[12]
    auxiliary = model.model[-1].citrus_aux
    assert model.model[0].structure.weight.grad.abs().sum() > 0
    assert shape_stream.updates[0].spatial.weight.grad.abs().sum() > 0
    assert shape_stream.semantic_queries[-1].conv.weight.grad.abs().sum() > 0
    assert fusion.residual_scale.grad.abs().sum() > 0
    assert fusion.shape_to_p3[1].conv.weight.grad.abs().sum() > 0
    assert auxiliary.boundary_predictor[-1].weight.grad.abs().sum() > 0
    assert auxiliary.query_predictor.weight.grad.abs().sum() > 0


def test_topology_variant_backward_reaches_dual_prototypes() -> None:
    """The optional topology variant must train both detail and separation branches."""
    model = SegmentationModel(TOPOLOGY_YAML, ch=3, nc=1, verbose=False)
    model.args = _model_args(citrus_topology=0.05)
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    prototype = model.model[-1].proto
    assert prototype.detail_prototypes.conv.weight.grad.abs().sum() > 0
    assert prototype.topology_predictor.weight.grad.abs().sum() > 0


def test_primary_and_deploy_candidates_remain_nano_scale() -> None:
    """Keep both accuracy and deployment candidates inside the declared compute budget."""
    limits = {
        "06_d06_shape_semantic_full.yaml": (3_100_000, 13.0),
        "07_d07_deploy_lite.yaml": (2_850_000, 12.0),
    }
    for filename, (parameter_limit, flop_limit) in limits.items():
        model = SegmentationModel(YAML_DIR / filename, ch=3, nc=80, verbose=False).eval()
        assert sum(parameter.numel() for parameter in model.parameters()) < parameter_limit
        assert get_flops(model, imgsz=640) < flop_limit


def test_public_load_uses_explicit_shift_map() -> None:
    """Inserted D layers must not prevent transfer of the standard YOLO neck."""
    checkpoint_path = ROOT / "yolo11n-seg.pt"
    if not checkpoint_path.is_file():
        pytest.skip("YOLO11n-seg checkpoint is not available for transfer audit")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    wrapper = YOLO(str(FULL_YAML), task="segment", verbose=False).load(str(checkpoint_path))
    target = wrapper.model.state_dict()

    # Standard layer 13 (first neck C3k2) moved to D layer 15. Check one tensor exactly.
    source_key = "model.13.cv1.conv.weight"
    target_key = "model.15.cv1.conv.weight"
    assert source_key in source and target_key in target
    assert torch.equal(target[target_key].cpu(), source[source_key].cpu())
