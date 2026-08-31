"""Construction, gradient, and complexity tests for the G_0839 citrus series."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, CitrusDualResolutionBackbone, SegmentCitrusSDR
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "G_0839_series"
FULL_YAML = YAML_DIR / "05_g05_full_sdr.yaml"


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_public_yaml_build_forward_and_flops(yaml_path: Path) -> None:
    """Every G_0839 YAML must build and infer through the public YOLO API."""
    wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
    output = wrapper.model.eval()(torch.randn(1, 3, 64, 64))
    assert isinstance(output, tuple)
    assert get_flops(wrapper.model, imgsz=64) > 0


@pytest.mark.parametrize("filename", [f"0{index}_g0{index}_{suffix}.yaml" for index, suffix in enumerate((
    "lite_control", "preserve", "search", "discriminate", "boundary_refine", "full_sdr"
))])
def test_series_uses_one_registered_head(filename: str) -> None:
    """All stages must remain standard-YAML models with the same registered head family."""
    model = SegmentationModel(YAML_DIR / filename, ch=3, nc=1, verbose=False)
    assert isinstance(model.model[-1], SegmentCitrusSDR)


def test_g01_really_replaces_the_c3k2_backbone() -> None:
    """The treatment backbone must be structurally different from the YOLO control."""
    model = SegmentationModel(YAML_DIR / "01_g01_preserve.yaml", ch=3, nc=1, verbose=False)
    backbone = model.model[0]
    assert isinstance(backbone, CitrusDualResolutionBackbone)
    assert not any(isinstance(module, C3k2) for module in backbone.modules())
    features = backbone(torch.randn(1, 3, 128, 128))
    assert [feature.shape[-1] for feature in features] == [32, 16, 8, 4]


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Create two adjacent masks with strip-like concavities."""
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


def _model_args() -> IterableSimpleNamespace:
    """Enable exactly the documented full-model auxiliary objectives."""
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.10,
        citrus_concavity=0.0,
        citrus_query=0.03,
        citrus_contrast=0.05,
        citrus_exclusive=0.0,
        citrus_quality=0.0,
        citrus_topology=0.05,
        citrus_vfl=0.0,
        nwd_ratio=0.0,
    )
    return IterableSimpleNamespace(**args)


def test_full_model_backward_reaches_backbone_and_every_sdr_stage() -> None:
    """The registered losses must train the new backbone and all full SDR predictions."""
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    model.args = _model_args()
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    backbone = model.model[0]
    support = model.model[-1].sdr_support
    assert backbone.exchanges[0].detail_update[0].conv.weight.grad.abs().sum() > 0
    assert support.query_predictor[-1].weight.grad.abs().sum() > 0
    assert support.context_predictor[-1].weight.grad.abs().sum() > 0
    assert support.boundary_predictor[-1].weight.grad.abs().sum() > 0
    assert support.topology_predictor.weight.grad.abs().sum() > 0
    assert support.prototype_residual[-1].weight.grad.abs().sum() > 0


def test_full_series_stays_inside_nano_budget() -> None:
    """The final architecture must stay below the declared parameter/FLOP ceiling."""
    model = SegmentationModel(FULL_YAML, ch=3, nc=80, verbose=False).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_850_000
    assert get_flops(model, imgsz=640) < 12.0
