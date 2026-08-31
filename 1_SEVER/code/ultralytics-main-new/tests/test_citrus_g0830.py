"""Construction, identity initialization, gradient, and budget tests for G_0830."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import CitrusBilateralExchange, CitrusFrequencyAlignedConcat, CitrusRepMixerStage
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "G_0830_series"
FULL_YAML = YAML_DIR / "03_g03_frequency_neck.yaml"


@pytest.mark.parametrize("yaml_path", sorted(YAML_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_public_yaml_build_forward_and_flops(yaml_path: Path) -> None:
    """Every YAML must build, infer, and expose measurable complexity through public APIs."""
    wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
    output = wrapper.model.eval()(torch.randn(1, 3, 64, 64))
    assert isinstance(output, tuple)
    assert get_flops(wrapper.model, imgsz=64) > 0


def test_bilateral_exchange_is_exact_identity_at_initialization() -> None:
    """New backbone paths must not perturb mapped pretrained features before learning."""
    module = CitrusBilateralExchange((32, 128), ratio=4).eval()
    detail = torch.randn(2, 32, 32, 32)
    semantic = torch.randn(2, 128, 8, 8)
    detail_out, semantic_out = module((detail, semantic))
    torch.testing.assert_close(detail_out, detail)
    torch.testing.assert_close(semantic_out, semantic)


@pytest.mark.parametrize("direction", ["topdown", "bottomup"])
def test_frequency_concat_is_exact_concat_at_initialization(direction: str) -> None:
    """Frequency alignment must preserve the following pretrained PAN bottleneck's input."""
    module = CitrusFrequencyAlignedConcat((64, 128), direction=direction).eval()
    first = torch.randn(2, 64, 16, 16)
    second = torch.randn(2, 128, 16, 16)
    output = module((first, second))
    torch.testing.assert_close(output, torch.cat((first, second), dim=1))


def test_g04_replaces_deep_c3k2_with_registered_repmixer() -> None:
    """The lightweight treatment must contain two genuine non-CSP deep stages."""
    model = SegmentationModel(YAML_DIR / "04_g04_deep_repmixer.yaml", ch=3, nc=1, verbose=False)
    assert sum(isinstance(module, CitrusRepMixerStage) for module in model.modules()) == 2


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Create adjacent fruits with strip-like missing regions for a gradient smoke test."""
    masks = torch.zeros(1, 64, 64)
    masks[0, 8:34, 7:30] = 1
    masks[0, 17:29, 23:30] = 0
    masks[0, 12:40, 30:54] = 2
    masks[0, 23:35, 30:37] = 0
    return {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.29, 0.33, 0.35, 0.40], [0.66, 0.41, 0.37, 0.43]]),
        "masks": masks,
    }


def _model_args() -> IterableSimpleNamespace:
    """Return the fixed structure-screening auxiliary vector."""
    args = dict(DEFAULT_CFG_DICT)
    args.update(
        overlap_mask=True,
        citrus_boundary=0.15,
        citrus_concavity=0.0,
        citrus_query=0.03,
        citrus_contrast=0.0,
        citrus_exclusive=0.0,
        citrus_quality=0.0,
        citrus_topology=0.0,
        citrus_vfl=0.0,
        nwd_ratio=0.0,
    )
    return IterableSimpleNamespace(**args)


def test_full_model_backward_reaches_every_zero_initialized_exchange() -> None:
    """One loss pass must produce finite gradients for all new residual gates."""
    model = SegmentationModel(FULL_YAML, ch=3, nc=1, verbose=False)
    model.args = _model_args()
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    bilateral = [module for module in model.modules() if isinstance(module, CitrusBilateralExchange)]
    aligned = [module for module in model.modules() if isinstance(module, CitrusFrequencyAlignedConcat)]
    assert len(bilateral) == 3
    assert len(aligned) == 4
    assert all(module.detail_scale.grad is not None for module in bilateral)
    assert all(module.semantic_scale.grad is not None for module in bilateral)
    assert all(module.detail_scale.grad is not None for module in aligned)
    assert all(module.semantic_scale.grad is not None for module in aligned)


def test_series_respects_declared_nano_budget() -> None:
    """Every treatment must stay near nano scale rather than hiding a large backbone."""
    for yaml_path in sorted(YAML_DIR.glob("*.yaml")):
        model = SegmentationModel(yaml_path, ch=3, nc=80, verbose=False).eval()
        assert sum(parameter.numel() for parameter in model.parameters()) < 3_100_000
        assert get_flops(model, imgsz=640) < 12.0
