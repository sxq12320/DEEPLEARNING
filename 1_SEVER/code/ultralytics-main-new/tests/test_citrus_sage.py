"""Build, gradient, operator-contract, checkpoint, and complexity tests for SAGE."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import (
    C3k2SAGE,
    C3k2SAGEShape,
    CitrusSAGEFuse,
    CitrusSAGEInnovationPyramid,
    CitrusSAGEPyramid,
    SegmentCitrusSAGE,
    SegmentCitrusSAGEV2,
)
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_series"
SAGE_V1_YAMLS = sorted(YAML_DIR.glob("SAGE0*.yaml"))
SAGE_V2_YAMLS = sorted(YAML_DIR.glob("SAGE1*.yaml"))
SAGE_V3_YAMLS = sorted(YAML_DIR.glob("SAGE2*.yaml"))
SAGE_YAMLS = [*SAGE_V1_YAMLS, *SAGE_V2_YAMLS, *SAGE_V3_YAMLS]


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Create tiny and touching strip-occluded instances."""
    masks = torch.zeros(1, 64, 64)
    masks[0, 5:11, 6:12] = 1
    masks[0, 18:43, 14:34] = 2
    masks[0, 27:34, 27:34] = 0
    masks[0, 17:44, 34:54] = 3
    masks[0, 29:36, 34:40] = 0
    return {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0], [0.0]]),
        "bboxes": torch.tensor(
            [[0.14, 0.13, 0.10, 0.10], [0.38, 0.48, 0.31, 0.39], [0.69, 0.48, 0.31, 0.42]]
        ),
        "masks": masks,
    }


def _attach_loss_args(model: SegmentationModel) -> None:
    """Attach complete default/custom loss settings before criterion creation."""
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
    model.args = IterableSimpleNamespace(**args)


@pytest.mark.parametrize("yaml_path", SAGE_YAMLS, ids=lambda path: path.stem)
def test_every_sage_yaml_builds_and_forwards(yaml_path: Path) -> None:
    """Every public SAGE YAML must work through the standard YOLO API."""
    assert len(SAGE_V1_YAMLS) == 5
    assert len(SAGE_V2_YAMLS) == 8
    assert len(SAGE_V3_YAMLS) == 8
    model = YOLO(str(yaml_path), task="segment", verbose=False).model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None


def test_sage04_shapes_gradients_and_operator_contract() -> None:
    """The final route must train on step one and contain no latency-risk operator chain."""
    route = CitrusSAGEFuse((32, 64, 128, 256, 64), stage=4, width=24).train()
    features = [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 128, 8, 8),
        torch.randn(2, 256, 4, 4),
        torch.randn(2, 64, 16, 16),
    ]
    output = route(features)
    assert output.shape == (2, 64, 16, 16)
    output.square().mean().backward()
    assert route.detail_project[1].conv.weight.grad.abs().sum() > 0
    assert route.agreement.weight.grad.abs().sum() > 0
    assert route.route_scale.grad.abs().sum() > 0

    modules = list(route.modules())
    assert sum(isinstance(module, torch.nn.Conv2d) for module in modules) == 8
    assert sum(isinstance(module, torch.nn.PixelUnshuffle) for module in modules) == 1
    forbidden = (torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.MultiheadAttention)
    assert not any(isinstance(module, forbidden) for module in modules)


def test_sage04_standard_loss_backward_reaches_fusion() -> None:
    """Standard segmentation loss must reach both semantic and geometry routes."""
    model = SegmentationModel(YAML_DIR / "SAGE04_agreement_gate.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model)
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()
    head = model.model[-1]
    assert isinstance(head, SegmentCitrusSAGE)
    assert head.sage_fuse.p5_project.conv.weight.grad.abs().sum() > 0
    assert head.sage_fuse.detail_project[1].conv.weight.grad.abs().sum() > 0


def test_sage_series_stays_close_to_official_nano_compute() -> None:
    """SAGE is rejected if it hides a heavy neck behind a small route width."""
    for yaml_path in SAGE_YAMLS:
        model = SegmentationModel(yaml_path, ch=3, nc=80, verbose=False).eval()
        assert sum(parameter.numel() for parameter in model.parameters()) < 3_200_000
        assert get_flops(model, imgsz=640) < 11.5


def test_sage_v2_joint_loss_reaches_backbone_pyramid_and_topology() -> None:
    """The complete method must send gradients through every claimed architectural contribution."""
    model = SegmentationModel(YAML_DIR / "SAGE15_full_task_loss.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model)
    model.args.citrus_topology = 0.10
    model.args.citrus_boundary = 0.10
    model.args.citrus_query = 0.03
    model.args.citrus_concavity = 0.03
    model.args.citrus_exclusive = 0.02
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    assert isinstance(model.model[6], C3k2SAGE)
    assert isinstance(model.model[23], CitrusSAGEPyramid)
    assert isinstance(model.model[-1], SegmentCitrusSAGEV2)
    assert model.model[6].sage_scale.grad.abs().sum() > 0
    assert model.model[23].output3[0].weight.grad.abs().sum() > 0
    assert model.model[23].topology_predictor.weight.grad.abs().sum() > 0


def test_sage_v3_joint_loss_reaches_shape_innovation_and_topology() -> None:
    """The v3 task loss must reach the backbone, both innovation cells and shared topology gate."""
    model = SegmentationModel(YAML_DIR / "SAGE26_occlusion_topology.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model)
    model.args.citrus_topology = 0.10
    model.args.citrus_boundary = 0.10
    model.args.citrus_query = 0.03
    model.args.citrus_concavity = 0.03
    model.args.citrus_exclusive = 0.02
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()

    assert isinstance(model.model[6], C3k2SAGEShape)
    assert isinstance(model.model[23], CitrusSAGEInnovationPyramid)
    assert isinstance(model.model[-1], SegmentCitrusSAGEV2)
    assert model.model[6].sage_scale.grad.abs().sum() > 0
    assert model.model[23].innovation4.fuse.conv.weight.grad.abs().sum() > 0
    assert model.model[23].innovation3.fuse.conv.weight.grad.abs().sum() > 0
    assert model.model[23].topology_predictor.weight.grad.abs().sum() > 0


def test_sage_v3_style_swap_is_exactly_inactive_during_evaluation() -> None:
    """The colour-statistics ablation must add no stochastic inference behaviour."""
    block = C3k2SAGEShape(32, 32, style_probability=1.0, style_mix=1.0).eval()
    inputs = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        first = block(inputs)
        second = block(inputs)
    assert torch.equal(first, second)


def test_sage_v3_hot_path_excludes_fragmented_dynamic_operators() -> None:
    """The redesigned neck must not repeat the operators that slowed the Light series."""
    route = CitrusSAGEInnovationPyramid((32, 64, 128, 256, 64, 128, 256), (64, 128, 256), 24)
    modules = list(route.modules())
    forbidden = (torch.nn.MultiheadAttention, torch.nn.Unfold, torch.nn.Fold, torch.nn.AdaptiveAvgPool2d)
    assert not any(isinstance(module, forbidden) for module in modules)
    assert sum(isinstance(module, torch.nn.PixelUnshuffle) for module in modules) == 1


@pytest.mark.parametrize("yaml_path", SAGE_YAMLS, ids=lambda path: path.stem)
def test_every_sage_yaml_accepts_official_checkpoint(yaml_path: Path) -> None:
    """Every SAGE model must initialize through YOLO(yaml).load(yolo11n-seg.pt)."""
    model = YOLO(str(yaml_path), task="segment", verbose=False)
    model.load(str(ROOT / "yolo11n-seg.pt"))
    with torch.no_grad():
        output = model.model.eval()(torch.randn(1, 3, 128, 128))
    assert output is not None


def test_sage04_preserves_full_official_core_initialization() -> None:
    """Every non-SAGE tensor must retain an exact shape-compatible official source."""
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), task="segment", verbose=False).model.state_dict()
    target = YOLO(str(YAML_DIR / "SAGE04_agreement_gate.yaml"), task="segment", verbose=False).model.state_dict()
    official_core = {key: value for key, value in target.items() if not key.startswith("model.23.sage_fuse.")}
    unmatched = [key for key, value in official_core.items() if key not in source or source[key].shape != value.shape]
    assert not unmatched
