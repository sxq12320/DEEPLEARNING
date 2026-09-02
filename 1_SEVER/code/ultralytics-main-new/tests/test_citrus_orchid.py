"""Build, routing, loss, checkpoint, and complexity tests for ORCHID."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import CitrusORCHIDMaskRouter, CitrusORCHIDNeck, SegmentCitrusORCHID
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "ORCHID_series"
ORCHID_YAMLS = sorted(YAML_DIR.glob("ORCHID*.yaml"))


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Create a tiny fruit and two touching, strip-occluded fruits."""
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
            [
                [0.14, 0.13, 0.10, 0.10],
                [0.38, 0.48, 0.31, 0.39],
                [0.69, 0.48, 0.31, 0.42],
            ]
        ),
        "masks": masks,
    }


def _attach_loss_args(model: SegmentationModel, **overrides: float) -> None:
    """Attach complete loss settings before constructing the criterion."""
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
    model.args = IterableSimpleNamespace(**args)


@pytest.mark.parametrize("yaml_path", ORCHID_YAMLS, ids=lambda path: path.stem)
def test_every_orchid_yaml_builds_and_forwards(yaml_path: Path) -> None:
    """Every public ORCHID model must work through the standard YOLO YAML API."""
    assert len(ORCHID_YAMLS) == 7
    model = YOLO(str(yaml_path), task="segment", verbose=False).model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None


def test_mask_router_shapes_and_first_step_gradients() -> None:
    """Candidate routing must train the query, detail, context, and residual scale on step one."""
    router = CitrusORCHIDMaskRouter((32, 64, 128, 256, 64), mode=3, route_channels=24).train()
    features = [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 128, 8, 8),
        torch.randn(2, 256, 4, 4),
        torch.randn(2, 64, 16, 16),
    ]
    refined, query, contrast = router(features)
    assert refined.shape == (2, 64, 16, 16)
    assert query.shape == contrast.shape == (2, 1, 32, 32)
    (refined.square().mean() + query.square().mean() + contrast.square().mean()).backward()
    assert router.query_predictor.weight.grad.abs().sum() > 0
    assert router.p2_detail.conv.weight.grad.abs().sum() > 0
    assert router.route_scale.grad.abs().sum() > 0


def test_single_canvas_neck_shapes_and_gradients() -> None:
    """The non-PAN neck must expose three detection levels and one P2 query map."""
    neck = CitrusORCHIDNeck((32, 64, 128, 256), (64, 128, 256), route_channels=24).train()
    features = [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 128, 8, 8),
        torch.randn(2, 256, 4, 4),
    ]
    outputs = neck(features)
    assert [tuple(output.shape) for output in outputs] == [
        (2, 64, 16, 16),
        (2, 128, 8, 8),
        (2, 256, 4, 4),
        (2, 1, 32, 32),
    ]
    sum(output.square().mean() for output in outputs).backward()
    assert neck.query_predictor.weight.grad.abs().sum() > 0
    assert neck.detail_scale.grad.abs().sum() > 0
    assert neck.p4_scale.grad.abs().sum() > 0


def test_orchid03_loss_backward_reaches_supervised_query() -> None:
    """The QueryDet-style auxiliary term must produce finite gradients in the causal gate."""
    model = SegmentationModel(YAML_DIR / "ORCHID03_supervised_query_router.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model, citrus_query=0.10)
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()
    head = model.model[-1]
    assert isinstance(head, SegmentCitrusORCHID)
    assert head.orchid_router.query_predictor.weight.grad.abs().sum() > 0


def test_orchid05_loss_backward_reaches_decam_reference() -> None:
    """Camouflage contrast supervision must train the local-reference path."""
    model = SegmentationModel(YAML_DIR / "ORCHID05_decam_reference.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model, citrus_query=0.10, citrus_contrast=0.05)
    total, components = model.loss(_synthetic_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()
    predictor = model.model[-1].orchid_router.contrast_predictor[-1]
    assert predictor.weight.grad is not None and predictor.weight.grad.abs().sum() > 0


def test_orchid_series_stays_in_the_nano_compute_range() -> None:
    """The series must not hide a large backbone or transformer inside its routing claim."""
    for yaml_path in ORCHID_YAMLS:
        model = SegmentationModel(yaml_path, ch=3, nc=80, verbose=False).eval()
        assert sum(parameter.numel() for parameter in model.parameters()) < 3_100_000
        assert get_flops(model, imgsz=640) < 12.5


@pytest.mark.parametrize("yaml_path", ORCHID_YAMLS, ids=lambda path: path.stem)
def test_every_orchid_yaml_accepts_official_checkpoint(yaml_path: Path) -> None:
    """Every model remains trainable via YOLO(yaml).load(yolo11n-seg.pt)."""
    model = YOLO(str(yaml_path), task="segment", verbose=False)
    model.load(str(ROOT / "yolo11n-seg.pt"))
    with torch.no_grad():
        output = model.model.eval()(torch.randn(1, 3, 128, 128))
    assert output is not None
