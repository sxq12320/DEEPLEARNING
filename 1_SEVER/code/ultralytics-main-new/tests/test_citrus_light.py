"""Build, gradient-flow and complexity tests for the Citrus Light series."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules import C2PSA, C3k2, CitrusLightAFPN, CitrusLightStage, CitrusRepMixerStage
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "Light_series"
LIGHT_YAMLS = sorted(YAML_DIR.glob("Light*.yaml"))


def _synthetic_segmentation_batch() -> dict[str, torch.Tensor]:
    """Create two labeled citrus-like instances for a real loss/backward smoke test."""
    masks = torch.zeros(1, 64, 64)
    masks[0, 9:31, 8:28] = 1
    masks[0, 13:38, 28:50] = 2
    return {
        "img": torch.rand(1, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.28, 0.31, 0.31, 0.34], [0.61, 0.40, 0.34, 0.39]]),
        "masks": masks,
    }


def _attach_loss_args(model: SegmentationModel, **overrides: float) -> None:
    """Attach complete Ultralytics loss arguments before criterion construction."""
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


def _tensor_sum(output) -> torch.Tensor:
    """Return one differentiable scalar from nested Ultralytics head outputs."""
    if torch.is_tensor(output):
        return output.float().mean()
    if isinstance(output, dict):
        values = [_tensor_sum(value) for value in output.values()]
    elif isinstance(output, (list, tuple)):
        values = [_tensor_sum(value) for value in output]
    else:
        values = []
    if not values:
        raise TypeError(f"No tensor found in output of type {type(output)!r}")
    return torch.stack(values).sum()


@pytest.mark.parametrize("yaml_path", LIGHT_YAMLS, ids=lambda path: path.stem)
def test_light_yaml_build_and_forward(yaml_path: Path):
    """Every public Light YAML must build through the official YOLO entry point."""
    assert len(LIGHT_YAMLS) == 8
    model = YOLO(str(yaml_path), task="segment", verbose=False).model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None


def test_light_stage_has_first_step_gradients():
    """The partial spatial mixer must not be frozen by a zero-initialized branch."""
    stage = CitrusLightStage(32, 48, blocks=2, expansion=0.5, division=4, layer_scale=0.1).train()
    loss = stage(torch.randn(2, 32, 16, 16)).square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in stage.named_parameters()
        if "partial_conv.weight" in name
    ]
    assert len(gradients) == 2
    assert all(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)


def test_light_afpn_shapes_and_first_step_gradients():
    """Progressive P2-to-P5 fusion must preserve shapes and train every transition."""
    neck = CitrusLightAFPN([64, 128, 128, 256], [64, 128, 256], gate_channels=8).train()
    features = [
        torch.randn(2, 64, 32, 32, requires_grad=True),
        torch.randn(2, 128, 16, 16, requires_grad=True),
        torch.randn(2, 128, 8, 8, requires_grad=True),
        torch.randn(2, 256, 4, 4, requires_grad=True),
    ]
    outputs = neck(features)
    assert [tuple(output.shape) for output in outputs] == [
        (2, 64, 16, 16),
        (2, 128, 8, 8),
        (2, 256, 4, 4),
    ]
    sum(output.square().mean() for output in outputs).backward()
    assert neck.p2_to_p3.project.conv.weight.grad.abs().sum() > 0
    assert neck.gather_p3.weight.weight.grad.abs().sum() > 0
    assert neck.gather_p3.mix_logit.grad.abs().sum() > 0
    assert neck.distribute_p3.weight.weight.grad.abs().sum() > 0


def test_light_afpn_starts_near_destination_identity():
    """An untrained fusion node must preserve most destination evidence."""
    neck = CitrusLightAFPN([64, 128, 128, 256], [64, 128, 256], gate_channels=8).eval()
    destination = torch.ones(1, 64, 8, 8)
    source = torch.zeros_like(destination)
    output = neck.gather_p3(destination, source)
    assert torch.allclose(output, torch.full_like(output, 0.95), atol=1e-5)


def test_structure_screen_is_a_clean_backbone_by_neck_factorial():
    """The structural queue must not hide context/head changes inside its backbone comparison."""
    light00 = YOLO(str(YAML_DIR / "Light00_backbone_only.yaml"), task="segment", verbose=False).model
    light01 = YOLO(str(YAML_DIR / "Light01_neck_only.yaml"), task="segment", verbose=False).model
    light02 = YOLO(str(YAML_DIR / "Light02_joint_core.yaml"), task="segment", verbose=False).model
    light05 = YOLO(str(YAML_DIR / "Light05_repmixer_backbone_only.yaml"), task="segment", verbose=False).model
    light06 = YOLO(str(YAML_DIR / "Light06_repmixer_afpn.yaml"), task="segment", verbose=False).model

    assert isinstance(light00.model[6], CitrusLightStage)
    assert isinstance(light00.model[8], CitrusLightStage)
    assert isinstance(light01.model[6], C3k2)
    assert isinstance(light01.model[8], C3k2)
    assert isinstance(light02.model[6], CitrusLightStage)
    assert isinstance(light05.model[6], CitrusRepMixerStage)
    assert isinstance(light06.model[6], CitrusRepMixerStage)
    assert all(isinstance(model.model[10], C2PSA) for model in (light00, light01, light02, light05, light06))
    assert light00.model[-1].__class__.__name__ == light01.model[-1].__class__.__name__ == "Segment"
    assert light02.model[-1].__class__.__name__ == light05.model[-1].__class__.__name__ == "Segment"


def test_light03_and_light04_differ_only_in_the_quality_head():
    """The quality-ranking experiment must preserve the complete deploy feature extractor."""
    light03 = YOLO(str(YAML_DIR / "Light03_deploy_lite.yaml"), task="segment", verbose=False).model
    light04 = YOLO(str(YAML_DIR / "Light04_quality_rank.yaml"), task="segment", verbose=False).model
    signature03 = [module.__class__.__name__ for module in light03.model[:-1]]
    signature04 = [module.__class__.__name__ for module in light04.model[:-1]]
    assert signature03 == signature04
    assert light03.model[-1].__class__.__name__ == "SegmentCitrusLite"
    assert light04.model[-1].__class__.__name__ == "SegmentCitrusQualityLite"


def test_light03_full_model_backward_and_complexity():
    """The deploy candidate must backpropagate and stay inside the intended budget."""
    model = YOLO(str(YAML_DIR / "Light03_deploy_lite.yaml"), task="segment", verbose=False).model.train()
    loss = _tensor_sum(model(torch.randn(2, 3, 128, 128)))
    loss.backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_200_000
    assert get_flops(model, imgsz=640) < 9.5


def test_joint_core_uses_current_lightweight_segment_branch():
    """Deep-only compression must not silently activate the heavier legacy head."""
    light01 = YOLO(str(YAML_DIR / "Light01_neck_only.yaml"), task="segment", verbose=False).model
    light02 = YOLO(str(YAML_DIR / "Light02_joint_core.yaml"), task="segment", verbose=False).model
    class_params_01 = sum(parameter.numel() for parameter in light01.model[-1].cv3.parameters())
    class_params_02 = sum(parameter.numel() for parameter in light02.model[-1].cv3.parameters())
    assert light02.model[-1].legacy is False
    assert class_params_02 == class_params_01
    assert get_flops(light02, imgsz=640) < get_flops(light01, imgsz=640)


def test_light04_quality_head_builds_and_calibrates_scores():
    """The PR candidate must expose a mask-quality branch in the standard YAML path."""
    model = YOLO(str(YAML_DIR / "Light04_quality_rank.yaml"), task="segment", verbose=False).model
    head = model.model[-1]
    assert head.__class__.__name__ == "SegmentCitrusQualityLite"
    model.train()
    output = model(torch.randn(2, 3, 128, 128))
    assert "mask_quality" in output


def test_light04_quality_loss_backward_reaches_quality_predictor():
    """Mask-IoU calibration must contribute a finite loss and train its predictor."""
    model = SegmentationModel(YAML_DIR / "Light04_quality_rank.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model, citrus_quality=0.25)
    total, components = model.loss(_synthetic_segmentation_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()
    gradient = model.model[-1].quality_predictor[0][-1].weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_light03_vfl_nwd_loss_backward_is_finite():
    """The isolated PR loss ablation must exercise both published loss paths."""
    model = SegmentationModel(YAML_DIR / "Light03_deploy_lite.yaml", ch=3, nc=1, verbose=False)
    _attach_loss_args(model, citrus_vfl=0.25, nwd_ratio=0.25)
    total, components = model.loss(_synthetic_segmentation_batch())
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    assert model.criterion.citrus_vfl == pytest.approx(0.25)
    assert model.criterion.bbox_loss.nwd_ratio == pytest.approx(0.25)
    total.sum().backward()
    class_gradients = [
        parameter.grad
        for parameter in model.model[-1].cv3.parameters()
        if parameter.grad is not None
    ]
    assert class_gradients and sum(gradient.abs().sum() for gradient in class_gradients) > 0


def test_joint_core_official_api_transfers_reindexed_segment_head():
    """YOLO(yaml).load(checkpoint) must honor the shortened-neck head mapping without the batch runner."""
    checkpoint = ROOT / "yolo11n-seg.pt"
    source = YOLO(str(checkpoint), task="segment", verbose=False).model.model[23].state_dict()
    target = YOLO(str(YAML_DIR / "Light02_joint_core.yaml"), task="segment", verbose=False)
    target.load(str(checkpoint))
    target_head = target.model.model[15].state_dict()
    key = "cv2.0.0.conv.weight"
    assert torch.equal(target_head[key], source[key])


@pytest.mark.parametrize("yaml_path", LIGHT_YAMLS, ids=lambda path: path.stem)
def test_every_light_yaml_accepts_official_checkpoint(yaml_path: Path):
    """Every Light YAML must remain usable via YOLO(yaml).load(yolo11n-seg.pt)."""
    model = YOLO(str(yaml_path), task="segment", verbose=False)
    model.load(str(ROOT / "yolo11n-seg.pt"))
    with torch.no_grad():
        output = model.model.eval()(torch.randn(1, 3, 128, 128))
    assert output is not None
