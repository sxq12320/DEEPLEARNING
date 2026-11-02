"""J7 native-YAML, loss, aliasing and inference contracts; CPU checks, not AP claims."""

import ast
import copy
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import J7DeepStage, J7ScaleHub, SegmentCitrusJ7
from ultralytics.nn.modules.citrus_e_v12 import EV12RecognitionCorrection
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_j7_loss import J7SegmentationLoss
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA

ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / "0_orange_yaml" / "J_V7_series"
YAMLS = tuple(sorted(YAML_DIR.glob("J1*.yaml")))


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def model_for(number):
    path, = YAML_DIR.glob("J%d_*.yaml" % number)
    model = YOLO(str(path), task="segment", verbose=False).model
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    return model


def make_batch(empty=False):
    """An ordinary overlap-mask batch with one large and one tiny visible instance per image."""
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    # Centred near the stride-8 anchor at (20,20), so the tiny term is exercised
    # rather than merely testing a no-positive-anchor zero fallback.
    masks[:, 9:12, 9:12] = 2
    count = 0 if empty else 4
    return dict(
        img=torch.rand(2, 3, 128, 128, requires_grad=True),
        batch_idx=torch.tensor([0., 0., 1., 1.])[:count],
        cls=torch.zeros(count, 1),
        masks=masks * 0 if empty else masks,
        bboxes=torch.tensor([[.5, .5, .625, .625], [.1640625, .1640625, .046875, .046875]] * 2)[:count],
    )


@pytest.mark.parametrize("path", YAMLS, ids=lambda path: path.stem)
def test_all_ten_official_yaml_forward_backward(path):
    model = YOLO(str(path), task="segment", verbose=False).model
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    model.train()
    batch = make_batch()
    predictions = model(batch["img"])
    loss, components = model.loss(batch, predictions)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    assert batch["img"].grad is not None and torch.isfinite(batch["img"].grad).all()
    assert model.model[0].conv.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model[-1].stride.tolist() == [8., 16., 32.]
    for module in model.modules():
        if isinstance(module, (J7DeepStage, J7ScaleHub)):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    if isinstance(model.model[-1], SegmentCitrusJ7):
        assert "ev3_quality" not in predictions
        assert not hasattr(model.model[-1], "quality_predictor")
        assert model.model[-1].recognition_route == 0
        assert not any(isinstance(module, EV12RecognitionCorrection) for module in model.modules())
    model.eval()
    with torch.no_grad():
        prediction = model(torch.rand(1, 3, 128, 160))[0]
    assert prediction[0].shape[-1] == 420
    stride = 4 if path.stem.startswith(("J100_", "J108_")) else 2
    assert prediction[1].shape[-2:] == (128 // stride, 160 // stride)


@pytest.mark.parametrize("number", [102, 103, 104, 109])
def test_empty_annotation_batch_is_finite(number):
    model = model_for(number).train()
    loss, components = model.loss(make_batch(empty=True))
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("number", [102, 103, 104, 105, 106, 107, 108])
def test_zero_gain_arms_use_exact_standard_criterion(number):
    model = model_for(number)
    assert type(model.init_criterion()) is v8SegmentationLoss
    head = model.model[-1]
    assert head.assignment_mix == head.ring_gain == head.quality_gain == 0
    assert head.tiny_dice_gain == head.boundary_gain == head.neighbor_gain == 0


def test_objective_arm_reuses_only_declared_tiny_boundary_terms():
    model = model_for(109).train()
    assert isinstance(model.init_criterion(), J7SegmentationLoss)
    loss, _ = model.loss(make_batch())
    loss.sum().backward()
    criterion = model.criterion
    assert criterion.tiny_gain == .25 and criterion.boundary_gain == .5
    assert criterion.quality_gain == criterion.negative_gain == criterion.ring_gain == 0
    assert criterion.last_quality == 0
    assert criterion.last_tiny > 0
    assert criterion.last_geometry[0] > 0


@pytest.mark.parametrize("number,width", [(105, 48), (106, 32)])
def test_shared_weight_is_not_shared_bn_and_optimizer_updates_once(number, width):
    model = model_for(number)
    head = model.model[-1]
    assert head.shared_towers and head.tower_width == width
    for towers in (head.cv2, head.cv3, head.cv4):
        assert towers[0][1][0].weight is towers[2][1][0].weight
        assert towers[0][0].bn is not towers[1][0].bn
        assert towers[0][-1] is not towers[1][-1]
    optimizer = BaseTrainer.build_optimizer(object(), model, name="AdamW", lr=.001, decay=.0005)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    shared = head.cv2[0][1][0].weight
    shared.grad = torch.ones_like(shared)
    optimizer.step()
    assert optimizer.state[shared]["step"] == 1


def test_ema_updates_alias_once():
    model = model_for(105)
    ema = ModelEMA(model)
    ema.decay = lambda _: .5
    for parameter in ema.ema.parameters():
        parameter.zero_()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(2)
    ema.update(model)
    for towers in (ema.ema.model[-1].cv2, ema.ema.model[-1].cv3, ema.ema.model[-1].cv4):
        assert torch.equal(towers[0][1][0].weight, torch.ones_like(towers[0][1][0].weight))
        assert towers[0][1][0].weight is towers[2][1][0].weight


@pytest.mark.parametrize("number", [104, 105])
def test_fuse_and_state_dict_reload_preserve_predictions(number):
    model = model_for(number).eval()
    image = torch.rand(1, 3, 128, 160)
    restored = model_for(number).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        reference = model(image)[0]
        reloaded = restored(image)[0]
        fused_model = copy.deepcopy(model).fuse(verbose=False)
        fused = fused_model(image)[0]
    for a, b, c in zip(reference, reloaded, fused):
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, c, atol=1e-4, rtol=1e-4)
    if number == 105:
        assert fused_model.model[-1].cv2[0][1][0].weight is fused_model.model[-1].cv2[2][1][0].weight


def test_official_checkpoint_save_load_keeps_custom_graph_and_aliases(tmp_path):
    path, = YAML_DIR.glob("J105_*.yaml")
    wrapper = YOLO(str(path), task="segment", verbose=False)
    checkpoint = tmp_path / "j7_contract.pt"
    wrapper.save(str(checkpoint))
    restored = YOLO(str(checkpoint), task="segment", verbose=False).model.eval()
    head = restored.model[-1]
    assert isinstance(head, SegmentCitrusJ7) and not hasattr(head, "quality_predictor")
    assert head.cv2[0][1][0].weight is head.cv2[2][1][0].weight
    # Official save rounds the checkpoint to FP16, then load converts to FP32.
    rounded = copy.deepcopy(wrapper.model).half().float().eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        reference, reloaded = rounded(image)[0], restored(image)[0]
    for expected, actual in zip(reference, reloaded):
        torch.testing.assert_close(expected, actual)


def test_repeats_are_inside_stage_and_spatial_return_uses_semantic_n3():
    model = model_for(104).eval()
    stages = [module for module in model.model if isinstance(module, J7DeepStage)]
    assert len(stages) == 2 and all(len(stage.blocks) == 1 for stage in stages)
    assert len(J7DeepStage(16, 32, n=3).blocks) == 3
    hub, = (module for module in model.model if isinstance(module, J7ScaleHub))
    observed = {}
    hook = hub.return3.register_forward_pre_hook(lambda _, values: observed.update(return_input=values[0]))
    values = [torch.rand(1, 128, 16, 20), torch.rand(1, 128, 8, 10), torch.rand(1, 256, 4, 5)]
    with torch.no_grad():
        outputs = hub(values)
    hook.remove()
    assert observed["return_input"] is outputs[0]
    assert [value.shape[1] for value in outputs] == [64, 128, 256]


def test_all_ablation_switches_are_actual_graph_changes():
    assert model_for(104).model[-1].persistent is True
    assert model_for(107).model[-1].persistent is False
    coarse = model_for(108).model[-1]
    assert coarse.fine_mask is False and coarse.proto_stride == 4
    assert not hasattr(coarse, "fine_detail")
    assert not hasattr(coarse, "fine_to_proto")
    assert not hasattr(coarse, "fine_scale")
    assert len(YAMLS) == 10


def test_new_sources_parse_with_python38_grammar():
    for relative in ("ultralytics/nn/modules/citrus_j7.py", "ultralytics/utils/citrus_j7_loss.py"):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"), feature_version=(3, 8))
