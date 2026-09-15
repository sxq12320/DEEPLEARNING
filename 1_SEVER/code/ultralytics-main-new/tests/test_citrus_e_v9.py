"""V9 official YAML, loss isolation, provenance and deployment-shaped contracts."""

import importlib
from copy import deepcopy

import pytest
import torch
import yaml

from citrus_e_v9_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, YAML_DIR
from scripts.generate_citrus_e_v9_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_e_v9_loss import EV9SegmentationLoss, outside_gt_boxes, tiny_instance_dice
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def batch(empty=False):
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 0.0, 1.0, 1.0])[: 0 if empty else 4],
        cls=torch.zeros(0 if empty else 4, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.625, 0.625], [0.0703125, 0.0703125, 0.046875, 0.046875]] * 2)[
            : 0 if empty else 4
        ],
        masks=masks * (not empty),
    )


@pytest.mark.parametrize("i", range(10))
@pytest.mark.parametrize("empty", [False, True])
def test_build_forward_backward(i, empty):
    api = YOLO(str(YAML_DIR / f"{NAMES[i]}.yaml"), verbose=False)
    assert api.task == "segment"
    model = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False)
    args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    args.mask_ratio, args.overlap_mask, args.nwd_ratio = 2, True, 0.0
    model.args = args
    total, items = model.loss(batch(empty))
    assert isinstance(model.criterion, EV9SegmentationLoss)
    assert len(items) == 5 and torch.isfinite(total).all()
    total.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not empty and FACTORS[i][0]:
        assert model.model[-1].mask_fusion.gate.weight.grad.abs().sum() > 0
    assert model.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1, 3, 128, 160)
        result = model(x)
        assert result[0][1].shape == (1, 32, 64, 80)
        assert result[0][0].shape[-1] == 420
        fused = deepcopy(model).fuse(verbose=False)(x)
        torch.testing.assert_close(result[0][0], fused[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(result[0][1], fused[0][1], rtol=0.002, atol=0.02)
    assert 0 < get_flops(model, 640) < 11


@pytest.mark.parametrize("i", [0, 1])
def test_exact_legacy_controls(i):
    old_name = ["V8_00_phase_control", "V8_03_geometry_cp"][i]
    old = SegmentationModel(ROOT / f"0_orange_yaml/E_V8_series/{old_name}.yaml", nc=1, verbose=False).eval()
    new = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("i", range(10))
def test_pretrain_reload(i, tmp_path):
    api = YOLO(str(YAML_DIR / f"{NAMES[i]}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    for key in (
        "model.6.cv1.conv.weight",
        "model.13.cv1.conv.weight",
        "model.19.cv1.conv.weight",
        "model.23.cv4.0.0.conv.weight",
    ):
        torch.testing.assert_close(source[key], api.model.state_dict()[key], rtol=0, atol=0)
    saved = tmp_path / "model.pt"
    api.save(str(saved))
    loaded = YOLO(str(saved), verbose=False)
    assert loaded.model.yaml == api.model.yaml


def test_mask_neck_cannot_change_detector_at_fixed_weights():
    model = SegmentationModel(YAML_DIR / f"{NAMES[2]}.yaml", nc=1, verbose=False).eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        a = model(x)
        model.model[-1].mask_fusion.gain.fill_(1)
        b = model(x)
    torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
    assert not torch.equal(a[0][1], b[0][1])


def test_negative_excludes_unassigned_gt_and_margin():
    centers = torch.tensor([[20.0, 20.0], [29.0, 29.0], [42.0, 42.0]])
    boxes = torch.tensor([[18.0, 18.0, 22.0, 22.0]])
    assert outside_gt_boxes(centers, boxes).tolist() == [False, False, True]
    assert outside_gt_boxes(centers, boxes[:0]).all()


def test_tiny_loss_gt_balance_empty_and_gradient():
    proto = torch.randn(2, 16, 16, requires_grad=True)
    coeff = torch.randn(2, 2, requires_grad=True)
    gt = torch.zeros(1, 16, 16)
    gt[:, 4:7, 4:7] = 1
    boxes = torch.tensor([[4.0, 4.0, 7.0, 7.0]] * 2)
    a, n = tiny_instance_dice(coeff, proto, gt, torch.tensor([0, 0]), boxes, (2, 2))
    b, n2 = tiny_instance_dice(coeff.repeat(2, 1), proto, gt, torch.tensor([0, 0, 0, 0]), boxes.repeat(2, 1), (2, 2))
    torch.testing.assert_close(a, b)
    assert n == n2 == 1
    a.backward()
    assert coeff.grad.abs().sum() > 0 and torch.isfinite(proto.grad).all()
    zero, count = tiny_instance_dice(coeff, proto, gt * 0, torch.tensor([0, 0]), boxes, (2, 2))
    assert zero == count == 0


def test_manifest_runner_and_light_budget():
    assert configs() == [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    runner = importlib.import_module("20260915_citrus_e_v9_batch")
    entry = importlib.import_module("RUN_CITRUS_E_V9")
    assert runner.NAMES == NAMES and entry.EPOCHS == 300 and not entry.DRY_RUN
    assert RUN_OVERRIDES[NAMES[9]] == {**RUN_OVERRIDES[NAMES[8]], "cos_lr": True}
    full = SegmentationModel(YAML_DIR / f"{NAMES[8]}.yaml", nc=1, verbose=False)
    base = SegmentationModel(YAML_DIR / f"{NAMES[1]}.yaml", nc=1, verbose=False)
    assert get_flops(full, 640) < get_flops(base, 640) * 0.85
    assert sum(p.numel() for p in full.parameters()) < sum(p.numel() for p in base.parameters())
