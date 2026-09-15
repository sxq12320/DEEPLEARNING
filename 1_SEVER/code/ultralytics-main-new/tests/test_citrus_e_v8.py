"""Official YAML entry, state transfer, geometry, finite training and lightweight graph contracts."""

from copy import deepcopy
import importlib

import pytest
import torch
import yaml

from citrus_e_v8_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, YAML_DIR
from scripts.generate_citrus_e_v8_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_e_v8 import EV8ContextStage, EV8P4Reconcile
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(empty=False):
    masks = torch.zeros(2, 64, 64)
    masks[0, 6:54, 6:54] = 1
    masks[1, 10:58, 10:58] = 1
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 1.0])[: 0 if empty else 2],
        cls=torch.zeros(0 if empty else 2, 1),
        bboxes=torch.tensor([[0.46875, 0.46875, 0.75, 0.75], [0.53125, 0.53125, 0.75, 0.75]])[: 0 if empty else 2],
        masks=masks * (not empty),
    )


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
@pytest.mark.parametrize("empty", [False, True])
def test_build_train_infer(name, factors, empty):
    api = YOLO(str(YAML_DIR / f"{name}.yaml"), verbose=False)
    assert api.task == "segment"
    model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
    args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    args.mask_ratio, args.overlap_mask, args.nwd_ratio = 2, True, 0.0
    model.args = args
    loss, items = model.loss(batch(empty))
    assert torch.isfinite(loss).all() and len(items) == 5
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not empty:
        if factors[0]:
            assert model.model[6].blocks[0].local.conv.weight.grad.abs().sum() > 0
        if factors[1]:
            assert model.model[19].gate.weight.grad.abs().sum() > 0
        if factors[2]:
            assert torch.isfinite(model.criterion.last_geometry).all()
    assert model.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert not model.model[-1].deform_detail and not model.model[-1].pmce
    assert 0 < get_flops(model, 640) < 11
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1, 3, 128, 160)
        out = model(x)
        assert out[0][1].shape[-2:] == (64, 80)
        assert out[0][0].shape[-1] == 420
        fused = deepcopy(model).fuse(verbose=False)(x)
        torch.testing.assert_close(out[0][0], fused[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(out[0][1], fused[0][1], rtol=0.002, atol=0.02)


def test_exact_control():
    old = SegmentationModel(ROOT / "0_orange_yaml/E_V7_series/V7_00_control.yaml", nc=1, verbose=False).eval()
    new = SegmentationModel(YAML_DIR / f"{NAMES[0]}.yaml", nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
def test_transfer_and_reload(name, factors, tmp_path):
    model = YOLO(str(YAML_DIR / f"{name}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    src = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    for key in (
        "model.2.cv1.conv.weight",
        "model.4.cv2.conv.weight",
        "model.13.cv1.conv.weight",
        "model.16.cv1.conv.weight",
        "model.23.cv4.0.0.conv.weight",
    ):
        torch.testing.assert_close(model.model.state_dict()[key], src[key], rtol=0, atol=0)
    assert isinstance(model.model.model[6], EV8ContextStage) == bool(factors[0])
    assert isinstance(model.model.model[19], EV8P4Reconcile) == bool(factors[1])
    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = YOLO(str(path), verbose=False)
    assert loaded.model.yaml == model.model.yaml


def test_manifest_and_runner():
    assert configs() == [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    runner = importlib.import_module("20260914_citrus_e_v8_batch")
    assert runner.NAMES == NAMES and runner.RUN_OVERRIDES == RUN_OVERRIDES
    entry = importlib.import_module("RUN_CITRUS_E_V8")
    assert entry.EPOCHS == 300 and not entry.DRY_RUN
    source = (ROOT / "RUN_CITRUS_E_V8.py").read_text()
    assert "cache=True" in source and "amp=False" in source and "device_lock=False" in source
    for n, f in zip(NAMES, FACTORS):
        assert RUN_OVERRIDES[n]["copy_paste"] == (0.3 if f[3] else 0.0)


def test_neck_zero_gain_is_anchor_and_source_gradients():
    m = EV8P4Reconcile([128, 128, 128], 128).eval()
    inputs = [
        torch.rand(2, 128, 8, 10, requires_grad=True),
        torch.rand(2, 128, 8, 10, requires_grad=True),
        torch.rand(2, 128, 16, 20, requires_grad=True),
    ]
    with torch.no_grad():
        m.gain.zero_()
        torch.testing.assert_close(m(inputs), inputs[0], rtol=0, atol=0)
        m.gain.fill_(0.1)
    m(inputs).square().mean().backward()
    assert all(x.grad.abs().sum() > 0 for x in inputs)
