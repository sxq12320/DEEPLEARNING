"""V8 public entry, capacity preservation, input detail and real segmentation loss."""

from copy import deepcopy

import pytest
import torch
import yaml

from citrus_sage_v8_suite import NAMES, ROOT, YAML_DIR
from tests.test_citrus_sage_v4r import example_batch
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_sage_v8 import SAGEV8PhaseStem
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("empty", [False, True])
def test_public_model_loss_gradients_and_rectangular_inference(index, empty):
    path = YAML_DIR / f"{NAMES[index]}.yaml"
    api = YOLO(str(path), verbose=False)
    assert api.task == "segment"
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(example_batch(empty=empty))
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    head = model.model[-1]
    strides = [8, 16, 32] if index in (0, 4) else [4, 8, 16, 32] if index == 1 else [4, 8, 16]
    assert head.stride.tolist() == strides
    assert model.stride.max() == 32  # C5 context still needs /32 input padding
    assert items[-1] == 0
    if index in (1, 2, 3):
        assert len(head.cv2) == (3 if index == 1 else 2)
        for p in head.p2_cls.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
        if not empty:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.p2_cls.parameters())
    if index >= 3:
        assert model.model[0].shape_gain.grad is not None
    with torch.inference_mode():
        model.eval()
        output, raw = model(torch.rand(1, 3, 128, 160))
        anchors = sum((128 // s) * (160 // s) for s in strides)
        assert output[0].shape == (1, 37, anchors)
        assert output[1].shape == (1, 32, 32, 40)
        assert raw["scores"].shape[-1] == anchors
        model.model[-1].export = True
        assert model(torch.rand(1, 3, 160, 128))[1].shape[-2:] == (40, 32)
    assert 1.0 < get_flops(model, 640) < 11.0


@pytest.mark.parametrize("index", range(5))
def test_pretrained_keys_rebuild_save_load_fuse(index, tmp_path):
    api = YOLO(str(YAML_DIR / f"{NAMES[index]}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    for key, value in api.model.state_dict().items():
        if key.startswith(("model.23.cv2.0.", "model.23.cv2.1.", "model.0.conv.", "model.0.bn.")):
            torch.testing.assert_close(value, source[key], rtol=0, atol=0)
    rebuilt = SegmentationModel(api.model.yaml, nc=80, verbose=False)
    rebuilt.load(api.model, verbose=False)
    for key, value in api.model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    file = tmp_path / "new.pt"
    api.save(str(file))
    restored = YOLO(str(file), verbose=False).model.eval()
    x = torch.rand(1, 3, 128, 160)
    with torch.inference_mode():
        y = restored(x)[0]
        fused = deepcopy(restored).fuse(verbose=False)
        yf = fused(x)[0]
        torch.testing.assert_close(y[0], yf[0], rtol=0.001, atol=0.01)
        torch.testing.assert_close(y[1], yf[1], rtol=0.001, atol=0.01)


def test_control_identity_and_factor_isolation():
    cfg = [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    old = yaml.safe_load((ROOT / "0_orange_yaml/SAGE_V7_series/SAGE70_relay_control.yaml").read_text())
    assert cfg[0] == old
    assert all(c["head"][:-1] == old["head"][:-1] for c in cfg)
    assert cfg[2]["head"] == cfg[3]["head"]
    assert cfg[0]["head"] == cfg[4]["head"]
    assert cfg[3]["backbone"] == cfg[4]["backbone"]


def test_phase_brightness_offset_odd_size_and_zero_gain():
    stem = SAGEV8PhaseStem(3, 16).eval()
    x = torch.rand(2, 3, 31, 35)
    torch.testing.assert_close(stem.phase_detail(x), stem.phase_detail(x + 0.2), atol=1e-6, rtol=1e-5)
    assert stem(x).shape == (2, 16, 16, 18)
    stem.shape_gain.data.zero_()
    torch.testing.assert_close(stem(x), stem.act(stem.bn(stem.conv(x))), atol=0, rtol=0)


def test_foreground_inert_and_cache_locked():
    import RUN_SAGE_V8 as run
    from citrus_foreground import resolve_runner
    from citrus_protocol import fixed_train_args

    assert run.EPOCHS == 300 and not run.DRY_RUN and run.SUITE == "all"
    assert fixed_train_args()["cache"] is True and fixed_train_args()["amp"] is False
    _, spec = resolve_runner("SAGE_V8")
    assert (ROOT / spec.script).is_file()


def test_public_predict_pads_odd_rgb_image_for_retained_c5():
    import numpy as np

    model = YOLO(str(YAML_DIR / f"{NAMES[3]}.yaml"), verbose=False)
    results = model.predict(np.zeros((173, 319, 3), dtype=np.uint8), imgsz=160,
                            device="cpu", verbose=False, save=False)
    assert results[0].orig_shape == (173, 319)
    assert model.model.model[-1].stride.tolist() == [4, 8, 16]
