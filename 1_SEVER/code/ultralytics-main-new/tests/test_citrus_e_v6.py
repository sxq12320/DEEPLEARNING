"""E V6 contracts: official API, backward, stride-2 proto path, no inference GT."""

import importlib.util
from copy import deepcopy

import pytest
import torch
import yaml

from citrus_e_v6_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, YAML_DIR
from scripts.generate_citrus_e_v6_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(empty=False, mask_ratio=4):
    size = 128 // mask_ratio
    masks = torch.zeros(2, size, size)
    masks[0, 28 // mask_ratio:92 // mask_ratio, 28 // mask_ratio:92 // mask_ratio] = 1
    masks[1, 36 // mask_ratio:100 // mask_ratio, 36 // mask_ratio:100 // mask_ratio] = 1
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 1.0])[: 0 if empty else 2],
        cls=torch.zeros(0 if empty else 2, 1),
        bboxes=torch.tensor([[0.46875, 0.46875, 0.5, 0.5], [0.53125, 0.53125, 0.5, 0.5]])[: 0 if empty else 2],
        masks=masks * (not empty),
    )


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
@pytest.mark.parametrize("empty", [False, True])
def test_official_api_backward_and_inference(name, factors, empty):
    assert YOLO(str(YAML_DIR / (name + ".yaml")), verbose=False).task == "segment"
    model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False)
    args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    args.overlap_mask = True
    for key, value in RUN_OVERRIDES[name].items():
        setattr(args, key, value)
    model.args = args
    loss, items = model.loss(batch(empty, args.mask_ratio))
    from ultralytics.utils.citrus_e_v6_loss import EV6SegmentationLoss

    assert isinstance(model.criterion, EV6SegmentationLoss)
    assert model.criterion.bbox_loss.nwd_ratio == RUN_OVERRIDES[name]["nwd_ratio"]
    assert torch.isfinite(loss).all() and len(items) == 5
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    head = model.model[-1]
    if factors[0] and not empty:
        assert head.fine_scale.grad.abs().sum() > 0
        assert head.fine_to_proto.weight.grad.abs().sum() > 0
    assert get_flops(model, 640) < 10.4
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1, 3, 128, 160)
        out = model(x)
        assert "ev5_region_logits" not in out[1]
        fused = deepcopy(model).fuse(verbose=False)(x)
        torch.testing.assert_close(out[0][0], fused[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(out[0][1], fused[0][1], rtol=0.002, atol=0.02)


def test_control_exact_parity_with_v5_control():
    old = SegmentationModel(
        ROOT / "0_orange_yaml/E_V5_series/V5_00_control.yaml", nc=1, verbose=False
    ).eval()
    new = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
def test_proto_stride_and_resolution(name, factors):
    model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False).eval()
    head = model.model[-1]
    assert head.proto_stride == (2 if factors[0] else 4)
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 256, 256))
    proto = out[0][1]
    expected = 256 // (2 if factors[0] else 4)
    assert proto.shape[-2] == expected and proto.shape[-1] == expected


def test_fine_mask_leaves_candidate_path_unchanged():
    control = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False).eval()
    fine = SegmentationModel(YAML_DIR / (NAMES[2] + ".yaml"), nc=1, verbose=False).eval()
    fine.load_state_dict(control.state_dict(), strict=False)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 160)
        a, b = control(x), fine(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        assert a[0][1].shape[-1] * 2 == b[0][1].shape[-1]


@pytest.mark.parametrize("name", NAMES)
def test_pretraining_and_checkpoint(name, tmp_path):
    model = YOLO(str(YAML_DIR / (name + ".yaml")), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    for key in ("model.2.cv1.conv.weight", "model.4.cv2.conv.weight", "model.23.cv4.0.0.conv.weight"):
        torch.testing.assert_close(model.model.state_dict()[key], source[key], rtol=0, atol=0)
    path = tmp_path / "model.pt"
    model.save(str(path))
    restored = YOLO(str(path), verbose=False)
    assert type(restored.model.model[-1]) is type(model.model.model[-1])
    assert restored.model.model[-1].fine_mask == model.model.model[-1].fine_mask
    assert restored.model.model[-1].proto_stride == model.model.model[-1].proto_stride


def test_overrides_and_manifest_consistency():
    assert configs() == [yaml.safe_load((YAML_DIR / (n + ".yaml")).read_text()) for n in NAMES]
    for name, factors in zip(NAMES, FACTORS):
        ov = RUN_OVERRIDES[name]
        assert ov["mask_ratio"] == (2 if factors[0] or name == "V6_01_mr2" else 4)
        assert ov["nwd_ratio"] == (0.5 if factors[1] else 0.0)
        assert ov["copy_paste"] == (0.3 if factors[2] else 0.0)
    spec = importlib.util.spec_from_file_location("ev6_runner_test", ROOT / "20260911_citrus_e_v6_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.NAMES == NAMES and runner.RUN_OVERRIDES == RUN_OVERRIDES
    source = (ROOT / "RUN_CITRUS_E_V6.py").read_text()
    assert "cache=True" in source and "amp=False" in source and "device_lock=False" in source


@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("empty", [False, True])
def test_common_validation_raster(stride, empty, monkeypatch, tmp_path):
    from citrus_e_v6_training import EV6Validator
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops

    validator = EV6Validator(save_dir=tmp_path, args={"mask_ratio": 4})
    validator.proto_stride = stride
    validator.input_shape = (128, 160)
    # Simulate an opaque final-evaluation backend with no discoverable head.
    validator.proto_stride = 4
    validator.process = ops.process_mask
    n = 0 if empty else 1
    raw = [{"extra": torch.ones(n, 2), "bboxes": torch.tensor([[10., 10., 70., 70.]])[:n]}]
    monkeypatch.setattr(DetectionValidator, "postprocess", lambda *_: raw)
    proto = torch.ones(1, 2, 128 // stride, 160 // stride)
    result = validator.postprocess((torch.empty(0), proto))
    assert result[0]["masks"].shape == (n, 64, 80)
    assert validator.args.mask_ratio == 2
    if n:
        # Input-space box is mapped to the common raster once, not twice.
        assert result[0]["masks"][0, 10, 10] == 1
        assert result[0]["masks"][0, 40, 40] == 0


def test_quality_uses_supervision_grid_without_mutation(monkeypatch):
    from ultralytics.utils.citrus_e_v3_loss import EV3SegmentationLoss
    from ultralytics.utils.citrus_e_v6_loss import EV6SegmentationLoss

    criterion = object.__new__(EV6SegmentationLoss)
    original = torch.rand(2, 32, 32, 32, requires_grad=True)
    preds = {"proto": original}
    captured = {}

    def inspect(self, predictions, data):
        captured.update(predictions)
        return torch.tensor(0.)

    monkeypatch.setattr(EV3SegmentationLoss, "_quality_loss", inspect)
    criterion._quality_loss(preds, {"masks": torch.ones(2, 64, 64)})
    assert captured["proto"].shape[-2:] == (64, 64)
    assert not captured["proto"].requires_grad
    assert preds["proto"] is original


def test_native_validator_fine_backend_geometry(monkeypatch, tmp_path):
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils import ops

    validator = SegmentationValidator(save_dir=tmp_path)
    validator.proto_stride = 4  # Backend does not expose the real stride-2 head.
    validator.input_shape = (128, 160)
    validator.process = ops.process_mask
    raw = [{"extra": torch.ones(1, 2), "bboxes": torch.tensor([[10., 10., 70., 70.]])}]
    monkeypatch.setattr(DetectionValidator, "postprocess", lambda *_: raw)
    result = validator.postprocess((torch.empty(0), torch.ones(1, 2, 64, 80)))
    assert validator.proto_stride == 2
    assert result[0]["masks"].shape == (1, 64, 80)
