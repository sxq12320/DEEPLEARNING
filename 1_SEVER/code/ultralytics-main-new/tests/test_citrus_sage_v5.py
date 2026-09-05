"""Public API, independent structure/loss factors, actual detection-path gradients."""

from copy import deepcopy

import pytest
import torch
import yaml

from citrus_sage_v5_suite import NAMES, SUITES, YAML_DIR
from tests.test_citrus_sage_v4r import example_batch
from ultralytics import YOLO
from ultralytics.nn.modules import SegmentCitrusSAGEV5
from ultralytics.nn.modules.citrus_sage_v5 import SAGELateProto
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", NAMES[2:])
def test_yaml_build_forward_backward_flops_and_export(name):
    api = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment", verbose=False)
    model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    head = model.model[-1]
    assert isinstance(head, SegmentCitrusSAGEV5)
    loss, items = model.loss(example_batch())
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    for key, parameter in head.named_parameters():
        if any(s in key for s in ("refiner", "detail_", "relay_scale")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), key
    assert (items[-1] > 0) == (head.boundary_gain + head.neighbor_gain > 0)
    if head.relay_enabled:
        assert head.detail_relay.conv.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 128, 160))
        assert output[0][1].shape == (1, 32, 32, 40)
        head.export = True
        assert model(torch.rand(1, 3, 128, 160))[1].shape == (1, 32, 32, 40)
        head.export = False
    assert sum(p.numel() for p in model.parameters()) < 2_350_000
    assert get_flops(model, 640) < (8.0 if head.late_proto else 10.2)
    assert api.model is not None


def test_late_proto_spatial_mix_is_really_before_upsampling():
    proto = SAGELateProto(64, 64, 32)
    shapes = []
    hook = proto.cv2.register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape[-2:]))
    assert proto(torch.rand(2, 64, 20, 24)).shape == (2, 32, 40, 48)
    hook.remove()
    assert shapes == [(20, 24)]


def test_candidate_loss_reaches_detail_branch_without_mask_loss():
    head = SegmentCitrusSAGEV5(nc=1, npr=64, relay=True, late_proto=True, ch=(64, 128, 256, 64))
    features = [torch.rand(2, c, h, w) for c, h, w in [(64, 16, 20), (128, 8, 10), (256, 4, 5), (64, 32, 40)]]
    original = [x.clone() for x in features]
    outputs = head(features)
    outputs["scores"].square().mean().backward()
    assert head.refiner.detail[0].conv.weight.grad.abs().sum() > 0
    assert head.detail_relay.conv.weight.grad.abs().sum() > 0
    assert head.detail_to_proto.weight.grad is None  # This test has NO prototype loss.
    for before, after in zip(original, features):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_zero_relay_scale_restores_the_no_relay_forward():
    head = SegmentCitrusSAGEV5(nc=1, npr=64, relay=True, late_proto=True, ch=(64, 128, 256, 64)).train()
    head.relay_scale.data.zero_()
    control = deepcopy(head)
    control.relay_enabled = False
    features = [torch.rand(2, c, h, w) for c, h, w in [(64, 16, 20), (128, 8, 10), (256, 4, 5), (64, 32, 40)]]
    a, b = head(features), control(features)
    for key in ("scores", "boxes", "proto"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_factorial_geometry_and_optional_backbone_are_isolated():
    configs = {name: yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text()) for name in NAMES}
    anchor = configs[NAMES[1]]
    for name in NAMES[2:8]:
        assert configs[name]["backbone"] == anchor["backbone"]
        assert configs[name]["head"][:-1] == anchor["head"][:-1]
    geometry = [configs[n]["head"][-1][-1] for n in SUITES["geometry"]]
    assert {tuple(a[-2:]) for a in geometry} == {(0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1)}
    assert all(a[:-2] == geometry[0][:-2] for a in geometry)
    assert configs[NAMES[-1]]["head"] == configs[NAMES[4]]["head"]
    changed = [i for i, (a, b) in enumerate(zip(configs[NAMES[-1]]["backbone"], anchor["backbone"])) if a != b]
    assert changed == [8]


@pytest.mark.parametrize("empty", [True, False])
def test_geometry_empty_and_nonempty_finite(empty):
    model = SegmentationModel(YAML_DIR / f"{NAMES[7]}.yaml", nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(example_batch(empty=empty))
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert (items[-1] == 0) == empty


def test_pretrained_shared_keys_and_checkpoint_reload(tmp_path):
    checkpoint = YAML_DIR.parents[1] / "yolo11n-seg.pt"
    if not checkpoint.exists():
        pytest.skip("Local official initialization unavailable")
    source = YOLO(str(checkpoint), verbose=False)
    api = YOLO(str(YAML_DIR / f"{NAMES[4]}.yaml"), verbose=False).load(str(checkpoint))
    for cv in ("cv1", "cv2", "cv3"):
        torch.testing.assert_close(
            getattr(api.model.model[-1].proto, cv).conv.weight, getattr(source.model.model[-1].proto, cv).conv.weight
        )
    target = tmp_path / "v5.pt"
    api.save(str(target))
    restored = YOLO(str(target), verbose=False)
    with torch.no_grad():
        assert restored.model.eval()(torch.rand(1, 3, 128, 160))[0][1].shape == (1, 32, 32, 40)
