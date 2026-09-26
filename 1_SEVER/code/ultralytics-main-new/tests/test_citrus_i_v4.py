"""I V4 official YAML API, true loss, layer targets, and experiment contracts."""
import ast

import pytest
import torch
import yaml

from citrus_i_v4_suite import NAMES, ROOT, SUITES, YAML_DIR
from scripts.generate_citrus_i_v4_yaml import build_yaml
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_i_v4 import LayeredSceneNeck
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_i_v4_loss import balanced_layer_bce, layer_targets
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def model(name):
    net = YOLO(YAML_DIR / f"{name}.yaml", task="segment").model
    net.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    return net


@pytest.mark.parametrize("name", NAMES)
def test_build_forward_flops(name):
    assert yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text()) == build_yaml(name)
    net = model(name).eval()
    assert net.model[-1].stride.tolist() == [8, 16, 32]
    assert sum(p.numel() for p in net.parameters()) < 2_500_000
    assert 0 < get_flops(net, 640) < 12
    with torch.no_grad():
        result = net(torch.rand(1, 3, 128, 160))
    assert result[0][0].shape == (1, 37, 420)
    assert result[0][1].shape == (1, 32, 64, 80)


@pytest.mark.parametrize("name", NAMES)
def test_real_loss_backward(name):
    net = model(name).train()
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    batch = dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0., 0., 1., 1.]),
                 cls=torch.zeros(4, 1), masks=masks,
                 bboxes=torch.tensor([[.5, .5, .625, .625], [.0703125, .0703125, .046875, .046875]] * 2))
    loss, parts = net.loss(batch)
    assert torch.isfinite(loss).all() and parts.numel() == 5
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if name in ("I42_discovery_aux", "I43_region_feedback", "I44_boundary_protected", "I45_two_step"):
        assert net.model[-1].scene_neck.readout.weight.grad.abs().sum() > 0


def test_targets_keep_tiny_and_touching_interface_without_filling():
    masks = torch.zeros(1, 16, 16)
    masks[0, 2:8, 2:6] = 1
    masks[0, 2:8, 6:10] = 2
    masks[0, 14, 14] = 3
    source = masks.clone()
    targets = layer_targets(masks, torch.zeros(3), 1, True, (16, 16))
    assert targets[0, 1, 4, 5] == 1 and targets[0, 1, 4, 6] == 1
    assert targets[0, 0, 10, 10] == 0
    assert layer_targets(masks, torch.zeros(3), 1, True, (4, 4))[0, 0, 3, 3] == 1
    assert torch.equal(masks, source)


@pytest.mark.parametrize("overlap", [False, True])
def test_empty_layer_targets_and_loss(overlap):
    masks = torch.zeros(2 if overlap else 0, 16, 16)
    targets = layer_targets(masks, torch.empty(0), 2, overlap, (8, 8))
    assert targets.shape == (2, 2, 8, 8) and not targets.any()
    logits = torch.zeros_like(targets, requires_grad=True)
    balanced_layer_bce(logits, targets).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.sum() > 0


def test_extreme_probabilities_do_not_produce_nan():
    state = torch.rand(2, 16, 8, 8, requires_grad=True)
    for value in (0., 1.):
        difference = LayeredSceneNeck.region_difference(state, torch.full((2, 1, 8, 8), value))
        assert torch.isfinite(difference).all()
    difference.sum().backward()
    assert torch.isfinite(state.grad).all()


def test_empty_image_actual_loss_still_supervises_discovery():
    net = model("I44_boundary_protected").train()
    batch = dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.empty(0), cls=torch.empty(0, 1),
                 masks=torch.zeros(2, 64, 64), bboxes=torch.empty(0, 4))
    loss, _ = net.loss(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert net.model[-1].scene_neck.readout.weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_and_640_forward(tmp_path):
    wrapper = YOLO(YAML_DIR / "I44_boundary_protected.yaml", task="segment")
    destination = tmp_path / "iv4.pt"
    wrapper.save(destination)
    restored = YOLO(destination, task="segment").model.eval()
    with torch.no_grad():
        output = restored(torch.rand(1, 3, 640, 640))
    assert output[0][0].shape == (1, 37, 8400)
    assert output[0][1].shape == (1, 32, 320, 320)


def test_parallel_graph_and_replay():
    from scripts.generate_citrus_i_v3_yaml import build_yaml as v3_yaml
    source = v3_yaml("I30_rgb_control")
    source["nc"] = 1
    assert source == build_yaml("I40_replay")
    assert len(build_yaml("I44_boundary_protected")["head"]) == 1
    assert len(build_yaml("I46_achromatic")["backbone"]) == 23


def test_runner_registration_and_python38_syntax():
    from citrus_foreground import RUNNERS
    assert set(RUNNERS["CITRUS_I_V4"].suites) == set(SUITES)
    for path in ("RUN_CITRUS_I_V4.py", "20260925_citrus_i_v4_batch.py", "citrus_i_v4_suite.py",
                 "ultralytics/nn/modules/citrus_i_v4.py", "ultralytics/utils/citrus_i_v4_loss.py"):
        ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=8)
