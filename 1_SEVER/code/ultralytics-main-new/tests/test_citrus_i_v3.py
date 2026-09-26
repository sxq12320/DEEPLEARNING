"""I V3 graph, speed budget, data-identity and differentiability contracts."""

import ast

import pytest
import torch
import yaml

from citrus_i_v3_suite import FACTORS, NAMES, ROOT, SUITES, YAML_DIR
from scripts.generate_citrus_i_v3_yaml import build_yaml
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def two_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(name):
    instance = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
    instance.args = IterableSimpleNamespace(
        **{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True, "nwd_ratio": FACTORS[name]["nwd"]}
    )
    return instance


@pytest.mark.parametrize("name", NAMES)
def test_yaml_build_forward_and_budget(name):
    saved = yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    assert saved == build_yaml(name)
    instance = model(name).eval()
    assert instance.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert sum(p.numel() for p in instance.parameters()) < 2_500_000
    assert 0 < get_flops(instance, 640) < 12
    with torch.no_grad():
        pred = instance(torch.rand(1, 3, 128, 160))
    assert pred[0][1].shape == (1, 32, 64, 80)
    assert pred[0][0].shape[-1] == 420


@pytest.mark.parametrize("name", ("I30_rgb_control", "I33_contrast_p23", "I36_nwd_p23", "I38_assign_p23"))
def test_backward(name):
    instance = model(name)
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    batch = dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        cls=torch.zeros(4, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.625, 0.625], [0.0703125, 0.0703125, 0.046875, 0.046875]] * 2),
        masks=masks,
    )
    loss, components = instance.loss(batch)
    assert components.numel() == 5 and torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in instance.parameters() if p.grad is not None)


def test_replay_control_and_source_indices():
    source = yaml.safe_load((ROOT / "0_orange_yaml/I_V2_series/I20_corrected_control.yaml").read_text())
    assert build_yaml("I30_rgb_control") == source
    target = build_yaml("I33_contrast_p23")
    assert target["pretrained_layer_map"][3] == 0
    assert target["pretrained_layer_map"][4] == 1
    assert target["pretrained_layer_map"][5] == 2
    assert target["pretrained_layer_map"][8] == -1  # new P2 gray->RGB fusion


def test_runner_python38_and_registry():
    from citrus_foreground import RUNNERS

    paths = (
        "RUN_CITRUS_I_V3.py", "20260923_citrus_i_v3_batch.py", "citrus_i_v3_suite.py",
        "scripts/generate_citrus_i_v3_yaml.py", "ultralytics/nn/modules/citrus_i_v3.py",
    )
    for path in paths:
        ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=8)
    assert set(RUNNERS["CITRUS_I_V3"].suites) == set(SUITES)


def test_runner_has_no_split_manifest_gate():
    from importlib import import_module

    runner = import_module("20260923_citrus_i_v3_batch")
    assert not hasattr(runner, "check_split_manifest")
