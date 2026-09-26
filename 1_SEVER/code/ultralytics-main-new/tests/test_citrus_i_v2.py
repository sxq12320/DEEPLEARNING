"""I V2 official YAML, P2 candidate, backbone, loss, runner and compatibility contracts."""

import ast
import importlib

import pytest
import torch

from citrus_foreground import RUNNERS
from citrus_i_v2_suite import FACTORS, NAMES, ROOT, SUITES, YAML_DIR
from scripts.generate_citrus_i_v2_yaml import build_yaml
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_e_v11_loss import EV11SegmentationLoss
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(name):
    instance = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
    instance.args = IterableSimpleNamespace(
        **{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True, "nwd_ratio": 0.0}
    )
    return instance


def batch():
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        cls=torch.zeros(4, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.625, 0.625], [0.0703125, 0.0703125, 0.046875, 0.046875]] * 2),
        masks=masks,
    )


@pytest.mark.parametrize("name", NAMES)
def test_yaml_build_forward_and_budget(name):
    import yaml

    assert yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text(encoding="utf-8")) == build_yaml(name)
    instance = model(name).eval()
    expected = [8.0, 16.0, 32.0, 4.0] if FACTORS[name]["p2_mode"] >= 0 else [8.0, 16.0, 32.0]
    assert instance.model[-1].stride.tolist() == expected
    assert sum(parameter.numel() for parameter in instance.parameters()) < 2_500_000
    assert 0 < get_flops(instance, 640) < 15
    with torch.no_grad():
        prediction = instance(torch.rand(1, 3, 128, 160))
    assert prediction[0][1].shape == (1, 32, 64, 80)
    expected_candidates = 1700 if expected[-1] == 4 else 420
    assert prediction[0][0].shape[-1] == expected_candidates


@pytest.mark.parametrize(
    "name", ("I20_corrected_control", "I23_p2_semantic", "I26_ls_c34", "I29_p2_semantic_tinyassign")
)
def test_representative_loss_and_backward(name):
    instance = model(name)
    total, components = instance.loss(batch())
    assert isinstance(instance.criterion, EV11SegmentationLoss)
    assert components.numel() == 5 and torch.isfinite(total).all()
    total.sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in instance.parameters() if parameter.grad is not None)


def test_runner_python38_and_protocol_contracts():
    paths = (
        "RUN_CITRUS_I_V2.py",
        "20260922_citrus_i_v2_batch.py",
        "citrus_i_v2_suite.py",
        "scripts/generate_citrus_i_v2_yaml.py",
        "ultralytics/nn/modules/citrus_i_v2.py",
    )
    for path in paths:
        ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=8)
    assert set(RUNNERS["CITRUS_I_V2"].suites) == set(SUITES)
    runner = importlib.import_module("20260922_citrus_i_v2_batch")
    assert runner.NAMES == NAMES
    entry = importlib.import_module("RUN_CITRUS_I_V2")
    assert entry.DRY_RUN and entry.EPOCHS == 50 and not entry.main.__name__.startswith("nohup")


def test_tiny_assignment_uses_one_consistent_stride_predicate():
    source = (ROOT / "ultralytics/utils/tal.py").read_text(encoding="utf-8")
    assert "gt_bboxes_xywh[..., 2:] < self.stride[0]" not in source
    assert source.count("< self.stride_val") >= 2
