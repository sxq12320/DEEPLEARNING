"""V6 standard API, topology, finite standard loss, initialized exchange and launch contract."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from citrus_sage_v6_suite import NAMES, ROOT, SUITES, YAML_DIR, select_names
from tests.test_citrus_sage_v4r import example_batch
from ultralytics import YOLO
from ultralytics.nn.modules import SAGEV6Exchange, SAGEV6Stage
from ultralytics.nn.modules.block import Proto
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", NAMES)
def test_public_api_loss_backward_geometry_and_flops(name, tmp_path):
    api = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment", verbose=False)
    model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    assert isinstance(model.model[-1].proto, Proto)
    assert not model.model[-1].legacy
    loss, items = model.loss(example_batch())
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert (items[-1] > 0) == (name == NAMES[-1])
    for module in model.modules():
        if isinstance(module, (SAGEV6Exchange, SAGEV6Stage)):
            for key, parameter in module.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), key
    with torch.no_grad():
        model.eval()
        assert model(torch.rand(1, 3, 128, 160))[0][1].shape == (1, 32, 32, 40)
        model.model[-1].export = True
        assert model(torch.rand(1, 3, 160, 128))[1].shape == (1, 32, 40, 32)
        model.model[-1].export = False
    assert get_flops(model, 640) < 10.4
    assert sum(p.numel() for p in model.parameters()) < 2_350_000
    filename = tmp_path / f"{name}.pt"
    api.save(str(filename))
    restored = YOLO(str(filename), verbose=False).model.eval()
    with torch.no_grad():
        assert restored(torch.rand(1, 3, 128, 128))[0][1].shape[-2:] == (32, 32)


@pytest.mark.parametrize("shape", [(16, 20), (64, 80)])
def test_add_select_initial_equivalence_and_input_immutability(shape):
    add = SAGEV6Exchange((16, 32), 16, "add")
    select = SAGEV6Exchange((16, 32), 16, "select")
    select.load_state_dict(add.state_dict(), strict=False)
    x = [torch.rand(2, 16, 32, 40), torch.rand(2, 32, *shape)]
    before = [v.clone() for v in x]
    torch.testing.assert_close(add(x), select(x), rtol=0, atol=0)
    select(x).square().mean().backward()
    assert select.gate.weight.grad.abs().sum() > 0
    for a, b in zip(x, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_config_factors_and_original_control():
    cfg = {n: yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES}
    old = yaml.safe_load((ROOT / "0_orange_yaml/SAGE_series/SAGE51_detail_relay.yaml").read_text())
    assert cfg[NAMES[0]] == old
    assert cfg[NAMES[1]]["backbone"] == old["backbone"]
    assert cfg[NAMES[1]]["head"] == cfg[NAMES[2]]["head"]
    assert all(row[2] != "C3k2" for row in cfg[NAMES[2]]["backbone"])
    a, b = deepcopy(cfg[NAMES[3]]), deepcopy(cfg[NAMES[4]])
    for row in b["backbone"] + b["head"]:
        if row[2] == "SAGEV6Exchange":
            row[3][1] = "add"
    assert a == b
    a, b = deepcopy(cfg[NAMES[4]]), deepcopy(cfg[NAMES[5]])
    b["head"][-1][-1][-2:] = [0, 0]
    assert a == b
    assert select_names("all") == list(NAMES)
    assert set(SUITES["priority"]).issubset(NAMES)
    with pytest.raises(ValueError):
        select_names("all", "bad")


def test_repeat_count_and_candidate_gradient_to_persistent_detail():
    config = yaml.safe_load((YAML_DIR / f"{NAMES[4]}.yaml").read_text())
    config["backbone"][2][1] = 4
    model = SegmentationModel(config, nc=1, verbose=False)
    assert len(model.model[2].blocks) == 2  # depth=0.5, internally repeated ONCE
    prediction = model(torch.rand(2, 3, 128, 128))
    prediction["scores"].square().mean().backward()
    assert model.model[5].conv.weight.grad.abs().sum() > 0
    assert model.model[11].blocks[0].spatial.conv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("empty", [True, False])
def test_empty_and_tiny_instances_are_finite(empty):
    model = SegmentationModel(YAML_DIR / f"{NAMES[-1]}.yaml", nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, _ = model.loss(example_batch(empty=empty))
    loss.sum().backward()
    assert torch.isfinite(loss).all()


def test_launcher_import_is_inert_and_default_300(monkeypatch):
    from citrus_foreground import RUNNERS

    spec = importlib.util.spec_from_file_location("test_run_sage6", ROOT / "RUN_SAGE_V6.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run_foreground", lambda **kwargs: calls.append(kwargs))
    assert calls == []
    module.main()
    assert calls[0]["epochs"] == 300 and calls[0]["dry_run"] is False
    assert calls[0]["suite"] == "structure" and calls[0]["series"] == "SAGE_V6"
    assert Path(ROOT / RUNNERS["SAGE_V6"].script).is_file()


def test_semantic_initializer_remap_and_trainer_reconstruction():
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model
    target = YOLO(str(YAML_DIR / f"{NAMES[4]}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt")).model
    for dst, src in {8: 5, 13: 7, 15: 9, 16: 10}.items():
        for key, value in source.model[src].state_dict().items():
            torch.testing.assert_close(value, target.model[dst].state_dict()[key], rtol=0, atol=0)
    # .train() reconstructs the model and loads THIS adapted model, not the original checkpoint.
    rebuilt = SegmentationModel(target.yaml, nc=80, verbose=False)
    rebuilt.load(target, verbose=False)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)


def test_downward_exchange_pools_before_channel_expansion():
    block = SAGEV6Exchange((128, 16), 128)
    seen = []
    hook = block.source.register_forward_pre_hook(lambda _, inputs: seen.append(tuple(inputs[0].shape)))
    assert block([torch.rand(2, 128, 8, 10), torch.rand(2, 16, 32, 40)]).shape == (2, 128, 8, 10)
    hook.remove()
    assert seen == [(2, 16, 8, 10)]
