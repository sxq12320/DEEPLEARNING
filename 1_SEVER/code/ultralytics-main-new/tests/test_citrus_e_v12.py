"""V12 official entry, exact controls, task isolation, gradients, fuse and runner contracts."""

import ast
import importlib
from copy import deepcopy

import pytest
import torch
import yaml

from citrus_e_v11_suite import NAMES as OLD_NAMES
from citrus_e_v12_suite import FACTORS, NAMES, ROOT, SUITES, YAML_DIR
from citrus_foreground import RUNNERS
from scripts.generate_citrus_e_v12_yaml import configs
from ultralytics import YOLO
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


def batch(empty=False):
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    n = 0 if empty else 4
    return dict(
        img=torch.rand(2, 3, 128, 128),
        batch_idx=torch.tensor([0.0, 0.0, 1.0, 1.0])[:n],
        cls=torch.zeros(n, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.625, 0.625], [0.0703125, 0.0703125, 0.046875, 0.046875]] * 2)[:n],
        masks=masks * 0 if empty else masks,
    )


def model(i):
    m = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False)
    m.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True, "nwd_ratio": 0.0})
    return m


@pytest.mark.parametrize("i", range(len(NAMES)))
@pytest.mark.parametrize("empty", [False, True])
def test_build_backward_validation_fuse(i, empty):
    m = model(i)
    total, components = m.loss(batch(empty))
    assert isinstance(m.criterion, EV11SegmentationLoss)
    assert len(components) == 5 and torch.isfinite(total).all()
    total.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    if i >= 3 and not empty:
        assert all(route.gain.grad.abs().sum() > 0 for route in m.model[-1].recognition)
        # The route must learn beyond its initial zero gain on a later step.
        with torch.no_grad():
            for route in m.model[-1].recognition:
                route.gain.fill_(0.1)
        m.zero_grad(set_to_none=True)
        m.loss(batch())[0].sum().backward()
        assert all(route.reference.conv.weight.grad.abs().sum() > 0 for route in m.model[-1].recognition)
    assert m.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert 0 < get_flops(m, 640) < 11
    m.eval()
    with torch.no_grad():
        b = batch(empty)
        pred = m(b["img"])
        assert torch.isfinite(m.loss(b, pred)[0]).all()
        x = torch.rand(1, 3, 128, 160)
        a = m(x)
        fused = deepcopy(m).fuse(verbose=False)(x)
        assert a[0][0].shape[-1] == 420
        assert a[0][1].shape == (1, 32, 64, 80)
        torch.testing.assert_close(a[0][0], fused[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(a[0][1], fused[0][1], rtol=0.002, atol=0.02)


@pytest.mark.parametrize("i", [0, 1, 2, 3, 4, 8, 9])
def test_exact_replay_or_identity_initialization(i):
    parent = FACTORS[i][0]
    old = SegmentationModel(ROOT / f"0_orange_yaml/E_V11_series/{OLD_NAMES[parent]}.yaml", nc=1, verbose=False).eval()
    new = model(i).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=i < 3)
    assert not unexpected
    assert all("recognition" in k for k in missing)
    with torch.no_grad():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)
    if i < 3:
        assert old.model[-1].tiny_dice_gain == new.model[-1].tiny_dice_gain


def test_classification_route_does_not_change_geometry_or_mask_at_fixed_features():
    head = model(3).model[-1].eval()
    # Read actual tower input widths rather than relying on the declared YAML widths.
    dims = [tower[0].conv.in_channels for tower in head.cv2]
    features = [torch.rand(1, c, 16 // (2**i), 20 // (2**i)) for i, c in enumerate(dims)]
    with torch.no_grad():
        a = head.forward_head(features, **head.one2many)
        for route in head.recognition:
            route.gain.fill_(0.5)
        b = head.forward_head(features, **head.one2many)
    for key in ("boxes", "mask_coefficient", "ev3_quality"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert not torch.equal(a["scores"], b["scores"])


@pytest.mark.parametrize("i", range(len(NAMES)))
def test_yaml_and_pretrained_mapping_and_checkpoint(i, tmp_path):
    cfg = yaml.safe_load((YAML_DIR / f"{NAMES[i]}.yaml").read_text(encoding="utf-8"))
    assert cfg == configs()[i]
    api = YOLO(str(YAML_DIR / f"{NAMES[i]}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    state = api.model.state_dict()
    persistent = FACTORS[i][0] == 2
    for new, old in ((8 if persistent else 6, 6), (23 if persistent else 19, 19)):
        torch.testing.assert_close(
            state[f"model.{new}.cv1.conv.weight"], source[f"model.{old}.cv1.conv.weight"], rtol=0, atol=0
        )
    path = tmp_path / "model.pt"
    api.save(str(path))
    reloaded = YOLO(str(path), verbose=False)
    assert type(reloaded.model.model[-1]).__name__ == "SegmentCitrusEV12"
    assert reloaded.model.yaml == api.model.yaml


def test_python38_and_runner_contracts():
    files = [
        "RUN_CITRUS_E_V12.py",
        "20260918_citrus_e_v12_batch.py",
        "citrus_e_v12_suite.py",
        "ultralytics/nn/modules/citrus_e_v12.py",
    ]
    for path in files:
        ast.parse((ROOT / path).read_text(encoding="utf-8"), feature_version=8)
    assert set(RUNNERS["CITRUS_E_V12"].suites) == set(SUITES)
    assert "paper" in RUNNERS["CITRUS_E_V11"].suites
    runner = importlib.import_module("20260918_citrus_e_v12_batch")
    assert runner.NAMES == NAMES
    entry = importlib.import_module("RUN_CITRUS_E_V12")
    assert not entry.DRY_RUN and entry.EPOCHS == 300


def test_runner_provenance_sources_exist():
    file = ROOT / "20260918_citrus_e_v12_batch.py"
    tree = ast.parse(file.read_text(encoding="utf-8"))
    node = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "source_files" for t in n.targets)
    )
    from pathlib import Path

    paths = eval(
        compile(ast.Expression(node), str(file), "eval"),
        {"ROOT": ROOT, "YAML_DIR": YAML_DIR, "NAMES": NAMES, "Path": Path, "__file__": str(file)},
    )
    assert all(p.is_file() for p in paths)


def test_source_membership_is_recorded_without_mutating_dataset(tmp_path):
    import json
    from types import SimpleNamespace

    runner = importlib.import_module("20260918_citrus_e_v12_batch")
    manifest = tmp_path / "views.json"
    payload = {
        "groups": [
            {
                "source": "/actual/train/fruit.jpg",
                "instances": 3,
                "original_shape": [720, 1280],
                "views": [
                    {"global_view": True, "image": "train/images/source000000_view0.png"},
                    {"global_view": False, "image": "train/images/source000000_view1.png"},
                ],
            }
        ]
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    original = manifest.read_bytes()
    assert runner.record_source_membership(SimpleNamespace(data={"path": str(tmp_path)}), tmp_path)
    records = json.loads((tmp_path / "train_original_sources.json").read_text(encoding="utf-8"))
    assert records[0]["source"] == "/actual/train/fruit.jpg"
    assert records[0]["cached_views"] == 2 and manifest.read_bytes() == original
