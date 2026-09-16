"""V10 official YAML, backward/fuse, loss visibility and runner contracts."""

import importlib
import ast
import json
from copy import deepcopy

import pytest
import numpy as np
import torch
import yaml

from citrus_e_v10_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, YAML_DIR
from citrus_pr_diagnostics import write_pr_diagnostics
from scripts.generate_citrus_e_v10_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_e_v10 import EV10ContrastStem
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_e_v10_loss import EV10SegmentationLoss, visible_foreground_loss
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
def test_build_backward_fuse(i, empty):
    model = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False)
    args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    args.mask_ratio, args.overlap_mask, args.nwd_ratio = 2, True, 0.0
    model.args = args
    total, components = model.loss(batch(empty))
    assert isinstance(model.criterion, EV10SegmentationLoss)
    assert len(components) == 5 and torch.isfinite(total).all()
    total.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if FACTORS[i][0]:
        assert model.model[0].structure.gain.grad is not None
    if FACTORS[i][4]:
        assert model.model[-1].visibility.weight.grad.abs().sum() > 0
    if not empty and FACTORS[i][1]:
        assert model.model[-1].detail_transport.gate.weight.grad.abs().sum() > 0
    assert model.model[-1].stride.tolist() == [8.0, 16.0, 32.0]
    assert 0 < get_flops(model, 640) < 11
    # Trainer validation computes loss using eval-mode outputs; the training
    # auxiliary must not be required or accidentally evaluated there.
    with torch.no_grad():
        model.eval()
        validation_batch = batch(empty)
        predictions = model(validation_batch["img"])
        val_loss, _ = model.loss(validation_batch, predictions)
        assert torch.isfinite(val_loss).all()
        assert model.criterion.last_visibility is None
    with torch.inference_mode():
        model.eval()
        if FACTORS[i][0]:
            model.model[0].structure.gain.fill_(0.2)
        x = torch.rand(1, 3, 128, 160)
        a = model(x)
        assert a[0][1].shape == (1, 32, 64, 80)
        assert a[0][0].shape[-1] == 420
        b = deepcopy(model).fuse(verbose=False)(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0.002, atol=0.02)


@pytest.mark.parametrize("i", [0, 1])
def test_exact_controls(i):
    old_name = ["V9_01_geometry_cp_anchor", "V9_05_tiny_overlap"][i]
    old = SegmentationModel(ROOT / f"0_orange_yaml/E_V9_series/{old_name}.yaml", nc=1, verbose=False).eval()
    new = SegmentationModel(YAML_DIR / f"{NAMES[i]}.yaml", nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    assert old.model[-1].tiny_dice_gain == new.model[-1].tiny_dice_gain
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("i", range(10))
def test_pretrain_reload(i, tmp_path):
    api = YOLO(str(YAML_DIR / f"{NAMES[i]}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    assert api.task == "segment"
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    for key in ("model.0.conv.weight", "model.6.cv1.conv.weight", "model.19.cv1.conv.weight"):
        torch.testing.assert_close(api.model.state_dict()[key], source[key], rtol=0, atol=0)
    path = tmp_path / "model.pt"
    api.save(str(path))
    assert YOLO(str(path), verbose=False).model.yaml == api.model.yaml


def test_contrast_flat_field_and_zero_start():
    stem = EV10ContrastStem(3, 16).eval()
    x = torch.full((1, 3, 40, 56), 0.7)
    assert stem.contrast(x).abs().max() < 0.0001
    with torch.no_grad():
        torch.testing.assert_close(stem(x), stem.act(stem.bn(stem.conv(x))), rtol=0, atol=0)


@pytest.mark.parametrize("overlap", [True, False])
def test_visibility_includes_tiny_without_assignments(overlap):
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    masks = torch.zeros(1, 32, 32)
    masks[:, 3, 3] = 1
    value = visible_foreground_loss(logits, masks, torch.tensor([0.0]), overlap)
    value.backward()
    assert logits.grad[0, 0, 1, 1] < 0  # foreground promoted
    assert logits.grad[0, 0, 9, 9] > 0  # background suppressed
    assert torch.isfinite(value)


def test_visibility_area_balanced_and_background_only():
    x = torch.zeros(1, 1, 16, 16, requires_grad=True)
    labels = torch.zeros(1, 16, 16)
    labels[:, 1, 1] = 1
    labels[:, 5:10, 5:10] = 2
    visible_foreground_loss(x, labels, torch.tensor([0.0, 0.0]), True).backward()
    torch.testing.assert_close(x.grad[0, 0, 1, 1], x.grad[0, 0, 5:10, 5:10].sum())
    blank = torch.zeros_like(x, requires_grad=True)
    visible_foreground_loss(blank, labels * 0, torch.tensor([]), True).backward()
    assert (blank.grad > 0).all()


def test_shared_key_centering_does_not_change_softmax():
    q, k = torch.randn(2, 9, 8), torch.randn(2, 9, 8)
    a = (q @ k.transpose(-1, -2)).softmax(-1)
    b = (q @ (k - k.mean(1, keepdim=True)).transpose(-1, -2)).softmax(-1)
    torch.testing.assert_close(a, b)


def test_runner_manifest_and_budget():
    assert configs() == [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    runner = importlib.import_module("20260916_citrus_e_v10_batch")
    entry = importlib.import_module("RUN_CITRUS_E_V10")
    assert runner.NAMES == NAMES and not entry.DRY_RUN and entry.EPOCHS == 300
    assert RUN_OVERRIDES[NAMES[9]] == {**RUN_OVERRIDES[NAMES[8]], "cos_lr": True}
    full = SegmentationModel(YAML_DIR / f"{NAMES[8]}.yaml", nc=1, verbose=False)
    base = SegmentationModel(YAML_DIR / f"{NAMES[1]}.yaml", nc=1, verbose=False)
    assert get_flops(full, 640) < get_flops(base, 640)


def test_pr_diagnostic_does_not_modify_source(tmp_path):
    curve = dict(targets=10, rmax=0.8, precision=[1.0, 0.8, 0.4], recall=[0.1, 0.6, 0.8], confidence=[0.9, 0.5, 0.01])
    path = tmp_path / "paired_metrics.json"
    path.write_text(json.dumps(dict(empirical_mask_pr={m: {"0": curve} for m in ("global", "trustedmask")})))
    original = path.read_bytes()
    out = write_pr_diagnostics(path)
    assert original == path.read_bytes()
    summary = json.loads((out / "summary.json").read_text())
    assert summary[0]["last_observed_precision"] == 0.4
    assert summary[0]["observed_rmax"] == 0.8
    assert len((out / "global_class0_observed.csv").read_text().splitlines()) == 4


def test_complete_threshold_export_keeps_legacy_default():
    from eval_citrus_e_v5 import empirical_mask_pr

    stats = dict(
        target_cls=np.zeros(1), pred_cls=np.zeros(2001), conf=np.linspace(1, 0, 2001), tp_m=np.zeros((2001, 10))
    )
    stats["tp_m"][0, 0] = 1
    assert len(empirical_mask_pr(stats)["0"]["confidence"]) == 1000
    full = empirical_mask_pr(stats, max_points=None)["0"]
    assert len(full["confidence"]) == 2001
    assert full["rmax"] == 1


def test_source_provenance_files_exist_and_python38_syntax():
    runner = ROOT / "20260916_citrus_e_v10_batch.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "source_files" for t in n.targets)
    )
    files = eval(
        compile(ast.Expression(node.value), str(runner), "eval"),
        dict(ROOT=ROOT, YAML_DIR=YAML_DIR, NAMES=NAMES, Path=type(ROOT), __file__=str(runner)),
    )
    assert all(p.is_file() for p in files)
    for p in files:
        if p.suffix == ".py" and ("v10" in p.stem.lower() or p.stem == "citrus_pr_diagnostics"):
            ast.parse(p.read_text(encoding="utf-8"), feature_version=(3, 8))
