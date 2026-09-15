"""E V5 contracts: official API, backward, source-balanced grids, no inference GT."""

import importlib.util
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from citrus_e_v5_suite import FACTORS, NAMES, ROOT, YAML_DIR
from citrus_e_v5_slicing import UniformMultiScale, uniform_windows
from scripts.generate_citrus_e_v5_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_e_v5_loss import regional_discrimination, visible_foreground
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(empty=False):
    masks = torch.zeros(2, 32, 32)
    masks[0, 7:23, 7:23] = 1
    masks[1, 9:25, 9:25] = 1
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
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(batch(empty))
    assert torch.isfinite(loss).all() and len(items) == 5
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    head = model.model[-1]
    if factors[2]:
        assert head.region_classifier.weight.grad.abs().sum() > 0
        assert model.criterion.last_region is not None
    if factors[1] and not empty:
        assert head.mask_route_scale.grad.abs().sum() > 0
    assert get_flops(model, 640) < 10.15
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1, 3, 128, 160)
        out = model(x)
        assert "ev5_region_logits" not in out[1]
        fused = deepcopy(model).fuse(verbose=False)(x)
        torch.testing.assert_close(out[0][0], fused[0][0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(out[0][1], fused[0][1], rtol=0.002, atol=0.02)


def test_control_exact_parity_with_v4r05():
    old = SegmentationModel(
        ROOT / "0_orange_yaml/E_V4_series/reconstruction_20260910/V4R05_deep_quality.yaml", nc=1, verbose=False
    ).eval()
    new = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 128)
        a, b = old(x), new(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


def test_mask_route_leaves_candidate_path_unchanged():
    control = SegmentationModel(YAML_DIR / (NAMES[0] + ".yaml"), nc=1, verbose=False).eval()
    routed = SegmentationModel(YAML_DIR / (NAMES[2] + ".yaml"), nc=1, verbose=False).eval()
    routed.load_state_dict(control.state_dict(), strict=False)
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 160)
        a, b = control(x), routed(x)
        torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
        assert not torch.equal(a[0][1], b[0][1])


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
    assert restored.model.model[-1].region_gain == model.model.model[-1].region_gain


@pytest.mark.parametrize("shape", [(100, 200), (641, 479), (7, 9), (1, 1)])
def test_grid_covers_original_without_gt(shape):
    windows = uniform_windows(*shape)
    assert windows[0] == (0, 0, shape[1], shape[0])
    covered = np.zeros(shape, bool)
    for x0, y0, x1, y1 in windows[1:] or windows:
        assert 0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]
        covered[y0:y1, x0:x1] = True
    assert covered.all() and len(windows) == len(set(windows))
    black, white = np.zeros((*shape, 3), np.uint8), np.full((*shape, 3), 255, np.uint8)
    selector = UniformMultiScale()
    assert selector.windows(black) == selector.windows(white)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_region_empty_and_full(value):
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    embedding = torch.randn(2, 8, 8, 8, requires_grad=True)
    loss, _, _ = regional_discrimination(logits, embedding, torch.full_like(logits, value))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_visible_mask_keeps_concavity_and_vanished_ids():
    data = batch()
    data["masks"].zero_()
    data["masks"][0, 3, 3] = 2  # ID1 vanished; ID2 must remain foreground
    data["masks"][1, 3:20, 3:20] = 1
    data["masks"][1, 3:18, 8:11] = 0  # leaf occlusion/deep visible concavity
    fg = visible_foreground(data, (32, 32), True, "cpu")
    assert fg[0, 0, 3, 3] == 1 and fg[1, 0, 8, 9] == 0
    smaller = visible_foreground(data, (16, 16), True, "cpu")
    assert smaller[0].sum() == 1


def test_manifest_runner_and_protocol():
    assert configs() == [yaml.safe_load((YAML_DIR / (n + ".yaml")).read_text()) for n in NAMES]
    spec = importlib.util.spec_from_file_location("ev5_runner_test", ROOT / "20260911_citrus_e_v5_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.NAMES == NAMES and not runner.GUIDED
    source = (ROOT / "RUN_CITRUS_E_V5.py").read_text()
    assert "cache=True" in source and "amp=False" in source and "device_lock=False" in source


def test_empirical_pr_has_no_artificial_zero_or_recall_one():
    from eval_citrus_e_v5 import empirical_mask_pr

    stats = dict(
        target_cls=np.zeros(4), pred_cls=np.zeros(3), conf=np.array([0.9, 0.8, 0.7]), tp_m=np.array([[1], [0], [1]])
    )
    curve = empirical_mask_pr(stats)["0"]
    assert curve["rmax"] == 0.5 and curve["recall"][-1] == 0.5
    assert curve["precision"][-1] == 2 / 3
    stats["conf"][:] = 0.7  # threshold includes the entire tie, not a partial arbitrary order
    assert empirical_mask_pr(stats)["0"]["recall"] == [0.5]


def test_topology_proxies_do_not_call_duplicates_fragments():
    from eval_citrus_e_v5 import topology_flags

    split, merge = topology_flags(np.ones((1, 2)), np.array([100]), np.array([100, 100]))
    assert not split.any() and not merge.any()
    split, merge = topology_flags(np.full((1, 2), 0.5), np.array([100]), np.array([50, 50]))
    assert split.all() and not merge.any()
    split, merge = topology_flags(np.full((2, 1), 0.5), np.array([50, 50]), np.array([100]))
    assert not split.any() and merge.all()
