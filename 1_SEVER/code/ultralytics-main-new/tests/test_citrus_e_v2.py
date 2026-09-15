"""E V2: ordinary YAML API, actual gradients, paired mask merging and batch training."""

import importlib.util
import json
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from citrus_e_v2_suite import NAMES, ROOT, TILE_PROBABILITY, YAML_DIR
from eval_citrus_e_v2 import add_view_metadata, merge_views
from tests import test_citrus_e_batch as old_batch_tests
from tests.test_citrus_sage_v4r import example_batch
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads_and_visibility(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def batch_module():
    spec = importlib.util.spec_from_file_location("citrus_ev2_test_runner", ROOT / "20260908_citrus_e_v2_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_public_api_actual_segmentation_loss_and_fusion(name, empty):
    path = YAML_DIR / f"{name}.yaml"
    assert YOLO(str(path), verbose=False).task == "segment"
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(example_batch(empty=empty))
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    assert items[-1] == 0  # no silent inherited geometry objective
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not empty:
        for module in model.modules():
            if module.__class__.__module__.endswith("citrus_e_v2"):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.model[-1].stride.tolist() == [8, 16, 32]
    assert get_flops(model, 640) < 10.3
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1, 3, 128, 160)
        expected = model(x)[0]
        actual = deepcopy(model).fuse(verbose=False)(x)[0]
        torch.testing.assert_close(expected[0], actual[0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(expected[1], actual[1], rtol=0.002, atol=0.02)


@pytest.mark.parametrize("name", NAMES)
def test_initialization_reindex_and_own_checkpoint_roundtrip(name, tmp_path):
    api = YOLO(str(YAML_DIR / f"{name}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.state_dict()
    head_index = len(api.model.model) - 1
    # Head transferred deliberately from layer 23 to layer 15 in the new topology.
    key = f"model.{head_index}.cv2.0.0.conv.weight"
    torch.testing.assert_close(api.model.state_dict()[key], source["model.23.cv2.0.0.conv.weight"], rtol=0, atol=0)
    rebuilt = SegmentationModel(api.model.yaml, nc=80, verbose=False)
    rebuilt.load(api.model, verbose=False)
    for key, value in api.model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    filename = tmp_path / "model.pt"
    api.save(str(filename))
    assert YOLO(str(filename), verbose=False).task == "segment"


def test_factor_isolation_and_compute_budget():
    configs = [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    from scripts.generate_citrus_e_v2_yaml import configs as declared_configs

    assert configs == declared_configs()
    assert configs[0] == yaml.safe_load((ROOT / "0_orange_yaml/E_series/E01_sliced_control.yaml").read_text())
    assert configs[5] == configs[6] == configs[7]
    assert configs[0]["backbone"] == configs[1]["backbone"] == configs[2]["backbone"]
    assert configs[3]["backbone"] == configs[4]["backbone"] == configs[5]["backbone"]
    assert configs[0]["head"] == configs[3]["head"]
    assert TILE_PROBABILITY[NAMES[6]] == 0 and TILE_PROBABILITY[NAMES[7]] == 0.5
    control = SegmentationModel(configs[0], nc=1, verbose=False)
    full = SegmentationModel(configs[5], nc=1, verbose=False)
    assert sum(p.numel() for p in full.parameters()) < 0.8 * sum(p.numel() for p in control.parameters())
    assert get_flops(full, 640) < 0.92 * get_flops(control, 640)


def candidate(patch=None, offset=(0, 0), view=0, border=False, score=0.9, box=None):
    patch = np.ones((10, 10), bool) if patch is None else patch
    x, y = offset
    box = [x, y, x + patch.shape[1], y + patch.shape[0]] if box is None else box
    return dict(patch=patch, offset=offset, view=view, internal_border=border, score=score, box=box, cls=0)


def test_cross_view_duplicates_fragments_and_touching_instances():
    full = candidate()
    duplicate = candidate(view=1, score=0.8)
    fragment = candidate(patch=np.ones((2, 10), bool), view=2, border=True, score=0.95)
    touching = candidate(offset=(10, 0), view=1, score=0.7)
    result = merge_views([full, duplicate, fragment, touching], (20, 20), border_weight=0.5)
    assert len(result["conf"]) == 2
    assert result["masks"].sum((1, 2)).tolist() == [100, 100]
    assert full["score"] == 0.9 and fragment["score"] == 0.95  # never mutate cached predictions
    assert merge_views([], (20, 20))["masks"].shape == (0, 20, 20)
    # Same-view suppression belongs to the detector; do not add an unseen second rule.
    assert len(merge_views([full, candidate()], (20, 20))["conf"]) == 2


def test_disjoint_masks_with_overlapping_boxes_remain_separate():
    a = np.zeros((10, 10), bool)
    a[:, :4] = True
    b = np.zeros((10, 10), bool)
    b[:, 6:] = True
    result = merge_views([candidate(a), candidate(b, view=1)], (10, 10))
    assert len(result["conf"]) == 2
    assert not (result["masks"][0] & result["masks"][1]).any()


def test_only_internal_crop_edges_receive_border_flag():
    whole_edge = add_view_metadata([candidate(box=[0, 0, 10, 10])], (0, 0, 100, 100), (100, 100), 0)[0]
    crop_edge = add_view_metadata([candidate(box=[49, 20, 50, 30])], (0, 0, 50, 50), (100, 100), 1)[0]
    assert not whole_edge["internal_border"] and crop_edge["internal_border"]


def test_real_paired_evaluator_same_windows_same_merge_metrics(tmp_path, monkeypatch):
    """Real frozen detector; a fixed-window guide checks equal-budget evaluation plumbing."""
    import citrus_crop_guide
    from citrus_slicing import view_windows
    from eval_citrus_e_v2 import evaluate

    weights = (
        ROOT.parents[1]
        / "results/E/CITRUS_E9_GUIDED_DEVICEBOUND_ALL_300EP/E00_global_control_seed42/weights/best_mask.pt"
    )
    data = ROOT / "reports/sage_v8_20260907/diagnostics/local_validation.yaml"
    if not weights.is_file() or not data.is_file():
        pytest.skip("Optional local completed weights/data unavailable")

    class FixedWindowGuide:
        def __init__(self, checkpoint):
            self.sha256 = "TEST_ONLY_FIXED_WINDOWS_NOT_A_LEARNED_GUIDE"

        def heatmap(self, image):
            return np.zeros((96, 96), np.float32)

        def windows(self, image, fraction):
            return view_windows(*image.shape[:2], fraction)

    monkeypatch.setattr(citrus_crop_guide, "CropGuide", FixedWindowGuide)
    summary = evaluate(weights, data, tmp_path / "paired", limit=1, guide_checkpoint="test-only")
    for a, b in (("fixed_box", "guided_box"), ("crossmask", "guided_crossmask"), ("trustedmask", "guided_trustedmask")):
        assert summary[a]["metrics/mAP50(M)"] == pytest.approx(summary[b]["metrics/mAP50(M)"])
        assert summary[a]["errors25"] == summary[b]["errors25"]
        assert "operating_p90" in summary[a]


@pytest.mark.parametrize("interrupt", [False, True])
def test_actual_new_queue_safety(batch_module, tmp_path, monkeypatch, interrupt):
    monkeypatch.setattr(old_batch_tests, "NAMES", NAMES)
    old_batch_tests.test_queue_is_sequential_locked_and_interruptible(batch_module, tmp_path, monkeypatch, interrupt)


@pytest.mark.parametrize("name", NAMES)
def test_real_one_epoch_runner_and_source_balanced_data(batch_module, tmp_path, monkeypatch, name):
    old_batch_tests.test_real_trainer_calls_batch_callbacks_on_explicit_smoke_fixture(
        batch_module, tmp_path, monkeypatch, name
    )


@pytest.mark.parametrize("guided", [False, True])
def test_paired_evaluation_retry_and_equal_merging(batch_module, tmp_path, monkeypatch, guided):
    import eval_citrus_e_v2

    run = tmp_path / (NAMES[7] + "_seed42" if guided else NAMES[0] + "_seed42")
    partial = run / "paired_sliced_eval"
    partial.mkdir(parents=True)
    (partial / "preserve.txt").write_text("preserve")
    if guided:
        guide = tmp_path / "_crop_guide/guide.pt"
        guide.parent.mkdir()
        guide.write_bytes(b"fake")
    calls = []

    def fake(weights, data, output, **kwargs):
        assert ("guide_checkpoint" in kwargs) == guided
        calls.append(output)
        output.mkdir()
        (output / "paired_metrics.json").write_text("{}")

    monkeypatch.setattr(eval_citrus_e_v2, "evaluate", fake)
    batch_module.finish_paired_evaluation(run, "unused.yaml", "cpu")
    batch_module.finish_paired_evaluation(run, "unused.yaml", "cpu")
    assert calls == [run / "paired_sliced_eval_retry1"]
    assert (partial / "preserve.txt").read_text() == "preserve"
    assert json.loads((run / "paired_evaluation.json").read_text())["report"]
