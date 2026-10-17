"""CPU graph, objective, sharing and official-YAML-entry contracts; not CUDA training."""

import copy
import json
from types import SimpleNamespace
import numpy as np

import pytest
import torch

from citrus_j4_batch import implementation_hash, tracked_sources
from citrus_j4_suite import NAMES, YAML_DIR
from citrus_j4_budget import select_tiles
from scripts.generate_citrus_j4_yaml import configs, render
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import SegmentCitrusJ4
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import ModelEMA, get_flops


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_batch(empty=False):
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    n = 0 if empty else 4
    return dict(img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0., 0., 1., 1.])[:n],
                cls=torch.zeros(n, 1), masks=masks * 0 if empty else masks,
                bboxes=torch.tensor([[.5, .5, .625, .625], [.07, .07, .047, .047]] * 2)[:n])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_official_yaml_forward_backward(name, empty):
    model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    model.train()
    loss, components = model.loss(make_batch(empty))
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model[-1].stride.tolist() == [8., 16., 32.]
    model.eval()
    with torch.no_grad():
        predictions = model(torch.rand(1, 3, 128, 160))
        assert predictions[0][0].shape[-1] == 420
        assert predictions[0][1].shape[-2:] == ((32, 40) if name == NAMES[0] else (64, 80))


def test_yaml_is_current_and_source_digest_complete():
    assert len(implementation_hash()) == 64
    assert {"ultralytics/engine/trainer.py", "ultralytics/utils/torch_utils.py",
            "citrus_j_batch.py", "ultralytics/nn/modules/head.py",
            "ultralytics/utils/citrus_e_v7_loss.py"}.issubset(tracked_sources())
    for name, cfg in configs().items():
        assert (YAML_DIR / (name + ".yaml")).read_text("utf-8") == render(name, cfg)


def test_shared_spatial_not_bn_or_outputs():
    model = SegmentationModel(YAML_DIR / (NAMES[5] + ".yaml"), verbose=False)
    head = model.model[-1]
    assert isinstance(head, SegmentCitrusJ4)
    assert head.cv2[0][1][0].weight is head.cv2[2][1][0].weight
    assert head.cv3[0][1][0].weight is head.cv3[1][1][0].weight
    assert head.cv4[0][1][0].weight is head.cv4[1][1][0].weight
    assert head.cv2[0][1] is not head.cv2[2][1]
    assert head.cv2[0][0].bn is not head.cv2[1][0].bn
    assert head.cv2[0][-1] is not head.cv2[1][-1]
    assert head.cv2[0][1] is not head.cv3[0][1]


def test_shared_optimizer_updates_each_parameter_once():
    model = SegmentationModel(YAML_DIR / (NAMES[5] + ".yaml"), verbose=False)
    optimizer = BaseTrainer.build_optimizer(object(), model, name="AdamW", lr=0.001, decay=0.0005)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    tied = model.model[-1].cv2[0][1][0].weight
    assert sum(p is tied for p in parameters) == 1
    tied.grad = torch.ones_like(tied)
    optimizer.step()
    assert optimizer.state[tied]["step"] == 1


def test_ema_updates_shared_weight_once():
    model = SegmentationModel(YAML_DIR / (NAMES[5] + ".yaml"), verbose=False)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5
    for p in ema.ema.parameters():
        p.zero_()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2)
    ema.update(model)
    head = ema.ema.model[-1]
    for towers in (head.cv2, head.cv3, head.cv4):
        assert torch.equal(towers[0][1][0].weight, torch.ones_like(towers[0][1][0].weight))
        assert towers[0][1][0].weight is towers[2][1][0].weight


def test_empty_masks_container_returns_no_candidates():
    from eval_citrus_j2_gaze import predict_view

    result = SimpleNamespace(masks=SimpleNamespace(data=torch.empty(0, 384, 640)))
    model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    assert predict_view(model, np.zeros((100, 160, 3), np.uint8), (0, 0, 160, 100), (400, 640), "cpu") == []


def test_fuse_preserves_predictions_and_shared_weights():
    model = SegmentationModel(YAML_DIR / (NAMES[5] + ".yaml"), verbose=False).eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        before = model(image)[0]
        fused = copy.deepcopy(model).fuse(verbose=False)
        after = fused(image)[0]
    assert torch.allclose(before[0], after[0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(before[1], after[1], atol=1e-4, rtol=1e-4)
    head = fused.model[-1]
    assert head.cv2[0][1][0].weight is head.cv2[2][1][0].weight


@pytest.mark.parametrize("policy,expected", [("coverage", 3), ("random", 3), ("global", 1), ("all", 10)])
def test_budget_inference_forwards_only_selected_views(monkeypatch, policy, expected):
    import eval_citrus_j4_budget as evaluation

    calls = []

    def fake_predict(model, image, window, shape, device):
        calls.append(window)
        return []

    monkeypatch.setattr(evaluation, "predict_view", fake_predict)
    _, selected, shape = evaluation.infer_budget(None, np.zeros((100, 160, 3), np.uint8),
                                                 budget=2, policy=policy, seed=42)
    assert len(calls) == expected == 1 + len(selected)
    assert shape == (400, 640)


def test_cost_is_measured_not_claimed(tmp_path):
    rows = []
    for name in NAMES:
        model = SegmentationModel(YAML_DIR / (name + ".yaml"), verbose=False)
        count = sum(p.numel() for p in model.parameters())
        flops = get_flops(model, 640)
        assert 0 < flops < 14
        rows.append(dict(model=name, params=count, gflops=flops))
    assert rows[5]["params"] < rows[1]["params"] and rows[5]["gflops"] < rows[1]["gflops"]
    (tmp_path / "j4_costs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_budget_has_no_gt_input_and_keeps_fixed_cost():
    windows = [(0, 0, 40, 40), (30, 0, 70, 40), (60, 0, 100, 40)]
    candidates = [dict(box=np.array([68, 5, 72, 9]), score=.9, patch=np.ones((4, 4), bool))]
    selected = select_tiles(candidates, windows, budget=2)
    assert selected[0] == 2 and len(set(selected)) == 2
    assert len(select_tiles([], windows, 2)) == 2
    assert len(select_tiles(candidates, windows, 2, "random", 42)) == 2
    assert select_tiles(candidates, windows, 2, "global") == []
    assert select_tiles(candidates, windows, 2, "all") == [0, 1, 2]
    assert select_tiles(candidates, windows, 0) == []
    with pytest.raises(ValueError):
        select_tiles(candidates, windows, True)
