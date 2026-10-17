"""Minimal controls, gradients, prediction-cache and restart contracts."""

from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import torch
import yaml

from citrus_paper_suite import NAMES, YAML_DIR, configurations, jobs
from citrus_paper_batch import finished_training, main, verify_configs
from citrus_paper_eval import load_candidates, save_candidates, variants
from eval_citrus_e_v5 import merge_views
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def sample():
    masks = torch.zeros(2, 64, 64)
    masks[:, 12:52, 12:52] = 1
    masks[:, 3:6, 3:6] = 2
    return dict(
        img=torch.rand(2, 3, 128, 128), batch_idx=torch.tensor([0., 0., 1., 1.]),
        cls=torch.zeros(4, 1),
        bboxes=torch.tensor([[.5, .5, .625, .625], [.0703125, .0703125, .046875, .046875]] * 2),
        masks=masks,
    )


@pytest.mark.parametrize("name", NAMES)
def test_model_build_backward_and_grid(name):
    cfg = configurations()[name]
    assert yaml.safe_load((YAML_DIR / (name + ".yaml")).read_text(encoding="utf-8")) == cfg
    model = SegmentationModel(cfg, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**{
        **DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True, "nwd_ratio": 0.0,
    })
    total, components = model.loss(sample())
    assert torch.isfinite(total).all()
    total.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model[-1].stride.tolist() == [8., 16., 32.]
    flops = get_flops(model, 640)
    assert 0 < flops < 20
    model.eval()
    with torch.no_grad():
        pred = model(torch.rand(1, 3, 128, 128))
    assert pred[0][0].shape[-1] == 336
    expected = 32 if name == NAMES[0] else 64
    assert pred[0][1].shape[-2:] == (expected, expected)
    if name == NAMES[6]:
        assert not hasattr(model.model[-1], "fine_detail")
    if name == NAMES[10]:
        assert model.model[-1].quality_gain == 0
        assert model.model[-1].quality_calibration is False
    assert len(components) == 5


def candidates():
    return [
        dict(box=[1., 1., 4., 4.], score=.9, cls=0, patch=np.ones((3, 3), bool),
             offset=[1, 1], view=0, internal_border=False),
        dict(box=[1., 1., 3., 3.], score=.95, cls=0, patch=np.ones((2, 2), bool),
             offset=[1, 1], view=1, internal_border=True),
        dict(box=[6., 6., 8., 8.], score=.3, cls=0, patch=np.ones((2, 2), bool),
             offset=[6, 6], view=2, internal_border=False),
    ]


def test_cache_roundtrip_and_exact_old_merge(tmp_path):
    data = candidates()
    path = tmp_path / "cache.npz"
    save_candidates(path, data, [.01, .1])
    restored, timing = load_candidates(path)
    assert timing == [.01, .1]
    for a, b in zip(data, restored):
        np.testing.assert_array_equal(a["patch"], b["patch"])
        assert a["score"] == b["score"]
    modes = variants(restored, (10, 10))
    previous = merge_views(restored, (10, 10), border_weight=.5)
    np.testing.assert_array_equal(modes["trustedmask"]["masks"], previous["masks"])
    torch.testing.assert_close(modes["trustedmask"]["conf"], previous["conf"])
    np.testing.assert_array_equal(modes["score_only"]["masks"], modes["crossmask"]["masks"])
    np.testing.assert_array_equal(modes["selection_only"]["masks"], modes["trustedmask"]["masks"])


def test_frozen_queue_and_plan(tmp_path):
    verify_configs()
    q = jobs()
    assert len(q) == 15
    assert len([r for r in q if r["role"] == "main"]) == 6
    assert q == jobs()
    assert len(jobs("core")) == 6
    assert len(jobs(amp_audit=True)) == 17
    output = tmp_path / "not_created"
    result = main(["--data", str(tmp_path / "missing.yaml"), "--project", str(output), "--dry-run"])
    assert result == q and not output.exists()


def test_partial_checkpoint_not_completed(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/best_mask.pt").touch()
    (tmp_path / "results.csv").write_text("epoch,loss\n100,1\n", encoding="utf-8")
    assert not finished_training(tmp_path, 300)
    (tmp_path / "results.csv").write_text("epoch,loss\n300,1\n", encoding="utf-8")
    assert finished_training(tmp_path, 300)
