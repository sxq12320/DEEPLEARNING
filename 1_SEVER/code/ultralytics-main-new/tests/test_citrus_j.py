"""J official-YAML contracts, task losses and mathematical counterexamples."""

from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from citrus_j_batch import settings
from citrus_j_diagnostics import observed_mask_pr
from citrus_j_suite import FACTORS, NAMES, RUN_OVERRIDES, YAML_DIR, select_names
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.citrus_j import JContext, JFusion, JStage
from ultralytics.utils.citrus_j_loss import JSegmentationLoss, binary_boundary
from ultralytics.utils.torch_utils import get_flops

torch.set_num_threads(2)


def batch_fixture(overlap=True, empty=False):
    masks = torch.zeros((2, 48, 64)) if overlap else torch.zeros((0 if empty else 2, 48, 64))
    if not empty:
        masks[0, 12:24, 16:32] = 1
        masks[1, 22:26, 30:34] = 1
    return dict(
        img=torch.rand(2, 3, 96, 128),
        masks=masks,
        batch_idx=torch.tensor([] if empty else [0, 1]),
        cls=torch.zeros((0 if empty else 2, 1)),
        bboxes=torch.zeros((0, 4))
        if empty
        else torch.tensor([[0.375, 0.375, 0.25, 0.25], [0.5, 0.5, 0.0625, 0.08333333]]),
    )


@pytest.mark.parametrize("name", NAMES)
def test_all_yaml_train_eval_backward(name):
    model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model
    model.args = get_cfg(overrides={"overlap_mask": True, "mask_ratio": 2})
    model.train()
    batch = batch_fixture()
    predictions = model(batch["img"])
    assert predictions["scores"].shape == (2, 1, 252)
    loss, components = model.loss(batch, predictions)
    assert components.shape == (5,) and torch.isfinite(loss).all()
    loss.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    if name not in NAMES[:2]:
        assert isinstance(model.init_criterion(), JSegmentationLoss)
        assert not any(type(layer).__name__ == "C3k2" for layer in model.model)
        assert len(model.model[2].blocks) == 2
        assert len(model.model[6].blocks) == 3
        expected = (48, 64) if FACTORS[name]["proto"] == 2 else (24, 32)
        assert predictions["proto"].shape[-2:] == expected
    model.eval()
    with torch.no_grad():
        output, raw = model(batch["img"])
    assert output[0].shape == (2, 37, 252)
    assert output[1].shape[1] == 32 and "proto" in raw


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_joint_loss_empty_and_nonoverlap(overlap, empty):
    model = YOLO(str(YAML_DIR / "J16_joint_mask_loss.yaml"), task="segment", verbose=False).model
    model.args = get_cfg(overrides={"overlap_mask": overlap, "mask_ratio": 2})
    batch = batch_fixture(overlap, empty)
    predictions = model(batch["img"])
    loss, _ = model.loss(batch, predictions)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert model.criterion.extra_mask is None
    assert model.model[-1].scene[-1].weight.grad is not None


def test_key_mean_centering_is_softmax_noop():
    q, k = torch.randn(2, 7, 4, dtype=torch.double), torch.randn(2, 9, 4, dtype=torch.double)
    original = (q @ k.transpose(-1, -2)).softmax(-1)
    centered = (q @ (k - k.mean(1, keepdim=True)).transpose(-1, -2)).softmax(-1)
    torch.testing.assert_close(original, centered, atol=1e-12, rtol=1e-12)


def test_differential_read_constant_values_cancel():
    q1, q2, k1, k2 = torch.randn(2, 7, 4), torch.randn(2, 7, 4), torch.randn(2, 9, 4), torch.randn(2, 9, 4)
    constant = torch.randn(2, 1, 5).expand(-1, 9, -1)
    result = JContext.read(q1, k1, q2, k2, constant)
    assert result.abs().max() < 2e-6
    assert JContext.read(q1, k1, q2, k2, constant, False).abs().max() > 1e-3


def test_rearrangement_inverse_norm_and_fusion_bound():
    x = torch.randn(2, 3, 12, 16)
    rearranged = F.pixel_unshuffle(x, 2)
    torch.testing.assert_close(F.pixel_shuffle(rearranged, 2), x)
    torch.testing.assert_close(rearranged.norm(), x.norm())
    module = JFusion([8, 16, 24], 12).eval()
    values = module.aligned([torch.randn(2, 8, 12, 16), torch.randn(2, 16, 6, 8), torch.randn(2, 24, 3, 4)])
    weights = module.score(torch.cat(values, 1)).softmax(1)
    mixed = sum(value * weights[:, i : i + 1] for i, value in enumerate(values))
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    stacked = torch.stack(values)
    assert (mixed >= stacked.amin(0) - 1e-6).all() and (mixed <= stacked.amax(0) + 1e-6).all()


def test_boundary_keeps_visible_concavity():
    masks = torch.ones(1, 12, 12)
    masks[:, :8, 5:8] = 0
    band = binary_boundary(masks)
    assert band[0, 7, 6] == 1 and masks[0, 7, 6] == 0
    assert band[0, 4, 2] == 0


def test_observed_pr_complete_ties_no_padding(tmp_path):
    stats = dict(conf=[np.array([0.9, 0.9, 0.2])], tp_m=[np.array([[True], [False], [True]])], target_cls=[np.zeros(4)])
    result = observed_mask_pr(stats, tmp_path / "points.csv")
    assert result["observed_thresholds"] == 2
    assert result["rmax"] == 0.5 and result["endpoint"]["precision"] == pytest.approx(2 / 3)
    assert result["recall_at_precision_ge_090"] == 0
    assert observed_mask_pr({})["endpoint"] is None


def test_factors_and_recipe_are_isolated():
    assert len(select_names("all")) == 20 and len(select_names("priority")) == 7
    recipe = settings(50, 42)
    assert recipe["amp"] is False and recipe["cache"] is True and recipe["pretrained"] is False
    recipe["optimizer"] = "changed"
    assert settings(50, 42)["optimizer"] == "AdamW"
    assert RUN_OVERRIDES[NAMES[18]] == {"optimizer": "SGD"}
    baseline = deepcopy(FACTORS[NAMES[11]])
    assert baseline == FACTORS[NAMES[18]] == FACTORS[NAMES[19]]
    stage = JStage(16, 32, 3)
    assert len(stage.blocks) == 3


def test_i20_anchor_preserves_original_graph():
    original = yaml.safe_load((YAML_DIR.parent / "I_V2_series/I20_corrected_control.yaml").read_text(encoding="utf-8"))
    anchor = yaml.safe_load((YAML_DIR / "J01_i20_structure_anchor.yaml").read_text(encoding="utf-8"))
    for key in ("backbone", "head", "scales", "scale"):
        assert anchor.get(key) == original.get(key), key


def test_checkpoint_reloads_without_runtime_patches(tmp_path):
    model = YOLO(str(YAML_DIR / "J11_synthesis.yaml"), task="segment", verbose=False)
    model.save(tmp_path / "model.pt")
    loaded = YOLO(str(tmp_path / "model.pt"), task="segment", verbose=False)
    loaded.model.float().eval()
    with torch.no_grad():
        output, _ = loaded.model(torch.rand(1, 3, 96, 128))
    assert output[1].shape == (1, 32, 48, 64)


def test_official_flops_respects_rgb_rearrangement_contract():
    model = YOLO(str(YAML_DIR / "J11_synthesis.yaml"), task="segment", verbose=False).model
    assert model.yaml["channels"] == 3 and model.model[0].project.conv.in_channels == 20
    assert 4 < get_flops(model, 640) < 8
