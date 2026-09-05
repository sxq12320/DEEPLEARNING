"""Official API, independent ablations, geometry correctness and legacy preservation."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from citrus_sage_v4r_suite import NAMES, SUITES, YAML_DIR, select_names
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2_Faster, SAGEGatedStage, SegmentCitrusSAGEV4R
from ultralytics.nn.modules.citrus_sage_v4r import SAGEMaskCorrection
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.sage_v4r_loss import geometry_from_bce, instance_geometry_regions
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]


def example_batch(empty=False, overlap=True):
    masks = torch.zeros(1, 64, 64)
    if not empty:
        masks[0, 5:8, 5:8] = 1  # Too small for a meaningful extra boundary target.
        masks[0, 18:43, 14:34] = 2
        masks[0, 27:34, 27:34] = 0  # Visible concavity, NOT a hole to fill.
        masks[0, 18:43, 34:54] = 3
    if not overlap:
        masks = (masks[0] == torch.arange(1, 1 if empty else 4)[:, None, None]).float()
    return {
        "img": torch.rand(1, 3, 256, 256),
        "masks": masks,
        "batch_idx": torch.zeros(0 if empty else 3),
        "cls": torch.zeros(0 if empty else 3, 1),
        "bboxes": torch.zeros(0, 4)
        if empty
        else torch.tensor(
            [
                [6.5 / 64, 6.5 / 64, 3 / 64, 3 / 64],
                [24 / 64, 30.5 / 64, 20 / 64, 25 / 64],
                [44 / 64, 30.5 / 64, 20 / 64, 25 / 64],
            ]
        ),
    }


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", NAMES)
def test_every_yaml_public_api_forward_backward_cost(name):
    path = YAML_DIR / f"{name}.yaml"
    api = YOLO(str(path), task="segment", verbose=False)
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(example_batch())
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    for key, parameter in model.named_parameters():
        if any(term in key for term in ("refiner", "detail_to_proto", "detail_scale", "model.6.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), key
    head = model.model[-1]
    if isinstance(head, SegmentCitrusSAGEV4R):
        assert head.detail_to_proto.weight.grad.abs().sum() > 0
        assert (items[-1] > 0) == (head.boundary_gain + head.neighbor_gain > 0)
        assert model.criterion._geometry is None
    if "faster" in name:
        assert isinstance(model.model[6], C3k2_Faster)
    if "gated" in name:
        assert isinstance(model.model[6], SAGEGatedStage)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 160))
        assert output[0][1].shape == (1, 32, 32, 40)
        model.model[-1].export = True
        exported = model(torch.zeros(1, 3, 128, 160))
        assert exported[1].shape == (1, 32, 32, 40)
    params, flops = sum(p.numel() for p in model.parameters()), get_flops(model, 640)
    if name.startswith("SAGE4"):
        assert params < 2_350_000 and flops < 10.15
        assert model.model[-1].f[:3] == [16, 19, 10]
        assert all(isinstance(model.model[i], torch.nn.Identity) for i in (20, 21, 22))
    assert api.model is not None


def test_loss_factorial_and_backbone_are_not_confounded():
    configs = {name: yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text()) for name in NAMES[4:]}
    losses = [configs[name] for name in SUITES["geometry"]]
    assert {tuple(c["head"][-1][-1][-2:]) for c in losses} == {(0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1)}
    for config in losses:
        assert config["backbone"] == losses[0]["backbone"]
        assert config["head"][:-1] == losses[0]["head"][:-1]
    for name in SUITES["backbone"]:
        assert configs[name]["head"] == losses[0]["head"]
    assert select_names("screen") == list(NAMES[:5])
    with pytest.raises(ValueError):
        select_names("screen", NAMES[0] + "," + NAMES[0])


def test_geometry_preserves_concavity_and_ignores_unresolvable_boundary():
    ids = example_batch()["masks"][0]
    gt = (ids == torch.arange(1, 4)[:, None, None]).float()
    band, neighbors, eligible = instance_geometry_regions(gt, ids > 0)
    assert eligible.tolist() == [False, True, True]
    assert band[0].sum() == 0 and band[1].sum() > 0
    assert (neighbors * gt).sum() == 0 and neighbors[1].sum() > 0
    assert gt[1, 30, 30] == 0  # Nothing replaced the concave visible GT by its hull.
    assert (band[1, 27:34, 27:34]).sum() > 0
    logits = torch.zeros_like(gt, requires_grad=True)
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    bs, bc, ns, nc = geometry_from_bce(bce, gt, gt, torch.arange(3), ids > 0, (4, 4))
    (bs / bc + ns / nc).backward()
    assert logits.grad[1, 28, 34] > 0  # Gradient descent lowers intrusion into the neighbor fruit.
    assert logits.grad[1, 28, 27] > 0  # Visible notch remains negative.
    assert torch.all(logits.grad[0] == 0)  # Isolated tiny mask keeps ONLY the official loss.


def test_geometry_is_gt_permutation_and_positive_anchor_count_invariant():
    ids = example_batch()["masks"][0]
    gt = (ids == torch.arange(1, 4)[:, None, None]).float()
    bce = F.binary_cross_entropy_with_logits(torch.zeros_like(gt), gt, reduction="none")
    initial = geometry_from_bce(bce, gt, gt, torch.arange(3), ids > 0, (4, 4))
    inverse = torch.tensor([0, 1, 1, 1, 2])
    repeated = geometry_from_bce(bce[inverse], gt[inverse], gt, inverse, ids > 0, (4, 4))
    for a, b in zip(initial, repeated):
        torch.testing.assert_close(a, b)
    perm = torch.tensor([2, 0, 1])
    for a, b in zip(instance_geometry_regions(gt, ids > 0), instance_geometry_regions(gt[perm], ids > 0)):
        torch.testing.assert_close(a[perm], b)


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_geometry_criterion_matches_stock_main_loss(overlap, empty):
    model = SegmentationModel(YAML_DIR / f"{NAMES[7]}.yaml", nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "overlap_mask": overlap})
    batch = example_batch(empty=empty, overlap=overlap)
    preds = model(batch["img"])
    native = v8SegmentationLoss(model)
    revised = model.init_criterion()
    native_total, native_items = native(preds, batch)
    total, items = revised(preds, batch)
    torch.testing.assert_close(total[:4], native_total[:4], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(items[:4], native_items[:4], rtol=1e-5, atol=1e-5)
    total.sum().backward()
    assert torch.isfinite(total).all()
    assert (items[-1] == 0) == empty


def test_zero_detail_scale_preserves_native_proto():
    model = SegmentationModel(YAML_DIR / f"{NAMES[4]}.yaml", nc=1, verbose=False).eval()
    head = model.model[-1]
    head.detail_scale.data.zero_()
    features = [torch.randn(1, c, h, w) for c, h, w in [(64, 16, 20), (128, 8, 10), (256, 4, 5), (64, 32, 40)]]
    with torch.no_grad():
        expected = head.proto(features[0])
        actual = head(features)[0][1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reprojection_really_runs_once_and_gradients_reach_semantics():
    module = SAGEMaskCorrection(32, 64, 16, "reproject")
    calls = []
    hook = module.down_project.register_forward_hook(lambda *args: calls.append(1))
    detail = torch.randn(2, 32, 32, 40, requires_grad=True)
    semantic = torch.randn(2, 64, 16, 20, requires_grad=True)
    module(detail, semantic).square().mean().backward()
    hook.remove()
    assert len(calls) == 1
    assert detail.grad.abs().sum() > 0 and semantic.grad.abs().sum() > 0
    assert module.up_error.weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_and_pretrained_paths(tmp_path):
    checkpoint = ROOT / "yolo11n-seg.pt"
    if not checkpoint.exists():
        pytest.skip("Local checkpoint unavailable")
    source = YOLO(str(checkpoint), verbose=False)
    for name in NAMES[1:]:
        api = YOLO(str(YAML_DIR / f"{name}.yaml"), verbose=False).load(str(checkpoint))
        torch.testing.assert_close(
            api.model.model[-1].proto.cv1.conv.weight, source.model.model[-1].proto.cv1.conv.weight
        )
        torch.testing.assert_close(api.model.model[2].cv1.conv.weight, source.model.model[2].cv1.conv.weight)
    saved = tmp_path / "v4r.pt"
    api.save(str(saved))
    restored = YOLO(str(saved), verbose=False)
    with torch.no_grad():
        restored.model.eval()(torch.zeros(1, 3, 128, 128))
