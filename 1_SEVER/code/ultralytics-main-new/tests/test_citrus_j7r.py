"""J7R explicit observations and directed evidence paths; not AP or GPU claims."""

import copy
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import J7REvidenceRead, J7RHighEvidence, J7RPackedInput, SegmentCitrusJ7R
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA

ROOT = Path(__file__).resolve().parents[1]
YAMLS = tuple(path for index in range(110, 116)
              for path in (ROOT / "0_orange_yaml/J_V7R_series").glob("J{}_*.yaml".format(index)))


@pytest.fixture(autouse=True)
def small_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def model_for(number):
    path, = (ROOT / "0_orange_yaml/J_V7R_series").glob("J%d_*.yaml" % number)
    model = YOLO(str(path), task="segment", verbose=False).model
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    return model


def batch(empty=False):
    mask = torch.zeros(2, 64, 64)
    mask[:, 12:52, 12:52] = 1
    mask[:, 9:12, 9:12] = 2
    count = 0 if empty else 4
    return dict(img=torch.rand(2, 15, 128, 128, requires_grad=True),
                batch_idx=torch.tensor([0., 0., 1., 1.])[:count], cls=torch.zeros(count, 1),
                masks=mask * 0 if empty else mask,
                bboxes=torch.tensor([[.5, .5, .625, .625], [.1640625, .1640625, .046875, .046875]] * 2)[:count])


@pytest.mark.parametrize("path", YAMLS, ids=lambda p: p.stem)
def test_all_yaml_official_build_loss_backward_rectangular(path):
    model = YOLO(str(path), task="segment", verbose=False).model
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    sample = batch()
    losses, items = model.loss(sample)
    assert torch.isfinite(losses).all() and torch.isfinite(items).all()
    losses.sum().backward()
    assert torch.isfinite(sample["img"].grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert type(model.criterion) is v8SegmentationLoss
    assert model.stride.tolist() == [8., 16., 32.]
    head = model.model[-1]
    assert isinstance(head, SegmentCitrusJ7R) and not head.shared_towers
    assert not hasattr(head, "quality_predictor")
    assert head.tiny_dice_gain == head.boundary_gain == head.neighbor_gain == 0
    with torch.no_grad():
        prediction = model.eval()(torch.rand(1, 15, 128, 160))[0]
    assert prediction[0].shape[-1] == 420
    assert prediction[1].shape == (1, 32, 64, 80)
    assert sum(p.numel() for p in model.parameters()) < 2842803


@pytest.mark.parametrize("number", [110, 112, 113, 114, 115])
def test_empty_labels_finite(number):
    model = model_for(number)
    loss, items = model.loss(batch(empty=True))
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()


def test_no_implicit_rgb_fallback_and_phase_shapes():
    with pytest.raises(ValueError, match="packed15"):
        J7RPackedInput()(torch.rand(1, 3, 128, 128))
    with pytest.raises(ValueError, match="multiples of 32"):
        J7RPackedInput()(torch.rand(1, 15, 127, 128))
    values = J7RHighEvidence().eval()(torch.rand(1, 12, 128, 160))
    assert [tuple(x.shape) for x in values] == [(1, 16, 64, 80), (1, 16, 32, 40), (1, 16, 16, 20)]
    with pytest.raises(ValueError, match="aligned"):
        J7REvidenceRead([32, 16])([torch.rand(1, 32, 8, 8), torch.rand(1, 16, 9, 8)])


def test_zero_start_is_identity_but_two_steps_can_train_evidence():
    read = J7REvidenceRead([32, 16]).train()
    native, detail = torch.randn(2, 32, 8, 8), torch.randn(2, 16, 8, 8)
    torch.testing.assert_close(read([native, detail]), native, rtol=0, atol=0)
    optimizer = torch.optim.SGD(read.parameters(), lr=.1)
    read([native, detail]).square().mean().backward()
    assert read.gain.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    read([native, detail]).square().mean().backward()
    assert read.evidence.conv.weight.grad.abs().sum() > 0
    assert torch.isfinite(read.evidence.conv.weight.grad).all()


def test_final_proto_only_has_no_forward_path_to_instance_towers():
    model = model_for(113).eval()
    with torch.no_grad():
        model.model[-1].high_mask.gain.fill_(.4)
    first = torch.rand(1, 15, 128, 160)
    second = first.clone()
    second[:, 3:] = torch.rand_like(second[:, 3:])
    with torch.no_grad():
        out1, raw1 = model(first)
        out2, raw2 = model(second)
    for key in ("boxes", "scores", "mask_coefficient"):
        torch.testing.assert_close(raw1[key], raw2[key], rtol=0, atol=0)
    torch.testing.assert_close(out1[0], out2[0], rtol=0, atol=0)
    assert not torch.equal(raw1["proto"], raw2["proto"])
    torch.testing.assert_close(out1[1], raw1["proto"], rtol=0, atol=0)


def test_early_high_changes_semantic_input_not_a_second_instance_source():
    model = model_for(112).eval()
    read, = (m for m in model.model if isinstance(m, J7REvidenceRead))
    with torch.no_grad():
        read.gain.fill_(.4)
    observed = []
    hook = read.register_forward_hook(lambda _m, _x, out: observed.append(out.clone()))
    first = torch.rand(1, 15, 128, 160)
    second = first.clone()
    second[:, 3:] = torch.rand_like(second[:, 3:])
    with torch.no_grad():
        model(first)
        model(second)
    hook.remove()
    assert not torch.equal(observed[0], observed[1])
    assert model.model[-1].nl == 3


@pytest.mark.parametrize("number", [112, 114])
def test_fuse_and_ema_keep_complete_tensor_graph(number):
    model = model_for(number).eval()
    # Exercise a nonzero evidence contribution, not just the initial zero-gain identity.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, J7REvidenceRead):
                module.gain.fill_(.25)
    ema = ModelEMA(model)
    ema.update(model)
    fused = copy.deepcopy(model).fuse(verbose=False)
    image = torch.rand(1, 15, 128, 160)
    with torch.no_grad():
        expected, actual = model(image)[0], fused(image)[0]
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    assert ema.ema.yaml["channels"] == 15


def test_native_save_reload_and_reject_unverified_export(tmp_path):
    path, = (ROOT / "0_orange_yaml/J_V7R_series").glob("J114_*.yaml")
    wrapper = YOLO(str(path), task="segment", verbose=False)
    checkpoint = tmp_path / "j7r_contract.pt"
    wrapper.save(str(checkpoint))
    restored = YOLO(str(checkpoint), task="segment", verbose=False).model.eval()
    assert restored.yaml["channels"] == 15
    rounded = copy.deepcopy(wrapper.model).half().float().eval()
    image = torch.rand(1, 15, 128, 160)
    with torch.no_grad():
        for a, b in zip(rounded(image)[0], restored(image)[0]):
            torch.testing.assert_close(a, b)
    restored.model[-1].export = True
    with pytest.raises(ValueError, match="export"):
        restored(image)
