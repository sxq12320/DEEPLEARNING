"""Early evidence is a testable interface, not an accuracy or lossless-recovery claim."""

from pathlib import Path

import pytest
import torch

from citrus_j7r_initialization import initialize_native_pair
from ultralytics import YOLO
from ultralytics.nn.modules import J7REvidenceRead, J7RStemEvidence, Segment
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(number):
    path, = (ROOT / "0_orange_yaml/J_V7R_series").glob("J{}_*.yaml".format(number))
    torch.manual_seed(42)
    model = YOLO(str(path), task="segment", verbose=False).model
    initialize_native_pair(model, 42)
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    return model


@pytest.mark.parametrize("number", range(119, 124))
def test_forward_backward_and_rectangular(number):
    model = build(number)
    assert type(model.model[-1]) is Segment
    assert model.stride.tolist() == [8., 16., 32.]
    masks = torch.zeros(2, 64, 64)
    masks[:, 16:48, 16:48] = 1
    batch = {"img": torch.rand(2, 15, 128, 128), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.zeros(2, 1), "masks": masks, "bboxes": torch.tensor([[.5, .5, .5, .5]] * 2)}
    loss, items = model.loss(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        _, raw = model.eval()(torch.rand(1, 15, 128, 160))
    assert raw["proto"].shape == (1, 32, 32, 40)


def test_same_native_initialization_and_initial_function():
    models = [build(i).eval() for i in (116, 119, 120, 121, 122, 123)]
    assert len({m.j7r_native_initialization["common_state_sha256"] for m in models}) == 1
    image = torch.rand(1, 15, 128, 160)
    with torch.no_grad():
        outputs = [m(image)[1] for m in models]
    for output in outputs[1:]:
        for key in ("boxes", "scores", "proto", "mask_coefficient"):
            torch.testing.assert_close(output[key], outputs[0][key], rtol=0, atol=0)
    for a, b in ((2, 5), (3, 4)):
        for key, tensor in models[a].state_dict().items():
            torch.testing.assert_close(tensor, models[b].state_dict()[key], rtol=0, atol=0)


def test_real_evidence_reaches_stem_after_gain_learns():
    model = build(120).eval()
    read, = (m for m in model.modules() if isinstance(m, J7REvidenceRead))
    with torch.no_grad():
        read.gain.fill_(.2)
    image = torch.rand(1, 15, 128, 160, requires_grad=True)
    observed = []
    handle = read.register_forward_hook(lambda _m, _a, value: observed.append(value))
    model(image)
    handle.remove()
    observed[0].square().sum().backward()
    assert observed[0].shape == (1, 16, 64, 80)
    assert image.grad[:, 3:].abs().sum() > 0


def test_phase_center_is_orthogonal_projection_not_new_information():
    x = torch.randn(2, 12, 16, 20, dtype=torch.float64)
    q = J7RStemEvidence.center_phases
    y = q(x)
    torch.testing.assert_close(q(y), y)
    torch.testing.assert_close(y.reshape(2, 3, 4, 16, 20).mean(2), torch.zeros(2, 3, 16, 20, dtype=x.dtype))
    offset = torch.randn(2, 3, 1, 16, 20, dtype=x.dtype).expand(-1, -1, 4, -1, -1).reshape_as(x)
    torch.testing.assert_close(q(x + offset), y)
    assert y.norm() <= x.norm()
    # For each 4-phase block, WQ is simply another raw pointwise linear map.
    matrix = torch.eye(4, dtype=x.dtype) - torch.ones(4, 4, dtype=x.dtype) / 4
    w, phases = torch.randn(5, 4, dtype=x.dtype), x.reshape(2, 3, 4, -1)
    torch.testing.assert_close(w @ (matrix @ phases), (w @ matrix) @ phases)
