"""Check native semantics, zero-start learning and the exact two-factor model family."""

from copy import deepcopy

import pytest
import torch
import yaml

import RUN_CITRUS_MINIMAL_ABLATION as suite
import RUN_WHOLE_PRETRAIN_COMPARE as engine
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_minimal_relay import ShallowEvidenceRelay
from ultralytics.utils.loss import v8SegmentationLoss

torch.set_num_threads(1)


def config(name):
    return yaml.safe_load((suite.ROOT / suite.MODELS[name]).read_text(encoding="utf-8"))


def test_exact_factor_graphs_and_official_origin():
    original = engine.model_config("B00_yolo11n")
    assert config("R00_baseline") == original
    for a, b, name in ((1, 0, "R01_asym"), (0, 1, "R02_relay"), (1, 1, "R03_asym_relay")):
        expected = deepcopy(original)
        if a:
            expected["head"][9:12] = [[-1, 1, "nn.Identity", []] for _ in range(3)]
            expected["head"][-1][0][2] = 10
        if b:
            expected["head"][-1][0].append(2)
            expected["head"][-1][2] = "SegmentCitrusMinimal"
            expected["head"][-1][3].append(8)
        assert config(name) == expected


@pytest.mark.parametrize("base,candidate", [("R00_baseline", "R02_relay"), ("R01_asym", "R03_asym_relay")])
def test_zero_start_preserves_native_predictions_and_fusing(base, candidate):
    native = YOLO(str(suite.ROOT / suite.MODELS[base]), verbose=False).model.eval()
    model = YOLO(str(suite.ROOT / suite.MODELS[candidate]), verbose=False).model.eval()
    missing = model.load_state_dict(native.state_dict(), strict=False)
    assert missing.missing_keys and all(".relay." in k for k in missing.missing_keys)
    assert not missing.unexpected_keys
    image = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        expected = native(image)[0]
        actual = model(image)[0]
        assert actual[0].shape == (1, 37, 8400)
        assert actual[1].shape == (1, 32, 160, 160)
        assert model.stride.tolist() == [8, 16, 32]
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        # Exercise a genuinely active relay before checking Conv-BN fusion.
        model.model[-1].relay.gain.fill_(0.7)
        active = model(image)[0]
        fused = model.fuse(verbose=False)(image)[0]
        for x, y in zip(active, fused):
            torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-4)


def test_zero_gain_can_start_learning_and_gate_can_reject_whole_location():
    relay = ShallowEvidenceRelay(64, 64).eval()
    detail, prediction = torch.randn(2, 64, 16, 16), torch.randn(2, 64, 8, 8)
    optimizer = torch.optim.SGD(relay.parameters(), lr=1)
    target = torch.randn_like(prediction)
    for step in range(3):
        optimizer.zero_grad()
        loss = (relay(detail, prediction) - target).square().mean()
        loss.backward()
        assert relay.gain.grad.isfinite().all() and relay.gain.grad.abs().sum() > 0
        if step >= 1:
            for p in (relay.project.weight, relay.detail.conv.weight, relay.gate.weight):
                assert p.grad.isfinite().all() and p.grad.abs().sum() > 0
        if step >= 2:
            assert relay.context.conv.weight.grad.abs().sum() > 0
        optimizer.step()
    with torch.no_grad():
        relay.gain.fill_(1)
        relay.gate.weight.zero_()
        relay.gate.bias.fill_(30)
        opened = relay(detail, prediction)
        relay.gate.bias.fill_(-30)
        closed = relay(detail, prediction)
        assert (opened - prediction).abs().max() > 1e-4
        torch.testing.assert_close(closed, prediction, rtol=0, atol=1e-7)


@pytest.mark.parametrize("name", list(suite.MODELS))
def test_native_loss_backward(name):
    model = YOLO(str(suite.ROOT / suite.MODELS[name]), verbose=False).model.train()
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    assert type(model.init_criterion()) is v8SegmentationLoss
    masks = torch.zeros(2, 32, 32)
    masks[:, 8:24, 8:24] = 1
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.zeros(2, 1), "bboxes": torch.tensor([[.5, .5, .5, .5]] * 2), "masks": masks}
    loss, components = model.loss(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    assert model.model[0].conv.weight.grad.isfinite().all()
    if name in ("R02_relay", "R03_asym_relay"):
        assert model.model[-1].relay.gain.grad.abs().sum() > 0


def test_four_arm_defaults_and_original_runner_still_independent():
    args = engine.arguments([], model_catalog=suite.MODELS, default_models=list(suite.MODELS))
    assert args.default_models == list(suite.MODELS)
    assert list(engine.arguments([]).model_catalog)[0] == "B00_yolo11n"
    hashes = engine.source_fingerprint([suite.__file__])
    assert "RUN_CITRUS_MINIMAL_ABLATION.py" in hashes
