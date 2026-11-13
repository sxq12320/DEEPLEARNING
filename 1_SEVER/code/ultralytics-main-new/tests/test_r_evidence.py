"""Verify factor isolation, native loss and active contrast/semantic fusion."""
from copy import deepcopy

import pytest
import torch
import yaml

import RUN_R_EVIDENCE as suite
import RUN_WHOLE_PRETRAIN_COMPARE as engine
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import v8SegmentationLoss

ARMS = list(suite.MODELS)[:8]
FACTORS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
           (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
torch.set_num_threads(1)


@pytest.mark.parametrize("name,factors", zip(ARMS, FACTORS))
def test_only_three_factors(name, factors):
    a, b, c = factors
    expected = deepcopy(engine.model_config("B00_yolo11n"))
    if a:
        expected["head"][9:12] = [[-1, 1, "nn.Identity", []] for _ in range(3)]
        expected["head"][-1][0][2] = 10
    if b:
        expected["backbone"][0][2] = "EV10ContrastStem"
    if c:
        expected["head"][-1][2] = "SegmentCitrusE12Simple"
        expected["head"][-1][3].extend([False, True])
    actual = yaml.safe_load((suite.ROOT / suite.MODELS[name]).read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("name", ARMS)
def test_native_segmentation_loss_can_train(name):
    model = YOLO(str(suite.ROOT / suite.MODELS[name]), verbose=False).model.train()
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    assert type(model.init_criterion()) is v8SegmentationLoss
    masks = torch.zeros(2, 32, 32)
    masks[:, 6:26, 6:26] = 1
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.zeros(2, 1), "bboxes": torch.tensor([[.5, .5, .625, .625]] * 2), "masks": masks}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(2):
        optimizer.zero_grad()
        loss, components = model.loss(batch)
        assert torch.isfinite(loss).all() and torch.isfinite(components).all()
        loss.sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert parameter.grad.isfinite().all()
        stem, head = model.model[0], model.model[-1]
        if hasattr(stem, "structure"):
            assert stem.structure.gain.grad.abs().sum() > 0
            if step:
                assert stem.structure.conv.weight.grad.abs().sum() > 0
        if hasattr(head, "semantic"):
            for semantic in head.semantic:
                assert semantic.gain.grad.abs().sum() > 0
                if step:
                    assert semantic.expand.weight.grad.abs().sum() > 0
        optimizer.step()


def test_zero_start_and_active_fusion():
    native = YOLO(str(suite.ROOT / suite.MODELS["RE01_asym"]), verbose=False).model.eval()
    model = YOLO(str(suite.ROOT / suite.MODELS["RE07_full"]), verbose=False).model.eval()
    missing = model.load_state_dict(native.state_dict(), strict=False)
    assert not missing.unexpected_keys
    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        expected, actual = native(image)[0], model(image)[0]
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        model.model[0].structure.gain.fill_(0.5)
        for module in model.model[-1].semantic:
            module.gain.fill_(0.5)
        before = model(image)[0]
        after = model.fuse(verbose=False)(image)[0]
        for x, y in zip(before, after):
            torch.testing.assert_close(x, y, rtol=3e-4, atol=3e-4)


def test_single_data_and_recipe_defaults():
    assert "grouped_dedup" in suite.DATA and "min8" not in suite.DATA
    assert suite.CACHE is True and suite.AMP is False and suite.EPOCHS == 300
    assert len(suite.MODELS) == 10
    assert engine.model_config("REF_V12_06", suite.MODELS)["head"][-1][2] == "SegmentCitrusEV12"
