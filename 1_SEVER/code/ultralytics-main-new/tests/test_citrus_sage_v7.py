"""V7 public API, anchor geometry, gradients and inherited compatibility."""

from copy import deepcopy

import pytest
import torch
import yaml

from citrus_sage_v7_suite import NAMES, ROOT, YAML_DIR
from tests.test_citrus_sage_v4r import example_batch
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_public_build_loss_backward_and_strides(name, empty):
    api = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment", verbose=False)
    assert api.task == "segment"
    model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(example_batch(empty=empty))
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert items[-1] == 0  # no hidden auxiliary loss
    head = model.model[-1]
    expected = [4, 8, 16, 32] if name in NAMES[2:] else [8, 16, 32]
    assert model.stride.tolist() == expected
    assert head.npr == 64
    if name != NAMES[0]:
        assert not hasattr(head, "cv2")
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.candidate_stems.parameters())
    with torch.inference_mode():
        model.eval()
        y, raw = model(torch.rand(1, 3, 128, 160))
        anchors = sum((128 // s) * (160 // s) for s in expected)
        assert y[0].shape == (1, 37, anchors)
        assert y[1].shape == (1, 32, 32, 40)
        assert raw["mask_coefficient"].shape[-1] == anchors
        head.export = True
        assert model(torch.rand(1, 3, 160, 128))[1].shape[-2:] == (40, 32)
    assert get_flops(model, 640) < 10.2


@pytest.mark.parametrize("name", NAMES[1:])
def test_pretraining_rebuild_and_checkpoint(name, tmp_path):
    api = YOLO(str(YAML_DIR / f"{name}.yaml"), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    rebuilt = SegmentationModel(api.model.yaml, nc=80, verbose=False)
    rebuilt.load(api.model, verbose=False)
    for key, value in api.model.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    path = tmp_path / "new.pt"
    api.save(str(path))
    restored = YOLO(str(path), verbose=False).model.eval()
    with torch.inference_mode():
        expected = api.model.eval()(torch.zeros(1, 3, 128, 128))[0][0]
        actual = restored(torch.zeros(1, 3, 128, 128))[0][0]
        torch.testing.assert_close(expected, actual, rtol=0.003, atol=0.03)  # saved checkpoint FP16
        fused = deepcopy(restored).fuse(verbose=False)
        torch.testing.assert_close(actual, fused(torch.zeros(1, 3, 128, 128))[0][0], rtol=0.001, atol=0.01)


def test_factors_and_direct_p2_gradient():
    configs = [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    assert configs[0] == yaml.safe_load((ROOT / "0_orange_yaml/SAGE_V6_series/SAGE60_relay_control.yaml").read_text())
    assert all(c["backbone"] == configs[0]["backbone"] for c in configs)
    assert all(c["head"][:-1] == configs[0]["head"][:-1] for c in configs)
    model = SegmentationModel(YAML_DIR / f"{NAMES[2]}.yaml", nc=1, verbose=False)
    pred = model(torch.rand(2, 3, 128, 128))
    pred["scores"][..., : 32 * 32].square().mean().backward()
    assert model.model[-1].refiner.detail[0].conv.weight.grad.abs().sum() > 0
    assert model.model[-1].candidate_classes[0][0].weight.grad.abs().sum() > 0


def test_foreground_import_inert():
    import RUN_SAGE_V7 as run
    from citrus_foreground import RUNNERS

    assert run.EPOCHS == 300 and not run.DRY_RUN
    assert (ROOT / RUNNERS["SAGE_V7"].script).exists()
