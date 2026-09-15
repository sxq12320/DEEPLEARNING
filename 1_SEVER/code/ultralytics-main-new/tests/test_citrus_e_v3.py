"""Independent tests; no dependency on user-deleted reports or local training outputs."""

import importlib.util
from copy import deepcopy

import pytest
import torch
import yaml

from citrus_e_v3_suite import FACTORS, NAMES, ROOT, YAML_DIR
from scripts.generate_citrus_e_v3_yaml import configs
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_e_v3_loss import quality_targets
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(empty=False):
    masks = torch.zeros(2, 32, 32)
    masks[0, 7:23, 7:23] = 1
    masks[1, 9:25, 9:25] = 1
    return dict(img=torch.rand(2,3,128,128), batch_idx=torch.tensor([0.,1.])[:0 if empty else 2],
                cls=torch.zeros(0 if empty else 2,1),
                bboxes=torch.tensor([[.46875,.46875,.5,.5],[.53125,.53125,.5,.5]])[:0 if empty else 2],
                masks=masks * (not empty))


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
@pytest.mark.parametrize("empty", [False, True])
def test_api_backward_and_fused_inference(name, factors, empty):
    path = YAML_DIR / (name + ".yaml")
    assert YOLO(str(path), verbose=False).task == "segment"
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(batch(empty))
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if factors[2] and not empty:
        assert items[-1] > 0
        assert model.model[-1].quality_predictor[0][-1].weight.grad.abs().sum() > 0
    else:
        assert items[-1] == 0
    # Existing E20/E30 control is 10.097 GFLOPs here; quality-only adds 0.033.
    assert get_flops(model, 640) < 10.15
    with torch.inference_mode():
        model.eval()
        x = torch.rand(1,3,128,160)
        expected = model(x)[0]
        actual = deepcopy(model).fuse(verbose=False)(x)[0]
        torch.testing.assert_close(expected[0], actual[0], rtol=.002, atol=.02)
        torch.testing.assert_close(expected[1], actual[1], rtol=.002, atol=.02)


def test_factors_and_yaml_manifest():
    assert configs() == [yaml.safe_load((YAML_DIR/(n+".yaml")).read_text()) for n in NAMES]
    assert configs()[0] == yaml.safe_load((ROOT/"0_orange_yaml/E_V2_series/E20_fixed_control.yaml").read_text())
    assert FACTORS == list(dict.fromkeys(FACTORS)) and len(FACTORS) == 8


@pytest.mark.parametrize("name", NAMES)
def test_initialization_and_checkpoint(name, tmp_path):
    model = YOLO(str(YAML_DIR/(name+".yaml")), verbose=False).load(str(ROOT/"yolo11n-seg.pt"))
    source = YOLO(str(ROOT/"yolo11n-seg.pt"), verbose=False).model.state_dict()
    for key in ("model.2.cv1.conv.weight", "model.4.cv2.conv.weight", "model.23.cv4.0.0.conv.weight"):
        torch.testing.assert_close(model.model.state_dict()[key], source[key], rtol=0, atol=0)
    filename = tmp_path/"roundtrip.pt"
    model.save(str(filename))
    loaded = YOLO(str(filename), verbose=False)
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=.001, atol=.001)


def test_quality_target_uses_predicted_boxes_and_detaches():
    prototype = torch.ones(1,8,8,requires_grad=True)
    coefficients = torch.ones(2,1,requires_grad=True)
    gt = torch.ones(2,8,8)
    boxes = torch.tensor([[0,0,8,8],[0,0,4,8]],dtype=torch.float)
    target = quality_targets(coefficients,prototype,boxes,gt)
    torch.testing.assert_close(target,torch.tensor([1.,.5]))
    assert not target.requires_grad


def test_calibration_toggle_changes_only_scores():
    model = SegmentationModel(YAML_DIR/(NAMES[3]+".yaml"),nc=1,verbose=False).eval()
    head = model.model[-1]
    with torch.inference_mode():
        x = torch.rand(1,3,128,128)
        calibrated = model(x)[0]
        head.quality_calibration = False
        raw = model(x)[0]
    torch.testing.assert_close(calibrated[0][:,:4],raw[0][:,:4])
    torch.testing.assert_close(calibrated[0][:,5:],raw[0][:,5:])
    torch.testing.assert_close(calibrated[1],raw[1])
    assert (calibrated[0][:,4] < raw[0][:,4]).all()


def test_runner_contract():
    spec = importlib.util.spec_from_file_location("ev3_runner_test", ROOT/"20260909_citrus_e_v3_batch.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.NAMES == NAMES and not runner.GUIDED
    assert set(runner.TILE_PROBABILITY.values()) == {.5}


def test_quality_preserves_overlap_instance_ids(monkeypatch):
    model = SegmentationModel(YAML_DIR/(NAMES[3]+".yaml"),nc=1,verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    criterion = model.init_criterion()
    # ID 1 vanished in mask raster; surviving ID 2 must still match target index 1.
    criterion._assignment = (torch.ones(1,2,dtype=torch.bool),torch.tensor([[0,1]]),None,
                             torch.zeros(2,2),torch.ones(2,1))
    monkeypatch.setattr(criterion,"bbox_decode",lambda *args:torch.tensor([[[0.,0,8,8],[0.,0,8,8]]]))
    preds = dict(ev3_quality=torch.zeros(1,1,2,requires_grad=True), proto=torch.ones(1,1,8,8),
                 mask_coefficient=torch.ones(1,1,2), boxes=torch.zeros(1,64,2))
    data = dict(img=torch.zeros(1,3,8,8),masks=torch.full((1,8,8),2.),batch_idx=torch.tensor([0.,0.]))
    loss = criterion._quality_loss(preds,data)
    loss.backward()
    torch.testing.assert_close(loss,torch.tensor(.25))
    assert preds["ev3_quality"].grad[0,0,0] == 0
    assert preds["ev3_quality"].grad[0,0,1] != 0
