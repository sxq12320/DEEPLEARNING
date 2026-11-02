"""Raw and explicit-tensor J7R input contracts; no new decoder or metric."""

from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from citrus_j7r_data import J7R_INPUT_PROTOCOL
from citrus_j7r_predict import J7RPredictor
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.segment import SegmentationPredictor


def predictor(observation="raw"):
    instance = object.__new__(J7RPredictor)
    instance.model = SimpleNamespace(format="pt", fp16=False, stride=32,
                                    yaml={"channels": 15, "citrus_input_protocol": J7R_INPUT_PROTOCOL,
                                          "citrus_observation": observation})
    instance.args = SimpleNamespace(augment=False, compile=False, embed=None, rect=False)
    instance.imgsz = [640, 640]
    instance.device = torch.device("cpu")
    return instance


def test_raw_prediction_uses_original_pixels_and_global_affine():
    yy, xx = np.indices((1021, 1777))
    raw = ((xx + yy) % 2 * 255).astype(np.uint8)[..., None].repeat(3, -1)
    actual = predictor().preprocess([raw])
    assert actual.shape == (1, 15, 640, 640)
    high = F.pixel_shuffle(actual[:, 3:], 2)[0].permute(1, 2, 0).numpy()[..., ::-1]
    letterbox = LetterBox([640, 640], auto=False, stride=32)
    params = letterbox.get_params({"img": raw})
    width, height = params["new_unpad"]
    left, top = params["left"], params["top"]
    expected = cv2.resize(raw, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR) / 255
    np.testing.assert_allclose(high[top * 2:(top + height) * 2, left * 2:(left + width) * 2], expected, atol=1e-7)
    pseudo = F.interpolate(actual[:, :3], scale_factor=2, mode="bilinear", align_corners=False)
    assert (F.pixel_shuffle(actual[:, 3:], 2) - pseudo).abs().mean() > .001


@pytest.mark.parametrize("observation", ["upsampled", "global"])
def test_observation_controls_are_explicit(observation):
    raw = np.full((317, 563, 3), 100, dtype=np.uint8)
    actual = predictor(observation).preprocess([raw])
    if observation == "global":
        assert torch.count_nonzero(actual[:, 3:]) == 0
    else:
        torch.testing.assert_close(F.pixel_shuffle(actual[:, 3:], 2),
                                   F.interpolate(actual[:, :3], scale_factor=2, mode="bilinear", align_corners=False))


def test_explicit_tensor_is_not_normalized_twice_or_repacked():
    packed = torch.rand(1, 15, 32, 64)
    actual = predictor().preprocess(packed)
    torch.testing.assert_close(actual, packed, rtol=0, atol=0)
    assert not any("pending" in name for name in predictor().__dict__)


@pytest.mark.parametrize("value", [torch.rand(1, 3, 32, 32), torch.rand(1, 15, 31, 32),
                                   torch.full((1, 15, 32, 32), 255, dtype=torch.uint8),
                                   torch.full((1, 15, 32, 32), float("nan")),
                                   torch.full((1, 15, 32, 32), 1.1)])
def test_invalid_tensors_rejected(value):
    with pytest.raises(ValueError):
        predictor().preprocess(value)


@pytest.mark.parametrize("name,value", [("augment", True), ("compile", True), ("embed", [0])])
def test_unsupported_rgb_tta_paths_rejected(name, value):
    instance = predictor()
    setattr(instance.args, name, value)
    with pytest.raises(ValueError):
        instance.preprocess(torch.rand(1, 15, 32, 32))


def test_tensor_results_display_only_global_rgb(monkeypatch):
    observed = {}

    def postprocess(self, preds, img, orig_imgs):
        observed.update(model_input=img, displayed=orig_imgs)
        return "ok"

    monkeypatch.setattr(SegmentationPredictor, "postprocess", postprocess)
    packed = torch.rand(1, 15, 32, 32)
    assert predictor().postprocess(None, packed, packed) == "ok"
    assert observed["model_input"].shape[1] == 15 and observed["displayed"].shape[1] == 3
    assert packed.shape[1] == 15


def test_actual_model_native_predict_original_and_explicit_tensor():
    from ultralytics import YOLO

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "0_orange_yaml/J_V7R_series/J112_raw_c3_evidence.yaml"), task="segment", verbose=False)
    raw = np.full((317, 563, 3), 100, dtype=np.uint8)
    results = model.predict(source=[raw], predictor=J7RPredictor, device="cpu", imgsz=640,
                            conf=.99, save=False, verbose=False)
    assert results[0].orig_img.shape == (317, 563, 3)
    results = model.predict(source=torch.rand(1, 15, 64, 96), predictor=J7RPredictor, device="cpu", imgsz=640,
                            conf=.99, save=False, verbose=False)
    assert results[0].orig_img.shape == (64, 96, 3)
    with pytest.raises(ValueError):
        model.predict(source=torch.rand(1, 3, 64, 96), predictor=J7RPredictor, device="cpu", imgsz=640,
                      conf=.99, save=False, verbose=False)
