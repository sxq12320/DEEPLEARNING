"""Contract checks for the independent I_V4 segmenter, not accuracy claims."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "I48_native_set", "I49_native_evidence", "I50_native_scene",
    "I51_native_fallback", "I52_native_review", "I53_native_edge",
)


@pytest.mark.parametrize("name", NAMES)
def test_native_yaml_forward_backward_and_inference(name):
    path = ROOT / "0_orange_yaml/I_V4_series" / (name + ".yaml")
    wrapper = YOLO(str(path))
    assert wrapper.task == "segment"
    model = wrapper.model
    image = torch.randn(1, 3, 64, 64)
    raster = torch.zeros(1, 16, 16)
    raster[0, 2, 3] = 1  # one tiny visible fruit
    raster[0, 9:13, 8:12] = 2
    batch = {
        "img": image, "masks": raster, "batch_idx": torch.zeros(2, 1),
        "cls": torch.zeros(2, 1), "bboxes": torch.rand(2, 4),
    }
    model.train()
    loss, items = model(batch)
    assert loss.isfinite() and items.shape == (5,)
    loss.backward()
    assert model.model[-1].pixel.weight.grad is not None
    model.eval()
    with torch.no_grad():
        prediction, proto, raw = model(image)
    assert prediction.shape == (1, 5 + model.model[-1].nm, model.model[-1].nm)
    assert proto.shape == raw["masks"].shape
    identity = prediction[:, 5:].transpose(1, 2)
    assert torch.allclose(torch.bmm(identity, proto.flatten(2)), proto.flatten(2))


def test_native_empty_image_loss():
    model = YOLO(str(ROOT / "0_orange_yaml/I_V4_series/I52_native_review.yaml"), task="segment").model
    batch = {"img": torch.randn(1, 3, 64, 64), "masks": torch.zeros(1, 16, 16),
             "batch_idx": torch.zeros(0, 1), "cls": torch.zeros(0, 1), "bboxes": torch.zeros(0, 4)}
    model.train()
    loss, _ = model(batch)
    assert loss.isfinite()
    loss.backward()
