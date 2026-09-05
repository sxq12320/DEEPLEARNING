"""Native YAML, gradients, cache enumeration and geometric-target regression tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.data.base import BaseDataset
from ultralytics.nn.modules.citrus_sage_v4 import BoundedScaleUpdate, SAGEGatedStage, SegmentCitrusSAGEV4
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.sage_v4_loss import normalized_structure_loss, structure_targets
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
YAMLS = sorted((ROOT / "0_orange_yaml" / "SAGE_series").glob("SAGE3*.yaml"))


def synthetic_batch(empty=False):
    masks = torch.zeros(1, 64, 64)
    if not empty:
        masks[0, 5:11, 6:12] = 1
        masks[0, 18:43, 14:34] = 2
        masks[0, 27:34, 27:34] = 0
        masks[0, 17:44, 34:54] = 3
        masks[0, 29:36, 34:40] = 0
    return {
        "img": torch.rand(1, 3, 256, 256),
        "masks": masks,
        "batch_idx": torch.zeros(0 if empty else 3),
        "cls": torch.zeros(0 if empty else 3, 1),
        "bboxes": torch.zeros(0, 4)
        if empty
        else torch.tensor([[0.14, 0.13, 0.1, 0.1], [0.38, 0.48, 0.31, 0.39], [0.69, 0.48, 0.31, 0.42]]),
    }


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("path", YAMLS, ids=lambda path: path.stem)
def test_build_forward_backward_flops(path):
    assert len(YAMLS) == 6
    api = YOLO(str(path), task="segment", verbose=False)
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    total, components = model.loss(synthetic_batch())
    assert torch.isfinite(total).all() and torch.isfinite(components).all()
    total.sum().backward()
    for name, parameter in model.named_parameters():
        if any(key in name for key in ("detail", "update3", "update4", "structure", ".blocks.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    head = model.model[-1]
    if isinstance(head, SegmentCitrusSAGEV4):
        assert head.detail_to_proto.weight.grad.abs().sum() > 0
        assert (components[-1] > 0) == (head.structure_gain > 0)
    if "gated" in path.stem:
        assert isinstance(model.model[6], SAGEGatedStage)
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 128, 160))
        assert output[0][1].shape[-2:] == (32, 40)
    assert sum(p.numel() for p in model.parameters()) < 2_900_000
    assert 10 < get_flops(model, 640) < 10.7
    assert api.model is not None


def test_bounded_update_and_zero_scale_identity():
    update = BoundedScaleUpdate(8)
    measurement, prediction = torch.randn(2, 8, 16, 16), torch.randn(2, 8, 16, 16)
    output = update(measurement, prediction)
    torch.testing.assert_close(output, (measurement + prediction) / 2)
    assert (output >= torch.minimum(measurement, prediction) - 1e-6).all()
    assert (output <= torch.maximum(measurement, prediction) + 1e-6).all()


def test_structure_is_multilabel_and_separator_avoids_deep_interior():
    batch = synthetic_batch()
    target, active = structure_targets(batch["masks"], batch["batch_idx"], 1, (64, 64))
    fruit, boundary, separator = target.unbind(1)
    assert (fruit * boundary).sum() > 0  # legal overlap, never conflicting softmax classes
    assert separator.sum() > 0
    assert (separator * fruit * (1 - boundary)).sum() == 0
    assert torch.all(target <= active)
    logits = torch.zeros_like(target, requires_grad=True)
    loss = normalized_structure_loss(logits, target, active)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and 0 < loss < 2


def test_empty_structure_loss_does_not_scale_with_image_area():
    values = []
    for size in (32, 64, 160):
        logits = torch.zeros(2, 3, size, size, requires_grad=True)
        target, active = structure_targets(torch.zeros(2, size, size), torch.zeros(0), 2, (size, size))
        loss = normalized_structure_loss(logits, target, active)
        loss.backward()
        values.append(loss.detach())
        assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(values[0], values[-1])


def test_empty_batch_backward():
    model = SegmentationModel(YAMLS[4], nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    total, _ = model.loss(synthetic_batch(empty=True))
    total.sum().backward()
    assert torch.isfinite(total).all()
    assert model.model[-1].structure.weight.grad is not None


def test_rgb_cache_dedup_keeps_npy_sources_rgbd_and_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr("ultralytics.data.base.check_file_speeds", lambda *args, **kwargs: None)
    for name in ("fruit.jpg", "fruit.npy", "npy_source.npy"):
        (tmp_path / name).touch()

    def enumerate_files(channels, path):
        dummy = SimpleNamespace(channels=channels, fraction=1.0, prefix="test: ")
        return BaseDataset.get_img_files(dummy, str(path))

    rgb = enumerate_files(3, tmp_path)
    assert {Path(p).name for p in rgb} == {"fruit.jpg", "npy_source.npy"}
    assert len(enumerate_files(4, tmp_path)) == 3
    manifest = tmp_path / "images.txt"
    manifest.write_text(str(tmp_path / "fruit.npy") + "\n" + str(tmp_path / "fruit.jpg"), encoding="utf-8")
    assert len(enumerate_files(3, manifest)) == 2
    assert (tmp_path / "fruit.npy").exists()  # no dataset files deleted


def test_pretrained_head_remap_and_save_reload(tmp_path):
    checkpoint = ROOT / "yolo11n-seg.pt"
    if not checkpoint.is_file():
        pytest.skip("Local initialization checkpoint unavailable")
    source = YOLO(str(checkpoint), task="segment", verbose=False)
    for path in YAMLS:
        api = YOLO(str(path), task="segment", verbose=False).load(str(checkpoint))
        torch.testing.assert_close(
            api.model.model[-1].proto.cv1.conv.weight, source.model.model[-1].proto.cv1.conv.weight
        )
    output_path = tmp_path / "roundtrip.pt"
    api.save(str(output_path))
    reloaded = YOLO(str(output_path), task="segment", verbose=False)
    with torch.no_grad():
        reloaded.model.eval()(torch.zeros(1, 3, 128, 128))
