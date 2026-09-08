"""Frozen guide isolation, geometry, budget and train-only supervision."""

import json

import numpy as np
import pytest
import torch

from citrus_crop_guide import (GUIDE_SIZE, CropGuide, TinyCropGuide, guide_image, guide_target, heatmap_loss,
                               train_crop_guide, windows_from_heatmap)
from citrus_slicing import view_windows


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_small_heatmap_network_backward_and_geometry():
    image = np.zeros((100, 200, 3), np.uint8)
    rgb, geometry = guide_image(image)
    assert rgb.shape == (3, GUIDE_SIZE, GUIDE_SIZE)
    rows = [(0, np.array([[.2, .3], [.3, .3], [.3, .5], [.2, .5]]))]
    target = torch.from_numpy(guide_target(rows, geometry)[None])
    model = TinyCropGuide()
    assert sum(p.numel() for p in model.parameters()) < 20000
    prediction = model(torch.from_numpy(rgb[None]).float())
    assert prediction.shape == target.shape == (1, 1, GUIDE_SIZE // 8, GUIDE_SIZE // 8)
    loss = heatmap_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    empty = heatmap_loss(torch.zeros_like(target), torch.zeros_like(target))
    assert torch.isfinite(empty)


def test_content_changes_crop_positions_with_same_budget_and_size():
    first, second = np.zeros((96, 96), np.float32), np.zeros((96, 96), np.float32)
    first[10:20, 10:20] = 1
    second[70:80, 70:80] = 1
    left = windows_from_heatmap((100, 200), first)
    right = windows_from_heatmap((100, 200), second)
    assert left != right
    for windows in (left, right):
        assert len(windows) == len(set(windows)) == 5
        assert windows[0] == (0, 0, 200, 100)
        for x0, y0, x1, y1 in windows[1:]:
            assert (x1 - x0, y1 - y0) == (120, 60)
            assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 100
    assert windows_from_heatmap((100, 200), np.zeros((96, 96))) == view_windows(100, 200)
    assert windows_from_heatmap((100, 200), np.full((96, 96), np.nan)) == view_windows(100, 200)


def test_guide_train_never_reads_val_labels_and_can_prepare_views(tmp_path, monkeypatch):
    import cv2
    import yaml
    import citrus_slicing

    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir()
        for index in range(2):
            cv2.imwrite(str(source / split / "images" / f"{index}.png"), np.full((64, 96, 3), 128, np.uint8))
            (source / split / "labels" / f"{index}.txt").write_text("0 .2 .2 .4 .2 .4 .4 .2 .4\n")
    data = source / "data.yaml"
    data.write_text(yaml.safe_dump(dict(path=str(source), train="train/images", val="val/images", names={0: "fruit"})))
    original_read = citrus_slicing.read_polygons
    read_labels = []

    def train_read(path):
        assert "/val/" not in str(path).replace("\\", "/")
        read_labels.append(str(path))
        return original_read(path)

    monkeypatch.setattr(citrus_slicing, "read_polygons", train_read)
    output = tmp_path / "guide/guide.pt"
    train_crop_guide(data, output, epochs=1)
    assert len(read_labels) == 2
    assert train_crop_guide(data, output, epochs=1) == output
    with pytest.raises(FileExistsError, match="changed"):
        train_crop_guide(data, output, epochs=2)
    guide = CropGuide(output)
    assert not guide.model.training
    assert not any(p.requires_grad for p in guide.model.parameters())
    image = cv2.imread(str(source / "train/images/0.png"))
    assert np.isfinite(guide.heatmap(image)).all()
    assert len(guide.windows(image)) == 5
    prepared = citrus_slicing.prepare_views(data, tmp_path / "guided", imgsz=128, window_selector=guide)
    manifest = json.loads((prepared.parent / "views.json").read_text())
    assert manifest["signature"]["guide_sha256"] == guide.sha256
    assert manifest["training_sources"] == 2
    assert citrus_slicing.prepare_views(data, tmp_path / "guided", imgsz=128, window_selector=guide) == prepared
    with pytest.raises(FileExistsError, match="differs"):
        citrus_slicing.prepare_views(data, tmp_path / "guided", imgsz=128)
