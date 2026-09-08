"""E input geometry, public model API and source-balanced augmentation regression."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from citrus_e_suite import NAMES, ROOT, TILE_PROBABILITY, YAML_DIR, select_names
from citrus_slicing import SourceBalancedViewDataset, clip_instances, prepare_views, view_windows
from tests.test_citrus_sage_v4r import example_batch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", NAMES)
def test_public_build_loss_backward_fuse(name, tmp_path):
    path = YAML_DIR / f"{name}.yaml"
    assert YOLO(str(path), verbose=False).task == "segment"
    model = SegmentationModel(path, nc=1, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    loss, items = model.loss(example_batch())
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model[-1].stride.tolist() == [8, 16, 32]
    assert 1 < get_flops(model, 640) < 15
    with torch.inference_mode():
        x = torch.rand(1, 3, 128, 160)
        model.eval()
        expected = model(x)[0]
        fused = deepcopy(model).fuse(verbose=False)(x)[0]
        torch.testing.assert_close(expected[0], fused[0], rtol=0.002, atol=0.02)
        torch.testing.assert_close(expected[1], fused[1], rtol=0.002, atol=0.02)
    api = YOLO(str(path), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
    file = tmp_path / "model.pt"
    api.save(str(file))
    assert YOLO(str(file), verbose=False).task == "segment"


def test_declared_input_controls_are_identical_networks():
    configs = [yaml.safe_load((YAML_DIR / f"{n}.yaml").read_text()) for n in NAMES]
    assert configs[0] == configs[1]
    assert configs[2] == configs[3]
    assert configs[4] == configs[5]
    assert configs[3] == configs[6]
    assert configs[8] == configs[3]
    assert all(c["head"] == configs[0]["head"] for c in configs[:7])
    assert configs[0]["backbone"][1:] == configs[2]["backbone"][1:]
    assert configs[7]["backbone"] == configs[3]["backbone"]
    assert configs[7]["head"][-1][0] == [16, 13, 10, 2]
    assert all(layer[2] == "nn.Identity" for layer in configs[7]["head"][6:12])


def test_guided_experiment_dispatch_and_topdown_budget():
    assert len(NAMES) == len(set(NAMES)) == 9
    assert select_names("all") == NAMES
    assert TILE_PROBABILITY[NAMES[5]] == 0
    assert TILE_PROBABILITY[NAMES[6]] == .25
    assert TILE_PROBABILITY[NAMES[7]] == .5
    assert TILE_PROBABILITY[NAMES[8]] == .5
    assert select_names("guided") == [NAMES[3], NAMES[8]]
    control = SegmentationModel(YAML_DIR / f"{NAMES[3]}.yaml", nc=1, verbose=False)
    topdown = SegmentationModel(YAML_DIR / f"{NAMES[7]}.yaml", nc=1, verbose=False)
    assert sum(p.numel() for p in topdown.parameters()) < sum(p.numel() for p in control.parameters())
    assert get_flops(topdown, 640) < get_flops(control, 640)


def test_geometry_keeps_slivers_and_rejects_disconnected_instances():
    square = np.array([[.49, .2], [.7, .2], [.7, .4], [.49, .4]])
    clipped, reason = clip_instances([(0, square)], (100, 100), (0, 0, 50, 50))
    assert reason is None and len(clipped) == 1
    assert clipped[0][1][:, 0].min() == pytest.approx(.98)  # one pixel sliver kept
    u = np.array([[.1, .1], [.3, .1], [.3, .7], [.7, .7], [.7, .1], [.9, .1], [.9, .9], [.1, .9]])
    clipped, reason = clip_instances([(0, u)], (100, 100), (0, 0, 100, 50))
    assert not clipped and "disconnected" in reason
    full, reason = clip_instances([(0, u)], (100, 100), (0, 0, 100, 100))
    assert reason is None
    np.testing.assert_array_equal(full[0][1], u)
    assert clip_instances([(0, square)], (100, 100), (0, 60, 40, 100)) == ([], None)


def test_four_windows_cover_and_overlap():
    windows = view_windows(100, 120)
    assert len(windows) == 5 and windows[0] == (0, 0, 120, 100)
    coverage = np.zeros((100, 120), dtype=int)
    for x0, y0, x1, y1 in windows[1:]:
        coverage[y0:y1, x0:x1] += 1
    assert coverage.min() == 1 and coverage.max() == 4


def test_global_fusion_deduplicates_without_unioning_touching_fruits():
    from eval_citrus_sliced import merge_candidates

    patch = np.ones((10, 10), dtype=bool)
    first = dict(box=[0, 0, 10, 10], score=.9, cls=0, patch=patch, offset=(0, 0))
    duplicate = {**first, "score": .8}
    touching = dict(box=[10, 0, 20, 10], score=.7, cls=0, patch=patch, offset=(10, 0))
    predictions = merge_candidates([first, duplicate, touching], (20, 20))
    assert len(predictions["conf"]) == 2
    assert predictions["masks"].sum((1, 2)).tolist() == [100, 100]
    assert not (predictions["masks"][0] & predictions["masks"][1]).any()
    assert merge_candidates([], (20, 20))["masks"].shape == (0, 20, 20)


def test_prepared_source_balancing_cache_mosaic_and_labels(tmp_path):
    data = ROOT / "reports/sage_v5_20260904/fixture/data.yaml"
    if not data.exists():
        pytest.skip("Optional local fixture unavailable")
    # This old smoke fixture deliberately reuses the same train/val images;
    # use separate copies for the slicing split guard.
    import shutil
    original = check_det_dataset(str(data), autodownload=False)
    source = tmp_path / "source"
    for split in ("train", "val"):
        shutil.copytree(original[split], source / split / "images")
        shutil.copytree(str(original[split]).replace("images", "labels"), source / split / "labels")
    source_yaml = source / "data.yaml"
    source_yaml.write_text(yaml.safe_dump(dict(path=str(source), train="train/images", val="val/images", names={0: "fruit"})))
    unchanged = {p: p.read_bytes() for p in source.rglob("*.txt")}
    prepared = prepare_views(source_yaml, tmp_path / "views", imgsz=256)
    assert prepared == prepare_views(source_yaml, tmp_path / "views", imgsz=256)
    assert all(p.read_bytes() == b for p, b in unchanged.items())
    manifest = json.loads((prepared.parent / "views.json").read_text())
    assert manifest["training_sources"] == 4
    for group in manifest["groups"]:
        label = str(group["source"]).replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        from pathlib import Path
        assert (prepared.parent / group["views"][0]["label"]).read_text() == Path(label).read_text()
    config = check_det_dataset(str(prepared), autodownload=False)
    hyp = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    dataset = SourceBalancedViewDataset(img_path=config["train"], imgsz=256, batch_size=2, augment=True,
                                        hyp=hyp, rect=False, cache=True, stride=32, task="segment", data=config)
    assert len(dataset) == 4 and len(dataset.labels) >= 4
    dataset.tile_probability = 0
    assert all("view0" in dataset.get_image_and_label(i)["im_file"] for i in range(4))
    dataset.tile_probability = 1
    for i, group in enumerate(dataset.source_groups):
        if len(group) > 1:
            assert "view0" not in dataset.get_image_and_label(i)["im_file"]
    for i in range(12):  # Mosaic draws logical source indices, not cached-view indices.
        sample = dataset[i % 4]
        assert sample["img"].shape == (3, 256, 256)
        assert torch.isfinite(sample["bboxes"]).all()
    dataset.close_mosaic(hyp)
    assert dataset[0]["img"].shape == (3, 256, 256)
    with pytest.raises(FileExistsError, match="differs"):
        prepare_views(source_yaml, tmp_path / "views", imgsz=640)
