"""Synthetic paired-view geometry tests; no user data or labels are modified."""

import pickle
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from citrus_j6_data import J6Dataset, aligned_high_view, build_j6_dataset, prepare_j6_batch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset


def _cfg(**overrides):
    values = dict(task="segment", imgsz=640, mask_ratio=2, cache=False, rect=False, workers=0, mosaic=0.0,
                  mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0,
                  shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    values.update(overrides)
    return get_cfg(overrides=values)


@pytest.fixture
def synthetic_data(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i, (h, w) in enumerate(((317, 563), (421, 251))):
        yy, xx = np.indices((h, w))
        raw = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(images / f"sample{i}.png"), raw)
        (labels / f"sample{i}.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.8 0.2 0.8\n", encoding="utf-8")
    return str(images), {"names": {0: "orange_immature"}, "nc": 1, "channels": 3}


@pytest.mark.parametrize("shape", [(317, 563), (421, 251), (641, 1283), (640, 640), (3, 5)])
def test_aligned_raw_content_and_affine(shape):
    h, w = shape
    raw = np.tile(np.arange(w, dtype=np.uint8), (h, 1))[..., None].repeat(3, axis=2)
    r = 640 / max(h, w)
    h1, w1 = min(int(np.ceil(h * r)), 640), min(int(np.ceil(w * r)), 640)
    top, left = round((640 - h1) / 2 - 0.1), round((640 - w1) / 2 - 0.1)
    high, geo = aligned_high_view(raw, (640, 640), ((h1 / h, w1 / w), (left, top)))
    assert high.shape == (1280, 1280, 3)
    expected = cv2.resize(raw, (w1 * 2, h1 * 2), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(high[2 * top : 2 * (top + h1), 2 * left : 2 * (left + w1)], expected)
    np.testing.assert_allclose(geo["raw_to_high"], np.diag([2, 2, 1]) @ geo["raw_to_global"])
    corners = geo["raw_to_global"] @ np.asarray([[0, w], [0, h], [1, 1]])
    np.testing.assert_allclose(corners[:2], [[left, left + w1], [top, top + h1]], atol=1e-4)


def test_val_preserves_every_standard_gt_and_metadata(synthetic_data):
    images, data = synthetic_data
    cfg = _cfg()
    standard = build_yolo_dataset(deepcopy(cfg), images, 1, data, mode="val", rect=True)
    paired = build_j6_dataset(cfg, images, 1, data, mode="val")
    for index in range(2):
        expected, actual = standard[index], paired[index]
        assert expected.keys() <= actual.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value)
            else:
                assert actual[key] == value
        assert actual["high_img"].shape[-2:] == tuple(x * 2 for x in actual["img"].shape[-2:])


def test_high_image_contains_information_lost_by_low_resize():
    yy, xx = np.indices((1021, 1777))
    raw = ((xx + yy) % 2 * 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    h1 = int(np.ceil(1021 * 640 / 1777))
    top = round((640 - h1) / 2 - 0.1)
    high, _ = aligned_high_view(raw, (640, 640), ((h1 / 1021, 640 / 1777), (0, top)))
    low_content = cv2.resize(raw, (640, h1), interpolation=cv2.INTER_LINEAR)
    upsampled = cv2.resize(low_content, (1280, h1 * 2), interpolation=cv2.INTER_LINEAR)
    actual = high[2 * top : 2 * (top + h1)]
    assert np.mean(np.abs(actual.astype(float) - upsampled)) > 1


def test_training_one_augmentation_mask_grid_and_collation(synthetic_data):
    images, data = synthetic_data
    cfg = _cfg()
    paired = build_j6_dataset(cfg, images, 2, data)
    samples = [paired[0], paired[1]]
    assert cfg.mask_ratio == 2  # caller config was not mutated
    assert paired.j6_protocol["internal_mask_ratio"] == 4
    assert paired.transforms.transforms[-1].mask_ratio == 4
    for sample in samples:
        assert sample["img"].shape == (3, 640, 640)
        assert sample["high_img"].shape == (3, 1280, 1280)
        assert sample["masks"].shape[-2:] == (320, 320)
        assert sample["bboxes"].shape == (1, 4)
        assert (sample["bboxes"] >= 0).all() and (sample["bboxes"] <= 1).all()
        low = cv2.resize(sample["high_img"].numpy().transpose(1, 2, 0), (640, 640), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(sample["img"].numpy(), low.transpose(2, 0, 1))
    batch = J6Dataset.collate_fn(samples)
    assert batch["high_img"].shape == (2, 3, 1280, 1280)
    assert batch["batch_idx"].tolist() == [0, 1]
    paired.close_mosaic(cfg)
    assert paired.transforms.transforms[-1].mask_ratio == 4


def test_empty_labels_keep_empty_instance_shapes(synthetic_data):
    images, data = synthetic_data
    from pathlib import Path

    for path in (Path(images).parent / "labels").glob("*.txt"):
        path.write_text("", encoding="utf-8")
    paired = build_j6_dataset(_cfg(), images, 2, data)
    batch = J6Dataset.collate_fn([paired[0], paired[1]])
    assert batch["bboxes"].shape == (0, 4)
    assert batch["batch_idx"].numel() == 0
    assert batch["masks"].shape[-2:] == (320, 320)


def test_cache_is_bounded_and_dataset_is_worker_pickleable(synthetic_data):
    images, data = synthetic_data
    paired = build_j6_dataset(_cfg(cache=True), images, 2, data, mode="val", high_cache_mb=5)
    for i in (0, 1, 0):
        paired[i]
        assert paired._j6_high_cache_bytes <= paired.j6_high_cache_limit
    copy = pickle.loads(pickle.dumps(paired))
    torch.testing.assert_close(copy[0]["high_img"], paired[0]["high_img"])


def test_real_worker_collation(synthetic_data):
    images, data = synthetic_data
    paired = build_j6_dataset(_cfg(cache=True), images, 2, data, mode="val", high_cache_mb=5)
    loader = torch.utils.data.DataLoader(paired, batch_size=2, num_workers=1, collate_fn=paired.collate_fn,
                                        multiprocessing_context="spawn", timeout=60)
    batch = next(iter(loader))
    assert batch["img"].shape[:2] == (2, 3)
    assert batch["high_img"].shape[-2:] == tuple(x * 2 for x in batch["img"].shape[-2:])


def test_normalization_is_idempotent_and_does_not_change_global_image():
    image = torch.full((1, 3, 4, 4), 0.25)
    batch = {"img": image, "high_img": torch.full((1, 3, 8, 8), 255, dtype=torch.uint8)}
    prepare_j6_batch(batch, "cpu")
    assert batch["img"] is image
    assert torch.equal(batch["high_img"], torch.ones_like(batch["high_img"]))
    first = batch["high_img"].clone()
    prepare_j6_batch(batch, "cpu")
    torch.testing.assert_close(batch["high_img"], first)


@pytest.mark.parametrize("bad", [{"rect": True}, {"multi_scale": 0.5}, {"imgsz": 1280}, {"mask_ratio": 4}])
def test_refuse_unimplemented_training_geometry(synthetic_data, bad):
    images, data = synthetic_data
    with pytest.raises(ValueError):
        build_j6_dataset(_cfg(**bad), images, 2, data)


def test_invalid_geometry_and_empty_collation():
    raw = np.zeros((7, 11, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        aligned_high_view(raw, (640, 640), ((1, 1), (2, 0)))
    with pytest.raises(ValueError):
        J6Dataset.collate_fn([])
