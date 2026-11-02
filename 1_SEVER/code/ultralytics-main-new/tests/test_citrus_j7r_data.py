"""Packed-input/data contracts; synthetic files only, no original labels are changed."""

from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from citrus_j7r_data import (J7R_INPUT_PROTOCOL, build_j7r_dataset, j7r_model_spec, pack_j7r_batch,
                             pack_j7r_images)
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset


@pytest.fixture
def synthetic_rgb(tmp_path):
    image_dir, label_dir = tmp_path / "images", tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    for i, (height, width) in enumerate(((317, 563), (421, 251))):
        yy, xx = np.indices((height, width))
        image = np.stack((xx % 256, yy % 256, (xx + 2 * yy) % 256), -1).astype(np.uint8)
        assert cv2.imwrite(str(image_dir / "sample{}.png".format(i)), image)
        (label_dir / "sample{}.txt".format(i)).write_text("0 .2 .2 .7 .2 .7 .8 .2 .8\n", encoding="utf-8")
    return str(image_dir), {"names": {0: "orange_immature"}, "nc": 1, "channels": 3}


def cfg(**extra):
    values = dict(task="segment", imgsz=640, mask_ratio=2, cache=False, rect=False, workers=0,
                  mosaic=0., mixup=0., cutmix=0., copy_paste=0., degrees=0., translate=0., scale=0.,
                  shear=0., perspective=0., fliplr=0., flipud=0., hsv_h=0., hsv_s=0., hsv_v=0.)
    values.update(extra)
    return get_cfg(overrides=values)


def test_pack_exact_recovery_and_normalization_once():
    global_rgb = torch.rand(2, 3, 32, 64)
    high = torch.randint(256, (2, 3, 64, 128), dtype=torch.uint8)
    actual = pack_j7r_images(global_rgb, high, "raw")
    assert actual.shape == (2, 15, 32, 64)
    torch.testing.assert_close(actual[:, :3], global_rgb, rtol=0, atol=0)
    torch.testing.assert_close(F.pixel_shuffle(actual[:, 3:], 2), high.float() / 255, rtol=0, atol=0)
    normalized = pack_j7r_images(global_rgb, high.float() / 255, "raw")
    torch.testing.assert_close(actual, normalized, rtol=0, atol=0)


def test_explicit_degraded_and_global_controls_ignore_raw():
    global_rgb = torch.rand(1, 3, 32, 64)
    pseudo = pack_j7r_images(global_rgb, None, "upsampled")
    expected = F.interpolate(global_rgb, scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(F.pixel_shuffle(pseudo[:, 3:], 2), expected, rtol=0, atol=0)
    control = pack_j7r_images(global_rgb, torch.ones(1), "global")
    assert torch.count_nonzero(control[:, 3:]) == 0


@pytest.mark.parametrize("global_rgb,high,observation", [
    (torch.rand(1, 15, 32, 32), torch.rand(1, 3, 64, 64), "raw"),
    (torch.zeros(1, 3, 32, 32, dtype=torch.uint8), torch.rand(1, 3, 64, 64), "raw"),
    (torch.rand(1, 3, 32, 32), None, "raw"),
    (torch.rand(1, 3, 32, 32), torch.rand(2, 3, 64, 64), "raw"),
    (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 63, 64), "raw"),
    (torch.rand(1, 3, 32, 32), torch.zeros(1, 3, 64, 64, dtype=torch.long), "raw"),
    (torch.rand(1, 3, 32, 32), None, "silent_fallback"),
])
def test_invalid_input_contracts(global_rgb, high, observation):
    with pytest.raises(ValueError):
        pack_j7r_images(global_rgb, high, observation)


def test_batch_only_replaces_image_and_releases_high():
    masks = torch.ones(1, 16, 16)
    batch = {"img": torch.rand(1, 3, 32, 32), "high_img": torch.rand(1, 3, 64, 64), "masks": masks}
    packed = pack_j7r_batch(batch)
    assert packed is batch and packed["masks"] is masks and "high_img" not in packed
    with pytest.raises(ValueError):
        pack_j7r_batch(batch)


def test_sensor_stays_rgb_and_val_gt_is_bit_exact(synthetic_rgb):
    images, data = synthetic_rgb
    config = cfg()
    standard = build_yolo_dataset(deepcopy(config), images, 1, data, mode="val", rect=True)
    paired = build_j7r_dataset(config, images, 1, data, mode="val")
    for i in range(2):
        expected, actual = standard[i], paired[i]
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value)
            else:
                assert actual[key] == value
        assert actual["high_img"].shape[-2:] == tuple(2 * x for x in actual["img"].shape[-2:])
    assert data["channels"] == 3
    with pytest.raises(ValueError):
        build_j7r_dataset(config, images, 1, {**data, "channels": 15}, mode="val")


def test_training_one_augmentation_and_canonical_masks(synthetic_rgb):
    images, data = synthetic_rgb
    dataset = build_j7r_dataset(cfg(), images, 2, data)
    sample = dataset[0]
    assert sample["img"].shape == (3, 640, 640)
    assert sample["high_img"].shape == (3, 1280, 1280)
    assert sample["masks"].shape[-2:] == (320, 320)
    low = cv2.resize(sample["high_img"].numpy().transpose(1, 2, 0), (640, 640), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(sample["img"].numpy(), low.transpose(2, 0, 1))
    dataset.close_mosaic(cfg())
    assert dataset.transforms.transforms[-1].mask_ratio == 4


def test_model_spec_wrappers_and_explicit_semantics():
    from types import SimpleNamespace

    yaml = {"channels": 15, "citrus_input_protocol": J7R_INPUT_PROTOCOL, "citrus_observation": "raw"}
    wrapped = SimpleNamespace(backend=SimpleNamespace(model=SimpleNamespace(yaml=yaml)))
    assert j7r_model_spec(wrapped)["observation"] == "raw"
    with pytest.raises(TypeError):
        j7r_model_spec({"channels": 15})
    with pytest.raises(ValueError):
        j7r_model_spec({**yaml, "channels": 3})
    with pytest.raises(ValueError):
        j7r_model_spec({**yaml, "citrus_observation": "implicit"})


def test_native640_uses_native_dataset_without_high_and_same_gt(synthetic_rgb):
    images, data = synthetic_rgb
    config = cfg()
    native = build_yolo_dataset(deepcopy(config), images, 2, data, mode="train", rect=False)
    control = build_j7r_dataset(config, images, 2, data, data_policy="native640")
    assert type(control) is type(native)
    expected, actual = native[0], control[0]
    assert "high_img" not in actual and actual["masks"].shape[-2:] == (320, 320)
    for key in ("img", "masks", "bboxes", "cls"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    yaml = {"channels": 15, "citrus_input_protocol": J7R_INPUT_PROTOCOL, "citrus_observation": "global",
            "citrus_data_policy": "native640"}
    assert j7r_model_spec(yaml)["data_policy"] == "native640"
    with pytest.raises(ValueError):
        j7r_model_spec({**yaml, "citrus_observation": "raw"})
