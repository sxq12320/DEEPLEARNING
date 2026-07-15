"""Focused geometry and watershed tests for the U-Net baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from unet_common import (  # noqa: E402
    crop_letterbox,
    letterbox_array,
    restore_binary_mask,
    watershed_instances,
)


def test_letterbox_crop_and_restore_preserve_geometry() -> None:
    mask = np.zeros((30, 50), dtype=np.uint8)
    mask[5:20, 10:40] = 255
    boxed, info = letterbox_array(mask, 64, is_mask=True)

    assert boxed.shape == (64, 64)
    assert crop_letterbox(boxed, info).shape == (
        info.resized_height,
        info.resized_width,
    )
    restored = restore_binary_mask(boxed, info)
    assert restored.shape == mask.shape
    assert restored[10, 20]
    assert not restored[0, 0]


def test_watershed_separates_touching_objects() -> None:
    size = 96
    rows, columns = np.ogrid[:size, :size]
    first = (rows - 48) ** 2 + (columns - 36) ** 2 <= 18**2
    second = (rows - 48) ** 2 + (columns - 60) ** 2 <= 18**2
    probability = np.where(first | second, 0.95, 0.05).astype(np.float32)
    _, info = letterbox_array(
        np.zeros((size, size), dtype=np.uint8), size, is_mask=True
    )

    instances = watershed_instances(
        probability,
        info,
        probability_threshold=0.5,
        min_distance=10,
        min_area=50,
        max_instances=10,
    )

    assert len(instances) == 2
    assert all(mask.shape == (size, size) for mask, _ in instances)
    assert all(score > 0.9 for _, score in instances)
