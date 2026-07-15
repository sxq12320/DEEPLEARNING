"""Focused tests for the Torchvision Mask R-CNN baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_dataset import prepare_dataset  # noqa: E402
from torchvision_maskrcnn_common import (
    build_maskrcnn_model,
    validate_prepared_dataset,
)  # noqa: E402


def create_source(root: Path) -> None:
    """Create one valid polygon instance in every split."""
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (32, 24), (20, 100, 20)).save(image_dir / f"{split}.jpg")
        (label_dir / f"{split}.txt").write_text(
            "0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n",
            encoding="utf-8",
        )


def test_validate_prepared_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    create_source(source)
    prepare_dataset(source, prepared, ["orange_immature"], mode="copy")

    report = validate_prepared_dataset(
        prepared, ("train", "val", "test"), ["orange_immature"]
    )

    assert report["train"]["images"] == 1
    assert report["train"]["instances"] == 1
    assert report["test"]["categories"] == ["orange_immature"]


def test_build_maskrcnn_has_background_and_one_foreground_class() -> None:
    pytest.importorskip("torchvision")

    model = build_maskrcnn_model(
        num_foreground_classes=1,
        imgsz=640,
        initialization="none",
    )

    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
    assert model.transform.min_size == (640,)
    assert model.transform.max_size == 640
