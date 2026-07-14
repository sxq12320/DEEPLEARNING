"""Tests for the dataset conversion pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_dataset import parse_label_file, prepare_dataset  # noqa: E402


def create_tiny_source(root: Path) -> None:
    """Create a three-split dataset with one positive and one negative per split."""
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (20, 10), (20, 80, 20)).save(image_dir / f"{split}_positive.jpg")
        Image.new("RGB", (20, 10), (10, 10, 10)).save(image_dir / f"{split}_negative.jpg")
        (label_dir / f"{split}_positive.txt").write_text(
            "0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n",
            encoding="utf-8",
        )
        (label_dir / f"{split}_negative.txt").write_text("", encoding="utf-8")


def test_prepare_dataset_builds_all_layouts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    create_tiny_source(source)

    report = prepare_dataset(source, output, ["orange_immature"], mode="copy")

    assert report["totals"] == {"images": 6, "instances": 3, "negative_images": 3}
    assert (output / "yolo" / "dataset.yaml").is_file()
    assert (output / "semantic" / "masks" / "train" / "train_negative.png").is_file()
    assert (output / "rfdetr" / "valid" / "_annotations.coco.json").is_file()

    coco = json.loads((output / "coco" / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["annotations"][0]["area"] > 0


def test_invalid_normalized_coordinate_is_rejected(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 0 0 1.2 0 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_label_file(label)

