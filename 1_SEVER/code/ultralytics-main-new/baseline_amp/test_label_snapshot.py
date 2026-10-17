"""Prepared framework views must not share mutable source TXT storage."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import prepare  # noqa: E402


def source_data(tmp_path):
    root = tmp_path / "source"
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.new("RGB", (32, 32), (80, 120, 70)).save(root / split / "images/one.png")
        (root / split / "labels/one.txt").write_bytes(b"0 .25 .25 .75 .25 .75 .75 .25 .75\n")
    data = root / "data.yaml"
    data.write_text("path: .\ntrain: train/images\nval: val/images\nnames: [orange_immature]\n", encoding="utf-8")
    return data


def test_source_edit_does_not_silently_change_only_yolo(tmp_path):
    data = source_data(tmp_path)
    output = tmp_path / "prepared"
    summary = prepare(data, output)
    label = data.parent / "train/labels/one.txt"
    derived = output / "yolo/train/labels/0000001_one.txt"
    before = derived.read_bytes()
    coco = output / "coco/annotations/instances_train.json"
    coco_before = coco.read_bytes()
    assert not label.samefile(derived)
    assert summary["label_storage"] == "independent_txt_snapshot_v1_20261008"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest[0]["label_sha256"] == hashlib.sha256(before).hexdigest()
    label.write_bytes(before.replace(b".25", b".20"))
    assert derived.read_bytes() == before and coco.read_bytes() == coco_before
    with pytest.raises(ValueError, match="Dataset changed"):
        prepare(data, output)


def test_same_length_derived_label_edit_is_rejected_without_repair(tmp_path):
    data = source_data(tmp_path)
    output = tmp_path / "prepared"
    prepare(data, output)
    source = data.parent / "train/labels/one.txt"
    before = source.read_bytes()
    derived = output / "yolo/train/labels/0000001_one.txt"
    edited = before.replace(b".25", b".20")
    derived.write_bytes(edited)
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="Conflicting derived label snapshot"):
        prepare(data, output)
    assert derived.read_bytes() == edited and source.read_bytes() == before


def test_legacy_hardlinked_labels_not_replaced(tmp_path):
    data = source_data(tmp_path)
    output = tmp_path / "prepared"
    prepare(data, output)
    source = data.parent / "train/labels/one.txt"
    derived = output / "yolo/train/labels/0000001_one.txt"
    derived.unlink()  # Only this test's disposable prepared copy.
    try:
        os.link(source, derived)
    except OSError:
        pytest.skip("The test filesystem does not support hard links")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="Legacy prepared label is hard-linked"):
        prepare(data, output)
    assert derived.samefile(source) and derived.read_bytes() == before


def test_edit_during_preparation_cannot_mark_success(tmp_path, monkeypatch):
    import prepare as preparation

    data = source_data(tmp_path)
    output = tmp_path / "prepared"
    original = preparation.materialize_label
    changed = False

    def mutate_after_copy(source, destination, content):
        nonlocal changed
        original(source, destination, content)
        if not changed:
            source.write_bytes(content.replace(b".25", b".20"))
            changed = True

    monkeypatch.setattr(preparation, "materialize_label", mutate_after_copy)
    with pytest.raises(ValueError, match="changed DURING preparation"):
        prepare(data, output)
    assert not (output / "summary.json").exists()
