"""Tests for non-destructive citrus JPEG EOI auditing."""

from pathlib import Path

import yaml

from audit_and_repair_citrus_jpeg import jpeg_eoi_end, prepare_copy, scan


def mock_jpeg(trailer: bytes = b"") -> bytes:
    """Create the marker structure needed to exercise entropy parsing."""
    entropy = b"\x01\x02\xff\x00\x03\xff\xd0\x04"
    return b"\xff\xd8\xff\xda\x00\x02" + entropy + b"\xff\xd9" + trailer


def test_jpeg_parser_distinguishes_trailer_from_missing_eoi():
    """Vendor bytes after EOI must not be mislabeled as a missing EOI marker."""
    standard = mock_jpeg()
    with_trailer = mock_jpeg(b"ctrace\x00DfxData\x00")

    assert jpeg_eoi_end(standard) == len(standard)
    assert jpeg_eoi_end(with_trailer) == len(standard)
    assert jpeg_eoi_end(with_trailer[:-17]) == -1


def test_prepare_copy_strips_trailer_without_overwriting_source(tmp_path: Path):
    """Copy mode preserves source bytes, strips only trailer bytes, and excludes stale caches."""
    source = tmp_path / "source"
    image_dir = source / "train" / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "sample.jpg"
    original = mock_jpeg(b"vendor-trailer")
    image.write_bytes(original)
    (source / "train" / "labels.cache").write_bytes(b"stale")
    (source / "data.yaml").write_text(
        yaml.safe_dump({"path": str(source), "train": "train/images", "val": "train/images", "names": {0: "fruit"}}),
        encoding="utf-8",
    )
    rows = scan(source)
    destination = tmp_path / "repaired"

    prepare_copy(source, destination, rows)

    assert image.read_bytes() == original
    assert (destination / "train" / "images" / "sample.jpg").read_bytes() == mock_jpeg()
    assert not (destination / "train" / "labels.cache").exists()
    assert yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))["path"] == str(
        destination.resolve()
    )
