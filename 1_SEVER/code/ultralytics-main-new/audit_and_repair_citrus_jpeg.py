"""Audit JPEG endings and optionally create a non-destructive repaired dataset copy.

Many orchard images contain vendor trace data after the JPEG EOI marker. Pillow
can decode them, but Ultralytics treats any file not ending in ``FF D9`` as
corrupt and may re-save it while building a cache. This tool distinguishes
trailing data from a truly missing EOI marker. The default mode is read-only.

Examples:
    python audit_and_repair_citrus_jpeg.py --source /data/orange_grouped
    python audit_and_repair_citrus_jpeg.py --source /data/orange_grouped \
        --destination /data/orange_grouped_eoi_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"E:\mastercode\data\orange_yolo_grouped_dedup_20260820")
DEFAULT_REPORT = ROOT / "1_results" / "_data_audit" / "jpeg_integrity_20260826"
JPEG_SUFFIXES = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    """Parse a read-only audit or non-destructive copy repair."""
    parser = argparse.ArgumentParser(description="Audit citrus JPEG EOI markers without overwriting source data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, help="New dataset directory; source is never modified.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def jpeg_eoi_end(data: bytes) -> int:
    """Return the byte offset immediately after the JPEG EOI marker, or -1.

    The parser follows JPEG marker lengths and handles entropy-coded byte
    stuffing/restart markers, avoiding false ``FF D9`` matches in APP trailers.
    """
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return -1
    position = 2
    in_scan = False
    while position + 1 < len(data):
        if in_scan:
            marker_start = data.find(b"\xff", position)
            if marker_start < 0 or marker_start + 1 >= len(data):
                return -1
            marker = data[marker_start + 1]
            while marker == 0xFF and marker_start + 2 < len(data):
                marker_start += 1
                marker = data[marker_start + 1]
            if marker == 0x00 or 0xD0 <= marker <= 0xD7:
                position = marker_start + 2
                continue
            if marker == 0xD9:
                return marker_start + 2
            position = marker_start
            in_scan = False
            continue

        marker_start = data.find(b"\xff", position)
        if marker_start < 0 or marker_start + 1 >= len(data):
            return -1
        marker_position = marker_start + 1
        while marker_position < len(data) and data[marker_position] == 0xFF:
            marker_position += 1
        if marker_position >= len(data):
            return -1
        marker = data[marker_position]
        position = marker_position + 1
        if marker == 0xD9:
            return position
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            return -1
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            return -1
        position += segment_length
        if marker == 0xDA:
            in_scan = True
    return -1


def inspect(path: Path, root: Path) -> dict[str, object]:
    """Inspect one JPEG without changing it."""
    data = path.read_bytes()
    end = jpeg_eoi_end(data)
    if end < 0:
        status = "missing_or_unparseable_eoi"
        trailing = -1
    else:
        trailing = len(data) - end
        status = "standard" if trailing == 0 else "trailing_data"
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "status": status,
        "eoi_end_offset": end,
        "trailing_bytes": trailing,
    }


def scan(root: Path) -> list[dict[str, object]]:
    """Inspect every JPEG under a dataset root."""
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES)
    return [inspect(path, root) for path in paths]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write audit rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def directory_size(root: Path) -> int:
    """Measure source bytes before a full-copy operation."""
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def prepare_copy(source: Path, destination: Path, rows: list[dict[str, object]]) -> None:
    """Copy the dataset and strip only bytes after a parsed EOI marker."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = int(directory_size(source) * 1.05)
    free = shutil.disk_usage(destination.parent).free
    if free < required:
        raise OSError(f"Need at least {required / 2**30:.2f} GiB free; only {free / 2**30:.2f} GiB is available.")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.cache"))
    for row in rows:
        if row["status"] == "trailing_data":
            path = destination / str(row["relative_path"])
            data = path.read_bytes()
            path.write_bytes(data[: int(row["eoi_end_offset"])])
        elif row["status"] != "standard":
            raise RuntimeError(
                f"Cannot safely repair {row['relative_path']}: no parsed EOI marker. "
                "Inspect it manually or use a documented Pillow re-encode step."
            )
    data_yaml = destination / "data.yaml"
    if data_yaml.is_file():
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        config["path"] = str(destination.resolve())
        data_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_manifest(root: Path, path: Path) -> None:
    """Write SHA-256 for images, labels, and YAML files in a repaired copy."""
    rows = []
    for item in sorted(file for file in root.rglob("*") if file.is_file() and not file.name.endswith(".cache")):
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        rows.append(
            {"relative_path": item.relative_to(root).as_posix(), "bytes": item.stat().st_size, "sha256": digest}
        )
    write_csv(path, rows)


def main() -> None:
    """Run the audit and optionally create a new dataset version."""
    args = parse_args()
    source = args.source.expanduser().resolve()
    report = args.report.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    rows = scan(source)
    if not rows:
        raise RuntimeError(f"No JPEGs found below {source}")
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / "jpeg_integrity.csv", rows)
    write_manifest(source, report / "source_dataset_sha256.csv")
    statuses = sorted({row["status"] for row in rows})
    counts = {status: sum(row["status"] == status for row in rows) for status in statuses}
    summary = {"source": str(source), "images": len(rows), "status_counts": counts, "source_modified": False}
    if args.destination:
        destination = args.destination.expanduser().resolve()
        prepare_copy(source, destination, rows)
        repaired_rows = scan(destination)
        write_csv(report / "jpeg_integrity_repaired.csv", repaired_rows)
        write_manifest(destination, report / "repaired_dataset_sha256.csv")
        summary["destination"] = str(destination)
        summary["repaired_status_counts"] = {
            status: sum(row["status"] == status for row in repaired_rows)
            for status in sorted({row["status"] for row in repaired_rows})
        }
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
