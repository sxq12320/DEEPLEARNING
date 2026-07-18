"""Standardize the current citrus LabelMe dataset and rebuild YOLO segmentation data."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from labelme_to_yolo_dataset import (
    DEFAULT_CLASS_NAME,
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SourceRecord,
    convert_dataset,
    find_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "data" / "orange_wuxi"
STANDARDIZED_ROOT = PROJECT_ROOT / "data" / "orange_standardized"
YOLO_ROOT = PROJECT_ROOT / "data" / "orange_yolo"
STAGING_ROOT = PROJECT_ROOT / "data" / "_orange_standardized_build"


def ensure_managed_path(path: Path, expected: Path) -> None:
    """Refuse destructive operations outside the explicitly managed path."""
    if path.resolve() != expected.resolve():
        raise ValueError(f"Refusing to modify unexpected path: {path.resolve()}")


def remove_managed_directory(path: Path, expected: Path) -> None:
    """Remove a generated directory after an exact-path check."""
    ensure_managed_path(path, expected)
    if path.exists():
        shutil.rmtree(path)


def read_valid_records(
    source_root: Path,
    class_name: str,
) -> tuple[list[tuple[SourceRecord, dict[str, Any]]], dict[str, Any]]:
    """Validate source annotations and return records that have matching JSON files."""
    records = find_records(source_root)
    valid: list[tuple[SourceRecord, dict[str, Any]]] = []
    missing_json: list[str] = []
    invalid_json: list[str] = []
    labels: Counter[str] = Counter()
    unsupported_shapes: Counter[str] = Counter()
    empty_annotations: list[str] = []

    for record in tqdm(records, desc="Validate source", unit="image", dynamic_ncols=True):
        if not record.json_path.is_file():
            missing_json.append(str(record.image_path))
            continue
        try:
            data = json.loads(record.json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid_json.append(str(record.json_path))
            continue

        shapes = data.get("shapes") or []
        if not shapes:
            empty_annotations.append(str(record.json_path))
        for shape in shapes:
            label = str(shape.get("label", ""))
            labels[label] += 1
            shape_type = str(shape.get("shape_type") or "polygon")
            if shape_type not in {"polygon", "rectangle"}:
                unsupported_shapes[shape_type] += 1
        valid.append((record, data))

    unexpected_labels = {
        label: count for label, count in labels.items() if label != class_name
    }
    report = {
        "source_root": str(source_root.resolve()),
        "discovered_images": len(records),
        "valid_pairs": len(valid),
        "excluded_missing_json": missing_json,
        "invalid_json": invalid_json,
        "empty_annotations": empty_annotations,
        "labels": dict(labels),
        "unexpected_labels": unexpected_labels,
        "unsupported_shapes": dict(unsupported_shapes),
    }
    if invalid_json:
        raise ValueError(f"Invalid JSON files found: {len(invalid_json)}")
    if unexpected_labels:
        raise ValueError(f"Unexpected labels found: {unexpected_labels}")
    if unsupported_shapes:
        raise ValueError(f"Unsupported LabelMe shape types found: {dict(unsupported_shapes)}")
    if not valid:
        raise ValueError("No valid image/annotation pairs were found.")
    return valid, report


def normalized_image_suffix(source: Path) -> str:
    """Use a consistent extension without recompressing JPEG data."""
    suffix = source.suffix.lower()
    return ".jpg" if suffix in {".jpg", ".jpeg"} else suffix


def build_standardized_dataset(
    source_root: Path = SOURCE_ROOT,
    output_root: Path = STANDARDIZED_ROOT,
    class_name: str = DEFAULT_CLASS_NAME,
) -> dict[str, Any]:
    """Create img/labels pairs named IMG_0000 onward and preserve a source mapping."""
    valid, report = read_valid_records(source_root, class_name)
    remove_managed_directory(STAGING_ROOT, STAGING_ROOT)
    image_dir = STAGING_ROOT / "img"
    label_dir = STAGING_ROOT / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    mapping_rows: list[dict[str, str]] = []
    total_instances = 0
    for index, (record, data) in enumerate(
        tqdm(valid, desc="Standardize pairs", unit="pair", dynamic_ncols=True)
    ):
        stem = f"IMG_{index:04d}"
        image_name = f"{stem}{normalized_image_suffix(record.image_path)}"
        label_name = f"{stem}.json"
        output_image = image_dir / image_name
        output_label = label_dir / label_name

        shutil.copy2(record.image_path, output_image)
        with Image.open(record.image_path) as image:
            width, height = image.size

        shapes = data.get("shapes") or []
        instances = sum(1 for shape in shapes if str(shape.get("label", "")) == class_name)
        total_instances += instances
        data["imagePath"] = f"../img/{image_name}"
        data["imageWidth"] = width
        data["imageHeight"] = height
        output_label.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mapping_rows.append(
            {
                "new_stem": stem,
                "new_image": image_name,
                "new_label": label_name,
                "source_subset": record.subset,
                "source_image": str(record.image_path),
                "source_json": str(record.json_path),
                "instances": str(instances),
            }
        )

    with (STAGING_ROOT / "rename_mapping.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "new_stem",
                "new_image",
                "new_label",
                "source_subset",
                "source_image",
                "source_json",
                "instances",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    (STAGING_ROOT / "missing_annotations.txt").write_text(
        "\n".join(report["excluded_missing_json"])
        + ("\n" if report["excluded_missing_json"] else ""),
        encoding="utf-8",
    )
    report.update(
        {
            "output_root": str(output_root.resolve()),
            "renamed_pairs": len(mapping_rows),
            "instances": total_instances,
            "first_name": mapping_rows[0]["new_stem"],
            "last_name": mapping_rows[-1]["new_stem"],
        }
    )
    (STAGING_ROOT / "standardization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    remove_managed_directory(output_root, STANDARDIZED_ROOT)
    STAGING_ROOT.replace(output_root)
    return report


def rebuild_all() -> dict[str, Any]:
    """Build the standardized LabelMe directory and overwrite orange_yolo."""
    standard_report = build_standardized_dataset()
    yolo_report = convert_dataset(
        source_root=STANDARDIZED_ROOT,
        output_root=YOLO_ROOT,
        class_name=DEFAULT_CLASS_NAME,
        seed=DEFAULT_SEED,
        final_ratios=DEFAULT_RATIOS,
        augment_splits=[],
        augmentations=[],
    )
    return {
        "standardized": {
            "path": str(STANDARDIZED_ROOT),
            "pairs": standard_report["renamed_pairs"],
            "instances": standard_report["instances"],
            "excluded_missing_json": len(standard_report["excluded_missing_json"]),
            "first_name": standard_report["first_name"],
            "last_name": standard_report["last_name"],
        },
        "orange_yolo": {
            "path": str(YOLO_ROOT),
            "split_counts": yolo_report["final_split_counts"],
            "instances_by_split": yolo_report["instances_by_split"],
            "missing_json": len(yolo_report["missing_json"]),
            "data_yaml": str(YOLO_ROOT / "data.yaml"),
        },
    }


def main() -> None:
    """Run the complete standardization and conversion workflow."""
    print(json.dumps(rebuild_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
