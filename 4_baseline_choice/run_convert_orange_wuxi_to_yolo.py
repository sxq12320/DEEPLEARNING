"""One-click strict conversion from orange_wuxi LabelMe data to YOLO segmentation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from labelme_to_yolo_dataset import (  # noqa: E402
    DEFAULT_CLASS_NAME,
    DEFAULT_OUTPUT,
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    DEFAULT_SOURCE,
    convert_dataset,
    find_records,
)


def validate_source() -> dict[str, object]:
    records = find_records(DEFAULT_SOURCE)
    missing_json: list[str] = []
    invalid_json: list[str] = []
    labels: Counter[str] = Counter()
    empty_annotations: list[str] = []

    for record in records:
        if not record.json_path.is_file():
            missing_json.append(str(record.json_path))
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
            labels[str(shape.get("label", ""))] += 1

    unexpected_labels = {
        label: count
        for label, count in labels.items()
        if label != DEFAULT_CLASS_NAME
    }
    return {
        "images": len(records),
        "json_files": len(records) - len(missing_json),
        "missing_json": missing_json,
        "invalid_json": invalid_json,
        "empty_annotations": empty_annotations,
        "labels": dict(labels),
        "unexpected_labels": unexpected_labels,
    }


def main() -> int:
    print("LabelMe -> YOLO segmentation")
    print(f"Source: {DEFAULT_SOURCE}")
    print(f"Output: {DEFAULT_OUTPUT}")
    print("Output is overwritten only after validation passes.")
    print()

    validation = validate_source()
    print(json.dumps(validation, ensure_ascii=False, indent=2))

    if validation["missing_json"] or validation["invalid_json"] or validation["unexpected_labels"]:
        print()
        print("Validation failed. Fix the listed annotations before conversion.")
        print("For a true negative image, save an explicit LabelMe JSON with an empty shapes list.")
        return 1

    print()
    print("Validation passed. Rebuilding orange_yolo...")
    report = convert_dataset(
        source_root=DEFAULT_SOURCE,
        output_root=DEFAULT_OUTPUT,
        class_name=DEFAULT_CLASS_NAME,
        seed=DEFAULT_SEED,
        final_ratios=DEFAULT_RATIOS,
        augment_splits=[],
        augmentations=[],
    )
    summary = {
        "output": str(DEFAULT_OUTPUT),
        "source_images": report["source_images"],
        "split_counts": report["final_split_counts"],
        "instances_by_split": report["instances_by_split"],
        "data_yaml": str(DEFAULT_OUTPUT / "data.yaml"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"\nConversion failed: {exc}")
        exit_code = 1
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(exit_code)
