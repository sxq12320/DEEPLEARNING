"""Validate the leakage-controlled citrus 7:2:1 and four-fold datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_ROOT = PROJECT_ROOT / "data" / "orange_yolo"
DEFAULT_CV_ROOT = PROJECT_ROOT / "data" / "orange_yolo_4fold"


def fail(message: str) -> None:
    """Raise one consistently formatted validation error."""
    raise AssertionError(message)


def read_image_list(path: Path, dataset_root: Path) -> List[Path]:
    """Resolve an Ultralytics image-list file relative to the dataset root."""
    if not path.is_file():
        fail(f"Missing split list: {path}")
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    resolved = [(dataset_root / line).resolve() for line in raw_lines]
    if len(resolved) != len(set(resolved)):
        fail(f"Duplicate image paths in {path.name}")
    for image_path in resolved:
        if not image_path.is_file():
            fail(f"Missing listed image: {image_path}")
    return resolved


def validate_yolo_labels(image_root: Path, label_root: Path) -> Tuple[int, int]:
    """Check image-label pairing and segmentation-coordinate validity."""
    image_paths = sorted(path for path in image_root.iterdir() if path.is_file())
    label_paths = sorted(label_root.glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}
    if image_stems != label_stems:
        fail(
            "Image-label stems differ: "
            f"missing_labels={sorted(image_stems - label_stems)[:10]}, "
            f"orphan_labels={sorted(label_stems - image_stems)[:10]}"
        )

    instance_count = 0
    for label_path in label_paths:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7 or (len(parts) - 1) % 2:
                fail(f"Invalid polygon length at {label_path}:{line_number}")
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as exc:
                fail(f"Non-numeric YOLO label at {label_path}:{line_number}: {exc}")
            if class_id != 0:
                fail(f"Unexpected class {class_id} at {label_path}:{line_number}")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
                fail(f"Coordinate outside [0, 1] at {label_path}:{line_number}")
            points = list(zip(coordinates[0::2], coordinates[1::2]))
            if len(set(points)) < 3:
                fail(f"Degenerate polygon at {label_path}:{line_number}")
            instance_count += 1
    return len(image_paths), instance_count


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """Read the grouping manifest."""
    if not path.is_file():
        fail(f"Missing grouping manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def listed_stems(paths: List[Path]) -> Set[str]:
    """Return image stems from a resolved split list."""
    return {path.stem for path in paths}


def validate_conventional_dataset(root: Path, expected_report: Dict[str, object]) -> Dict[str, object]:
    """Validate the physical train/val/test dataset."""
    split_stats: Dict[str, Dict[str, int]] = {}
    split_stems: Dict[str, Set[str]] = {}
    for split in ("train", "val", "test"):
        images, instances = validate_yolo_labels(root / split / "images", root / split / "labels")
        split_stats[split] = {"images": images, "instances": instances}
        split_stems[split] = {path.stem for path in (root / split / "images").iterdir() if path.is_file()}
        expected = expected_report["split_7_2_1"][split]
        if split_stats[split] != expected:
            fail(f"7:2:1 {split} statistics differ: actual={split_stats[split]}, expected={expected}")

    for left, right in combinations(split_stems, 2):
        overlap = split_stems[left] & split_stems[right]
        if overlap:
            fail(f"7:2:1 overlap between {left} and {right}: {sorted(overlap)[:10]}")
    return split_stats


def validate_cv_dataset(root: Path, expected_report: Dict[str, object]) -> Dict[str, object]:
    """Validate all four folds, fixed test membership, and grouping constraints."""
    image_count, instance_count = validate_yolo_labels(root / "images", root / "labels")
    expected_images = sum(
        int(expected_report["split_7_2_1"][split]["images"])
        for split in ("train", "val", "test")
    )
    expected_instances = expected_report["label_conversion"]["effective_yolo_instances"]
    if image_count != expected_images or instance_count != expected_instances:
        fail(
            f"Shared CV data differ: images={image_count}, instances={instance_count}, "
            f"expected_images={expected_images}, expected_instances={expected_instances}"
        )

    manifest = read_manifest(root / "group_split_manifest.csv")
    if len(manifest) != image_count:
        fail(f"Manifest rows={len(manifest)} but images={image_count}")
    manifest_by_stem = {row["stem"]: row for row in manifest}
    if len(manifest_by_stem) != len(manifest):
        fail("Duplicate stems in group_split_manifest.csv")

    expected_by_fold = {
        int(item["fold"]): item for item in expected_report["cross_validation"]["folds"]
    }
    fold_sets: Dict[int, Dict[str, Set[str]]] = {}
    fold_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for fold in range(1, 5):
        yaml_path = root / f"fold_{fold}" / "data.yaml"
        yaml_text = yaml_path.read_text(encoding="utf-8-sig") if yaml_path.is_file() else ""
        for split in ("train", "val", "test"):
            expected_line = f"{split}: ../fold_{fold}_{split}.txt"
            if expected_line not in yaml_text:
                fail(f"Missing '{expected_line}' in {yaml_path}")

        split_paths = {
            split: read_image_list(root / f"fold_{fold}_{split}.txt", root)
            for split in ("train", "val", "test")
        }
        split_sets = {split: listed_stems(paths) for split, paths in split_paths.items()}
        fold_sets[fold] = split_sets
        for left, right in combinations(split_sets, 2):
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                fail(f"Fold {fold} overlap between {left} and {right}: {sorted(overlap)[:10]}")
        union = set().union(*split_sets.values())
        if union != set(manifest_by_stem):
            fail(f"Fold {fold} does not cover every shared image exactly once")

        stats = {}
        for split, stems in split_sets.items():
            stats[split] = {
                "images": len(stems),
                "instances": sum(int(manifest_by_stem[stem]["yolo_instances"]) for stem in stems),
            }
        fold_stats[f"fold_{fold}"] = stats
        expected = expected_by_fold[fold]
        for split in ("train", "val", "test"):
            expected_stats = {
                "images": int(expected[f"{split}_images"]),
                "instances": int(expected[f"{split}_instances"]),
            }
            if stats[split] != expected_stats:
                fail(
                    f"Fold {fold} {split} statistics differ: "
                    f"actual={stats[split]}, expected={expected_stats}"
                )

    fixed_test = fold_sets[1]["test"]
    for fold in range(2, 5):
        if fold_sets[fold]["test"] != fixed_test:
            fail(f"Fold {fold} test set differs from fold 1")

    validation_sets = [fold_sets[fold]["val"] for fold in range(1, 5)]
    for left, right in combinations(range(4), 2):
        overlap = validation_sets[left] & validation_sets[right]
        if overlap:
            fail(f"Validation folds {left + 1} and {right + 1} overlap: {sorted(overlap)[:10]}")
    validation_union = set().union(*validation_sets)
    development = {row["stem"] for row in manifest if row["cv_role"] == "development"}
    if validation_union != development:
        fail(
            f"Validation union differs from development set: "
            f"union={len(validation_union)}, development={len(development)}"
        )

    forced_train = {
        row["stem"] for row in manifest if row["cv_role"] == "forced_train_all_folds"
    }
    if len(forced_train) != expected_report["grouping"]["forced_train_images"]:
        fail("Forced-training image count differs from audit report")
    for fold in range(1, 5):
        if not forced_train <= fold_sets[fold]["train"]:
            fail(f"Fold {fold} is missing correlated images from training")
        if forced_train & (fold_sets[fold]["val"] | fold_sets[fold]["test"]):
            fail(f"Fold {fold} leaks correlated images into validation/test")

    return {
        "shared": {"images": image_count, "instances": instance_count},
        "fixed_test_images": len(fixed_test),
        "development_images": len(development),
        "forced_train_images": len(forced_train),
        "folds": fold_stats,
    }


def validate_datasets(yolo_root: Path = DEFAULT_YOLO_ROOT, cv_root: Path = DEFAULT_CV_ROOT) -> Dict[str, object]:
    """Run all integrity checks and write a machine-readable report."""
    audit_path = cv_root / "split_audit_report.json"
    if not audit_path.is_file():
        fail(f"Missing audit report: {audit_path}")
    expected_report = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if not expected_report["leakage_audit"]["passed"]:
        fail("Dataset builder leakage audit did not pass")

    result = {
        "passed": True,
        "conventional_7_2_1": validate_conventional_dataset(yolo_root, expected_report),
        "cross_validation": validate_cv_dataset(cv_root, expected_report),
    }
    output_path = cv_root / "cv_integrity_report.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-root", type=Path, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--cv-root", type=Path, default=DEFAULT_CV_ROOT)
    return parser


def main() -> None:
    """Run validation from the command line."""
    args = build_parser().parse_args()
    result = validate_datasets(args.yolo_root.resolve(), args.cv_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
