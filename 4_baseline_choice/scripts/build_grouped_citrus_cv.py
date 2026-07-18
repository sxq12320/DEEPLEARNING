"""Build leakage-controlled citrus YOLO splits and a four-fold CV dataset.

Correlated samples are identified from source names, exact file hashes, and
timestamp-adjacent perceptual hashes. Every multi-image group is restricted to
training. Validation and test sets contain independent singleton images only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from labelme_to_yolo_dataset import (
    DEFAULT_CLASS_NAME,
    load_labelme_polygons,
    yolo_label_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "orange_standardized"
DEFAULT_YOLO = PROJECT_ROOT / "data" / "orange_yolo"
DEFAULT_CV = PROJECT_ROOT / "data" / "orange_yolo_4fold"
DEFAULT_SEED = 20260718
DEFAULT_TEST_COUNT = 96
DEFAULT_VAL_COUNT = 193
DEFAULT_FOLDS = 4
DEFAULT_MAX_TIME_GAP = 5.0
DEFAULT_PHASH_DISTANCE = 12


@dataclass(frozen=True)
class Sample:
    """One standardized image and its annotation metadata."""

    stem: str
    image_name: str
    image_path: Path
    json_path: Path
    source_image: str
    source_name: str
    instances: int
    cohort: str
    timestamp: datetime | None


@dataclass(frozen=True)
class GroupInfo:
    """One final correlation group."""

    group_id: str
    members: Tuple[str, ...]
    reasons: Tuple[str, ...]


class UnionFind:
    """Minimal disjoint-set structure used to merge correlated samples."""

    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        """Return the canonical root for an item."""
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        """Merge two sets."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_timestamp(name: str) -> datetime | None:
    """Parse supported camera timestamp patterns from a source file name."""
    match = re.search(r"IMG_(\d{8})_(\d{6})", name, re.IGNORECASE)
    if not match:
        match = re.search(r"IMG(\d{8})(\d{6})", name, re.IGNORECASE)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def source_cohort(name: str) -> str:
    """Return a broad acquisition-source category for stratification."""
    lower = name.lower()
    if lower.startswith("video"):
        return "video"
    if lower.startswith("expand_image"):
        return "ai_generated"
    if re.search(r"IMG_2026\d{4}_", name, re.IGNORECASE):
        return "camera_2026"
    if re.search(r"IMG2023\d{10}", name, re.IGNORECASE):
        return "camera_2023"
    return "other"


def load_samples(source_root: Path) -> List[Sample]:
    """Read the standardization mapping and validate every renamed pair."""
    mapping_path = source_root / "rename_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing rename mapping: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    samples: List[Sample] = []
    for row in rows:
        image_path = source_root / "img" / row["new_image"]
        json_path = source_root / "labels" / f"{row['new_stem']}.json"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing standardized image: {image_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing standardized JSON: {json_path}")
        source_name = Path(row["source_image"]).name
        samples.append(
            Sample(
                stem=row["new_stem"],
                image_name=row["new_image"],
                image_path=image_path,
                json_path=json_path,
                source_image=row["source_image"],
                source_name=source_name,
                instances=int(row["instances"]),
                cohort=source_cohort(source_name),
                timestamp=parse_timestamp(source_name),
            )
        )

    if len({sample.stem for sample in samples}) != len(samples):
        raise ValueError("Duplicate standardized stems found in rename_mapping.csv")
    return samples


def sha256_file(path: Path) -> str:
    """Calculate the exact content hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Calculate a 64-bit DCT perceptual hash."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8]
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = (low_frequency > median).flatten()
    value = 0
    for index, enabled in enumerate(bits):
        if enabled:
            value |= 1 << index
    return value


def add_group_edges(
    samples: Sequence[Sample],
    max_time_gap: float,
    phash_distance: int,
) -> Tuple[List[GroupInfo], List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    """Build explicit and image-similarity correlation groups."""
    sample_by_stem = {sample.stem: sample for sample in samples}
    union_find = UnionFind(sample_by_stem)
    edges: List[Dict[str, object]] = []

    def add_edge(left: str, right: str, reason: str, detail: str = "") -> None:
        union_find.union(left, right)
        edges.append({"left": left, "right": right, "reason": reason, "detail": detail})

    explicit: DefaultDict[str, List[str]] = defaultdict(list)
    for sample in samples:
        video = re.match(r"(video\d+)", sample.source_name, re.IGNORECASE)
        burst = re.match(r"(IMG\d{14})_BURST", sample.source_name, re.IGNORECASE)
        if video:
            explicit[f"video:{video.group(1).lower()}"].append(sample.stem)
        if burst:
            explicit[f"burst:{burst.group(1).lower()}"].append(sample.stem)
        if sample.source_name.lower().startswith("expand_image_"):
            explicit["ai_generated:expand_image"].append(sample.stem)

    for key, members in sorted(explicit.items()):
        reason = key.split(":", 1)[0]
        for member in members[1:]:
            add_edge(members[0], member, reason, key)

    same_timestamp: DefaultDict[datetime, List[str]] = defaultdict(list)
    for sample in samples:
        if sample.timestamp is not None:
            same_timestamp[sample.timestamp].append(sample.stem)
    for timestamp, members in sorted(same_timestamp.items()):
        if len(members) > 1:
            for member in members[1:]:
                add_edge(members[0], member, "same_timestamp", timestamp.isoformat())

    exact_hashes: DefaultDict[str, List[str]] = defaultdict(list)
    phashes: Dict[str, int] = {}
    hash_metadata: Dict[str, Dict[str, object]] = {}
    for sample in tqdm(samples, desc="Hash images", unit="image", dynamic_ncols=True):
        exact_hash = sha256_file(sample.image_path)
        phash = perceptual_hash(sample.image_path)
        exact_hashes[exact_hash].append(sample.stem)
        phashes[sample.stem] = phash
        hash_metadata[sample.stem] = {
            "sha256": exact_hash,
            "phash_hex": f"{phash:016x}",
        }

    for exact_hash, members in exact_hashes.items():
        if len(members) > 1:
            for member in members[1:]:
                add_edge(members[0], member, "exact_duplicate", exact_hash)

    for cohort in ("camera_2026", "camera_2023"):
        timed = sorted(
            (sample.timestamp, sample.stem)
            for sample in samples
            if sample.cohort == cohort and sample.timestamp is not None
        )
        for index, (timestamp, left) in enumerate(timed):
            other_index = index + 1
            while other_index < len(timed):
                other_time, right = timed[other_index]
                gap = (other_time - timestamp).total_seconds()
                if gap > max_time_gap:
                    break
                distance = (phashes[left] ^ phashes[right]).bit_count()
                if distance <= phash_distance:
                    add_edge(
                        left,
                        right,
                        "temporal_phash",
                        f"gap_seconds={gap:.1f};phash_distance={distance}",
                    )
                other_index += 1

    members_by_root: DefaultDict[str, List[str]] = defaultdict(list)
    for sample in samples:
        members_by_root[union_find.find(sample.stem)].append(sample.stem)

    reasons_by_root: DefaultDict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        root = union_find.find(str(edge["left"]))
        reasons_by_root[root].add(str(edge["reason"]))

    groups: List[GroupInfo] = []
    ordered_groups = sorted(
        members_by_root.values(),
        key=lambda members: min(members),
    )
    for index, members in enumerate(ordered_groups):
        root = union_find.find(members[0])
        groups.append(
            GroupInfo(
                group_id=f"G{index:04d}",
                members=tuple(sorted(members)),
                reasons=tuple(sorted(reasons_by_root[root])),
            )
        )
    return groups, edges, hash_metadata


def instance_bin(instances: int) -> str:
    """Map image-level instance counts into stable stratification bins."""
    if instances == 0:
        return "0"
    if instances == 1:
        return "1"
    if instances <= 3:
        return "2-3"
    if instances <= 7:
        return "4-7"
    if instances <= 15:
        return "8-15"
    return "16+"


def stratum(sample: Sample) -> Tuple[str, str]:
    """Return the sampling stratum for one independent image."""
    return sample.cohort, instance_bin(sample.instances)


def stable_seed(seed: int, *parts: object) -> int:
    """Derive a deterministic random seed without Python hash randomization."""
    text = "::".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def stratified_select(samples: Sequence[Sample], count: int, seed: int) -> Tuple[List[Sample], List[Sample]]:
    """Select an exact-size subset while preserving cohort and density strata."""
    if count < 0 or count > len(samples):
        raise ValueError(f"Cannot select {count} items from {len(samples)} samples")

    buckets: DefaultDict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[stratum(sample)].append(sample)

    exact = {key: len(items) * count / len(samples) for key, items in buckets.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    remainder_order = sorted(
        buckets,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1

    selected: List[Sample] = []
    leftover: List[Sample] = []
    for key in sorted(buckets):
        items = list(buckets[key])
        random.Random(stable_seed(seed, *key)).shuffle(items)
        selected.extend(items[: quotas[key]])
        leftover.extend(items[quotas[key] :])
    selected.sort(key=lambda sample: sample.stem)
    leftover.sort(key=lambda sample: sample.stem)
    return selected, leftover


def make_validation_folds(
    samples: Sequence[Sample],
    folds: int,
    seed: int,
) -> List[List[Sample]]:
    """Distribute independent development images across balanced validation folds."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    if len(samples) % folds != 0:
        raise ValueError(
            f"Development singleton count {len(samples)} must be divisible by {folds} "
            "for equal-size folds"
        )

    capacity = len(samples) // folds
    fold_items: List[List[Sample]] = [[] for _ in range(folds)]
    fold_instances = [0] * folds
    fold_strata: List[Counter[Tuple[str, str]]] = [Counter() for _ in range(folds)]

    buckets: DefaultDict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[stratum(sample)].append(sample)

    for key in sorted(buckets):
        items = list(buckets[key])
        random.Random(stable_seed(seed, "fold", *key)).shuffle(items)
        items.sort(key=lambda sample: sample.instances, reverse=True)
        for sample in items:
            available = [index for index in range(folds) if len(fold_items[index]) < capacity]
            chosen = min(
                available,
                key=lambda index: (
                    fold_strata[index][key],
                    fold_instances[index],
                    len(fold_items[index]),
                    index,
                ),
            )
            fold_items[chosen].append(sample)
            fold_instances[chosen] += sample.instances
            fold_strata[chosen][key] += 1

    for items in fold_items:
        items.sort(key=lambda sample: sample.stem)
    return fold_items


def build_yolo_labels(
    samples: Sequence[Sample],
    label_root: Path,
    class_name: str,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    """Convert all standardized LabelMe JSON files to shared YOLO polygons."""
    report: Dict[str, object] = {
        "missing_json": [],
        "ignored_labels": defaultdict(int),
        "unsupported_shapes": defaultdict(int),
        "degenerate_polygons": 0,
        "duplicate_bbox_polygons_removed": [],
    }
    effective_counts: Dict[str, int] = {}
    label_root.mkdir(parents=True, exist_ok=True)
    for sample in tqdm(samples, desc="Convert labels", unit="image", dynamic_ncols=True):
        with Image.open(sample.image_path) as image:
            width, height = image.size
        record = type(
            "Record",
            (),
            {"json_path": sample.json_path},
        )()
        polygons = load_labelme_polygons(record, width, height, class_name, report)
        if len(polygons) != sample.instances:
            raise ValueError(
                f"Instance count changed for {sample.stem}: mapping={sample.instances}, converted={len(polygons)}"
            )
        unique_polygons = []
        seen_boxes: Dict[Tuple[float, float, float, float], int] = {}
        for polygon_index, polygon in enumerate(polygons):
            points = np.asarray(polygon.xy, dtype=np.float32).reshape(-1, 2)
            box_key = (
                round(float(points[:, 0].min()), 6),
                round(float(points[:, 1].min()), 6),
                round(float(points[:, 0].max()), 6),
                round(float(points[:, 1].max()), 6),
            )
            if box_key in seen_boxes:
                report["duplicate_bbox_polygons_removed"].append(
                    {
                        "stem": sample.stem,
                        "kept_polygon_index": seen_boxes[box_key],
                        "removed_polygon_index": polygon_index,
                        "normalized_xyxy": box_key,
                    }
                )
                continue
            seen_boxes[box_key] = polygon_index
            unique_polygons.append(polygon)
        effective_counts[sample.stem] = len(unique_polygons)
        (label_root / f"{sample.stem}.txt").write_text(
            yolo_label_text(unique_polygons),
            encoding="utf-8",
        )
    report["ignored_labels"] = dict(report["ignored_labels"])
    report["unsupported_shapes"] = dict(report["unsupported_shapes"])
    report["source_instances"] = sum(sample.instances for sample in samples)
    report["effective_yolo_instances"] = sum(effective_counts.values())
    report["duplicate_bbox_count"] = len(report["duplicate_bbox_polygons_removed"])
    return report, effective_counts


def copy_pair(sample: Sample, image_dir: Path, label_dir: Path, shared_labels: Path) -> None:
    """Copy one image and its generated YOLO label."""
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, image_dir / sample.image_name)
    shutil.copy2(shared_labels / f"{sample.stem}.txt", label_dir / f"{sample.stem}.txt")


def write_conventional_dataset(
    output_root: Path,
    split_samples: Dict[str, Sequence[Sample]],
    shared_labels: Path,
    class_name: str,
) -> None:
    """Write the conventional physical train/val/test directory layout."""
    for split, samples in split_samples.items():
        for sample in tqdm(samples, desc=f"Write YOLO {split}", unit="image", dynamic_ncols=True):
            copy_pair(
                sample,
                output_root / split / "images",
                output_root / split / "labels",
                shared_labels,
            )
    yaml_text = (
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_image_list(path: Path, samples: Sequence[Sample]) -> None:
    """Write portable image paths relative to the list-file directory."""
    lines = [f"./images/{sample.image_name}" for sample in sorted(samples, key=lambda item: item.stem)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cv_dataset(
    output_root: Path,
    samples: Sequence[Sample],
    forced_train: Sequence[Sample],
    test_samples: Sequence[Sample],
    validation_folds: Sequence[Sequence[Sample]],
    shared_labels: Path,
    class_name: str,
    effective_counts: Dict[str, int],
) -> List[Dict[str, object]]:
    """Write one shared dataset and portable manifests for all CV folds."""
    image_root = output_root / "images"
    label_root = output_root / "labels"
    for sample in tqdm(samples, desc="Write CV shared data", unit="image", dynamic_ncols=True):
        copy_pair(sample, image_root, label_root, shared_labels)

    test_stems = {sample.stem for sample in test_samples}
    forced_stems = {sample.stem for sample in forced_train}
    development = [
        sample for sample in samples if sample.stem not in test_stems and sample.stem not in forced_stems
    ]
    fold_reports: List[Dict[str, object]] = []

    for fold_index, val_samples in enumerate(validation_folds, start=1):
        val_stems = {sample.stem for sample in val_samples}
        train_samples = [
            sample
            for sample in samples
            if sample.stem not in test_stems and sample.stem not in val_stems
        ]
        if len(train_samples) != len(forced_train) + len(development) - len(val_samples):
            raise AssertionError("Unexpected CV training count")

        write_image_list(output_root / f"fold_{fold_index}_train.txt", train_samples)
        write_image_list(output_root / f"fold_{fold_index}_val.txt", val_samples)
        write_image_list(output_root / f"fold_{fold_index}_test.txt", test_samples)

        fold_dir = output_root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        yaml_text = (
            f"train: ../fold_{fold_index}_train.txt\n"
            f"val: ../fold_{fold_index}_val.txt\n"
            f"test: ../fold_{fold_index}_test.txt\n"
            "names:\n"
            f"  0: {class_name}\n"
        )
        (fold_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
        fold_reports.append(
            {
                "fold": fold_index,
                "train_images": len(train_samples),
                "val_images": len(val_samples),
                "test_images": len(test_samples),
                "train_instances": sum(effective_counts[sample.stem] for sample in train_samples),
                "val_instances": sum(effective_counts[sample.stem] for sample in val_samples),
                "test_instances": sum(effective_counts[sample.stem] for sample in test_samples),
            }
        )
    return fold_reports


def verify_split(
    groups: Sequence[GroupInfo],
    assignment: Dict[str, str],
    allowed_multi_role: str = "train",
) -> List[Dict[str, object]]:
    """Return any group leakage or forbidden multi-group placement."""
    violations: List[Dict[str, object]] = []
    for group in groups:
        roles = sorted({assignment[member] for member in group.members})
        if len(roles) > 1:
            violations.append(
                {
                    "group_id": group.group_id,
                    "type": "cross_split",
                    "members": list(group.members),
                    "roles": roles,
                }
            )
        if len(group.members) > 1 and roles != [allowed_multi_role]:
            violations.append(
                {
                    "group_id": group.group_id,
                    "type": "multi_group_outside_train",
                    "members": list(group.members),
                    "roles": roles,
                }
            )
    return violations


def ensure_safe_destination(path: Path, data_root: Path) -> None:
    """Ensure an overwrite destination is a direct child of the expected data root."""
    resolved = path.resolve()
    expected_parent = data_root.resolve()
    if resolved.parent != expected_parent:
        raise ValueError(f"Refusing to overwrite unexpected path: {resolved}")
    if resolved.name not in {"orange_yolo", "orange_yolo_4fold"}:
        raise ValueError(f"Refusing to overwrite unapproved dataset directory: {resolved}")


def replace_directory(staging: Path, destination: Path, data_root: Path) -> None:
    """Replace an approved dataset directory with a completed staging directory."""
    ensure_safe_destination(destination, data_root)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(staging, destination)
    shutil.rmtree(staging)


def write_audit_files(
    audit_root: Path,
    samples: Sequence[Sample],
    groups: Sequence[GroupInfo],
    edges: Sequence[Dict[str, object]],
    hashes: Dict[str, Dict[str, object]],
    split_721: Dict[str, str],
    fixed_test: Set[str],
    fold_by_stem: Dict[str, int],
    effective_counts: Dict[str, int],
    report: Dict[str, object],
) -> None:
    """Write detailed, human-auditable grouping and split manifests."""
    audit_root.mkdir(parents=True, exist_ok=True)
    group_by_stem = {
        member: group
        for group in groups
        for member in group.members
    }
    with (audit_root / "group_split_manifest.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        fieldnames = [
            "stem",
            "image",
            "source_name",
            "source_image",
            "source_instances",
            "yolo_instances",
            "cohort",
            "timestamp",
            "group_id",
            "group_size",
            "group_reasons",
            "forced_train",
            "split_7_2_1",
            "cv_role",
            "validation_fold",
            "sha256",
            "phash_hex",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: item.stem):
            group = group_by_stem[sample.stem]
            if sample.stem in fixed_test:
                cv_role = "fixed_test"
                validation_fold = ""
            elif len(group.members) > 1:
                cv_role = "forced_train_all_folds"
                validation_fold = ""
            else:
                cv_role = "development"
                validation_fold = fold_by_stem[sample.stem]
            writer.writerow(
                {
                    "stem": sample.stem,
                    "image": sample.image_name,
                    "source_name": sample.source_name,
                    "source_image": sample.source_image,
                    "source_instances": sample.instances,
                    "yolo_instances": effective_counts[sample.stem],
                    "cohort": sample.cohort,
                    "timestamp": sample.timestamp.isoformat() if sample.timestamp else "",
                    "group_id": group.group_id,
                    "group_size": len(group.members),
                    "group_reasons": "|".join(group.reasons),
                    "forced_train": len(group.members) > 1,
                    "split_7_2_1": split_721[sample.stem],
                    "cv_role": cv_role,
                    "validation_fold": validation_fold,
                    "sha256": hashes[sample.stem]["sha256"],
                    "phash_hex": hashes[sample.stem]["phash_hex"],
                }
            )

    with (audit_root / "group_edges.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["left", "right", "reason", "detail"])
        writer.writeheader()
        writer.writerows(edges)

    (audit_root / "split_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_datasets(
    source_root: Path = DEFAULT_SOURCE,
    yolo_output: Path = DEFAULT_YOLO,
    cv_output: Path = DEFAULT_CV,
    class_name: str = DEFAULT_CLASS_NAME,
    seed: int = DEFAULT_SEED,
    test_count: int = DEFAULT_TEST_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    folds: int = DEFAULT_FOLDS,
    max_time_gap: float = DEFAULT_MAX_TIME_GAP,
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
) -> Dict[str, object]:
    """Build and verify the conventional and four-fold datasets."""
    source_root = source_root.resolve()
    yolo_output = yolo_output.resolve()
    cv_output = cv_output.resolve()
    data_root = source_root.parent
    ensure_safe_destination(yolo_output, data_root)
    ensure_safe_destination(cv_output, data_root)

    samples = load_samples(source_root)
    groups, edges, hashes = add_group_edges(samples, max_time_gap, phash_distance)
    group_by_stem = {
        member: group
        for group in groups
        for member in group.members
    }
    forced_train = [
        sample for sample in samples if len(group_by_stem[sample.stem].members) > 1
    ]
    independent = [
        sample for sample in samples if len(group_by_stem[sample.stem].members) == 1
    ]

    test_samples, development = stratified_select(independent, test_count, stable_seed(seed, "test"))
    val_samples, independent_train = stratified_select(
        development,
        val_count,
        stable_seed(seed, "val"),
    )
    train_samples = sorted([*forced_train, *independent_train], key=lambda sample: sample.stem)
    split_samples = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }
    split_721 = {
        sample.stem: split
        for split, split_items in split_samples.items()
        for sample in split_items
    }
    violations_721 = verify_split(groups, split_721)
    if violations_721:
        raise AssertionError(f"7:2:1 leakage detected: {violations_721[:3]}")

    validation_folds = make_validation_folds(development, folds, stable_seed(seed, "cv"))
    fold_by_stem = {
        sample.stem: fold_index
        for fold_index, fold_items in enumerate(validation_folds, start=1)
        for sample in fold_items
    }
    fixed_test_stems = {sample.stem for sample in test_samples}

    cv_violations: List[Dict[str, object]] = []
    for fold_index, fold_val in enumerate(validation_folds, start=1):
        val_stems = {sample.stem for sample in fold_val}
        assignment = {}
        for sample in samples:
            if sample.stem in fixed_test_stems:
                assignment[sample.stem] = "test"
            elif sample.stem in val_stems:
                assignment[sample.stem] = "val"
            else:
                assignment[sample.stem] = "train"
        for violation in verify_split(groups, assignment):
            violation["fold"] = fold_index
            cv_violations.append(violation)
    if cv_violations:
        raise AssertionError(f"CV leakage detected: {cv_violations[:3]}")

    staging_yolo = data_root / f".orange_yolo_build_{os.getpid()}"
    staging_cv = data_root / f".orange_yolo_4fold_build_{os.getpid()}"
    for staging in (staging_yolo, staging_cv):
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

    try:
        shared_labels = staging_cv / "_generated_labels"
        label_report, effective_counts = build_yolo_labels(samples, shared_labels, class_name)
        write_conventional_dataset(
            staging_yolo,
            split_samples,
            shared_labels,
            class_name,
        )
        fold_reports = write_cv_dataset(
            staging_cv,
            samples,
            forced_train,
            test_samples,
            validation_folds,
            shared_labels,
            class_name,
            effective_counts,
        )
        shutil.rmtree(shared_labels)

        report: Dict[str, object] = {
            "source_root": str(source_root),
            "seed": seed,
            "class_name": class_name,
            "grouping": {
                "max_time_gap_seconds": max_time_gap,
                "phash_distance_threshold": phash_distance,
                "groups_total": len(groups),
                "singleton_groups": sum(len(group.members) == 1 for group in groups),
                "multi_image_groups": sum(len(group.members) > 1 for group in groups),
                "forced_train_images": len(forced_train),
                "group_size_histogram": dict(
                    sorted(Counter(len(group.members) for group in groups).items())
                ),
                "edge_reason_counts": dict(Counter(str(edge["reason"]) for edge in edges)),
            },
            "split_7_2_1": {
                split: {
                    "images": len(items),
                    "instances": sum(effective_counts[sample.stem] for sample in items),
                }
                for split, items in split_samples.items()
            },
            "cross_validation": {
                "design": (
                    "Fixed independent test set; four balanced validation folds over the remaining "
                    "independent images; all correlated multi-image groups remain in training."
                ),
                "folds": fold_reports,
            },
            "label_conversion": label_report,
            "leakage_audit": {
                "split_7_2_1_violations": violations_721,
                "cross_validation_violations": cv_violations,
                "passed": not violations_721 and not cv_violations,
            },
        }

        write_audit_files(
            staging_yolo,
            samples,
            groups,
            edges,
            hashes,
            split_721,
            fixed_test_stems,
            fold_by_stem,
            effective_counts,
            report,
        )
        write_audit_files(
            staging_cv,
            samples,
            groups,
            edges,
            hashes,
            split_721,
            fixed_test_stems,
            fold_by_stem,
            effective_counts,
            report,
        )

        replace_directory(staging_yolo, yolo_output, data_root)
        replace_directory(staging_cv, cv_output, data_root)
        return report
    except Exception:
        for staging in (staging_yolo, staging_cv):
            if staging.exists():
                shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--yolo-output", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--cv-output", type=Path, default=DEFAULT_CV)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--max-time-gap", type=float, default=DEFAULT_MAX_TIME_GAP)
    parser.add_argument("--phash-distance", type=int, default=DEFAULT_PHASH_DISTANCE)
    return parser


def main() -> None:
    """Run dataset generation."""
    args = build_parser().parse_args()
    report = build_datasets(
        source_root=args.source,
        yolo_output=args.yolo_output,
        cv_output=args.cv_output,
        class_name=args.class_name,
        seed=args.seed,
        test_count=args.test_count,
        val_count=args.val_count,
        folds=args.folds,
        max_time_gap=args.max_time_gap,
        phash_distance=args.phash_distance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
