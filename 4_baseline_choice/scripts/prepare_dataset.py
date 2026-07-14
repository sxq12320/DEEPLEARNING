"""Convert YOLO polygon labels into all formats used by the baseline suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw

from baseline_common import save_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Polygon:
    """One validated polygon annotation."""

    class_id: int
    normalized_xy: Tuple[float, ...]


def parse_label_file(path: Path, num_classes: int = 1) -> List[Polygon]:
    """Parse one Ultralytics polygon label file."""
    polygons: List[Polygon] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return polygons
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"{path}:{line_number}: expected class plus at least three x/y points")
        class_id = int(fields[0])
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{path}:{line_number}: class id {class_id} is outside [0, {num_classes})")
        coordinates = tuple(float(value) for value in fields[1:])
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: normalized coordinates must be in [0, 1]")
        polygons.append(Polygon(class_id=class_id, normalized_xy=coordinates))
    return polygons


def polygon_pixels(polygon: Polygon, width: int, height: int) -> List[float]:
    """Convert normalized coordinates to clipped pixel coordinates."""
    points: List[float] = []
    for index in range(0, len(polygon.normalized_xy), 2):
        x = min(max(polygon.normalized_xy[index] * width, 0.0), float(width - 1))
        y = min(max(polygon.normalized_xy[index + 1] * height, 0.0), float(height - 1))
        points.extend((x, y))
    return points


def polygon_area(points: Sequence[float]) -> float:
    """Compute polygon area with the shoelace formula."""
    pairs = list(zip(points[0::2], points[1::2]))
    return abs(
        sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pairs, pairs[1:] + pairs[:1]))
    ) / 2.0


def polygon_bbox(points: Sequence[float]) -> List[float]:
    """Return a COCO x/y/width/height box."""
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    """Materialize a file using copy, hardlink, or automatic fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(source, destination)
        return "copy"


def image_files(directory: Path) -> List[Path]:
    """Return supported images in deterministic order."""
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def short_checksum(path: Path) -> str:
    """Return a short SHA-256 checksum for split manifests."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def convert_split(
    source_root: Path,
    output_root: Path,
    split: str,
    class_names: Sequence[str],
    mode: str,
    next_image_id: int,
    next_annotation_id: int,
) -> Tuple[Dict[str, object], Dict[str, object], int, int]:
    """Convert one split and return its report and COCO dictionary."""
    source_images = source_root / "images" / split
    source_labels = source_root / "labels" / split
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"Missing images/labels directories for split '{split}' under {source_root}")

    images = image_files(source_images)
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate image stems in {source_images}; labels are stem-based")

    coco_images: List[Dict[str, object]] = []
    coco_annotations: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    instances = 0
    negatives = 0
    materialization: Dict[str, int] = {"copy": 0, "hardlink": 0}

    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        polygons = parse_label_file(label_path, num_classes=len(class_names))
        if not polygons:
            negatives += 1

        targets = (
            output_root / "yolo" / "images" / split / image_path.name,
            output_root / "coco" / "images" / split / image_path.name,
            output_root / "rfdetr" / ("valid" if split == "val" else split) / image_path.name,
            output_root / "semantic" / "images" / split / image_path.name,
        )
        for target in targets:
            used_mode = link_or_copy(image_path, target, mode)
            materialization[used_mode] += 1

        yolo_label = output_root / "yolo" / "labels" / split / label_path.name
        link_or_copy(label_path, yolo_label, mode)

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            points = polygon_pixels(polygon, width, height)
            xy_pairs = list(zip(points[0::2], points[1::2]))
            draw.polygon(xy_pairs, fill=255)
            area = polygon_area(points)
            if area <= 0:
                raise ValueError(f"Degenerate polygon in {label_path}")
            coco_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": polygon.class_id + 1,
                    "segmentation": [points],
                    "area": area,
                    "bbox": polygon_bbox(points),
                    "iscrowd": 0,
                }
            )
            next_annotation_id += 1
            instances += 1

        mask_path = output_root / "semantic" / "masks" / split / f"{image_path.stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path)

        coco_images.append(
            {
                "id": next_image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        manifest.append(
            {
                "image_id": next_image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "instances": len(polygons),
                "image_sha256_16": short_checksum(image_path),
            }
        )
        next_image_id += 1

    categories = [{"id": index + 1, "name": name, "supercategory": "citrus"} for index, name in enumerate(class_names)]
    coco = {
        "info": {"description": "Immature citrus instance segmentation baseline dataset"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    report = {
        "split": split,
        "images": len(images),
        "instances": instances,
        "negative_images": negatives,
        "materialized_image_files": materialization,
        "manifest": manifest,
    }
    return report, coco, next_image_id, next_annotation_id


def write_format_metadata(output_root: Path, class_names: Sequence[str], split_coco: Dict[str, Dict[str, object]]) -> None:
    """Write annotations and portable metadata files."""
    yolo_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    import yaml

    with (output_root / "yolo" / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(yolo_yaml, handle, allow_unicode=True, sort_keys=False)

    for split, coco in split_coco.items():
        coco_path = output_root / "coco" / "annotations" / f"instances_{split}.json"
        save_json(coco_path, coco)
        rfdetr_split = "valid" if split == "val" else split
        save_json(output_root / "rfdetr" / rfdetr_split / "_annotations.coco.json", coco)


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    class_names: Sequence[str],
    splits: Iterable[str] = DEFAULT_SPLITS,
    mode: str = "auto",
) -> Dict[str, object]:
    """Prepare every framework layout from one source dataset."""
    if source_root.resolve() == output_root.resolve():
        raise ValueError("Source and output directories must differ")
    output_root.mkdir(parents=True, exist_ok=True)
    split_reports: List[Dict[str, object]] = []
    split_coco: Dict[str, Dict[str, object]] = {}
    image_id = 1
    annotation_id = 1
    for split in splits:
        report, coco, image_id, annotation_id = convert_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            class_names=class_names,
            mode=mode,
            next_image_id=image_id,
            next_annotation_id=annotation_id,
        )
        split_reports.append(report)
        split_coco[split] = coco
    write_format_metadata(output_root, class_names, split_coco)
    summary = {
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "class_names": list(class_names),
        "mode_requested": mode,
        "splits": split_reports,
        "totals": {
            "images": sum(int(report["images"]) for report in split_reports),
            "instances": sum(int(report["instances"]) for report in split_reports),
            "negative_images": sum(int(report["negative_images"]) for report in split_reports),
        },
    }
    save_json(output_root / "dataset_report.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source root containing images/<split> and labels/<split>.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--class-name", action="append", default=None, help="Class name; repeat for multiple classes.")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Splits to convert.")
    parser.add_argument(
        "--mode",
        choices=("auto", "copy", "hardlink"),
        default="auto",
        help="auto tries hardlinks first and falls back to copies.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    class_names = args.class_name or ["orange_immature"]
    summary = prepare_dataset(args.source, args.output, class_names, args.splits, args.mode)
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

