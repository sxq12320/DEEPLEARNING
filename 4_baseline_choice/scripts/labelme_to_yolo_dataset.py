"""Build the citrus YOLO segmentation dataset from LabelMe annotations.

The source dataset is expected to use the current orange_wuxi layout:
img/ + annotions_x/ and img_2/ + annotion_x_2/. Images from both subsets are
merged, shuffled, split by a 7:2:1 ratio, then exported to Ultralytics YOLO
polygon format. Augmentation helpers are available as optional CLI arguments
but are disabled by default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageEnhance
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(os.environ.get("ORANGE_LABELME_DIR", PROJECT_ROOT / "data" / "orange_wuxi"))
DEFAULT_OUTPUT = Path(os.environ.get("ORANGE_YOLO_DIR", PROJECT_ROOT / "data" / "orange_yolo"))
DEFAULT_SEED = 20260715
DEFAULT_CLASS_NAME = "orange_immature"
DEFAULT_RATIOS = {"train": 7.0, "val": 2.0, "test": 1.0}
DEFAULT_AUGMENTATIONS: Tuple[str, ...] = ()
DEFAULT_AUGMENT_SPLITS: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceRecord:
    """One source image and its matching LabelMe JSON path."""

    image_path: Path
    json_path: Path
    subset: str
    output_name: str


@dataclass(frozen=True)
class Polygon:
    """One normalized segmentation polygon."""

    xy: Tuple[float, ...]


def stable_seed(*parts: object) -> int:
    """Return a deterministic small seed from stable text parts."""
    text = "::".join(str(part) for part in parts)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % (2**32)
    return value


def clip(value: float, lower: float, upper: float) -> float:
    """Clip a value into a closed interval."""
    return min(max(value, lower), upper)


def polygon_area_pixels(points: Sequence[Tuple[float, float]]) -> float:
    """Compute polygon area in pixel coordinates."""
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, list(points[1:]) + [points[0]])
        )
    ) / 2.0


def find_records(source_root: Path) -> List[SourceRecord]:
    """Collect images from both current LabelMe subsets."""
    pairs = (
        ("img", source_root / "img", source_root / "annotions_x"),
        ("img_2", source_root / "img_2", source_root / "annotion_x_2"),
    )
    raw_records: List[Tuple[str, Path, Path]] = []
    for subset, image_dir, json_dir in pairs:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not json_dir.is_dir():
            raise FileNotFoundError(f"Missing annotation directory: {json_dir}")
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        raw_records.extend((subset, path, json_dir / f"{path.stem}.json") for path in images)

    name_counts = Counter(path.name.lower() for _, path, _ in raw_records)
    records: List[SourceRecord] = []
    for subset, image_path, json_path in raw_records:
        output_name = image_path.name
        if name_counts[image_path.name.lower()] > 1:
            output_name = f"{subset}_{image_path.name}"
        records.append(
            SourceRecord(
                image_path=image_path,
                json_path=json_path,
                subset=subset,
                output_name=output_name,
            )
        )
    return records


def split_counts_for_final_ratio(
    total: int,
    final_ratios: Dict[str, float],
    augment_splits: Iterable[str],
    augment_count: int,
) -> Dict[str, int]:
    """Compute source-image split counts so final augmented counts match ratios."""
    augment_splits = set(augment_splits)
    multipliers = {
        split: 1 + augment_count if split in augment_splits else 1
        for split in final_ratios
    }
    source_weights = {
        split: ratio / multipliers[split] for split, ratio in final_ratios.items()
    }
    scale = total / sum(source_weights.values())
    exact = {split: source_weights[split] * scale for split in final_ratios}
    counts = {split: int(exact[split]) for split in final_ratios}
    missing = total - sum(counts.values())
    order = sorted(final_ratios, key=lambda split: exact[split] - counts[split], reverse=True)
    for split in order[:missing]:
        counts[split] += 1
    return counts


def make_splits(
    records: Sequence[SourceRecord],
    seed: int,
    final_ratios: Dict[str, float],
    augment_splits: Iterable[str],
    augment_count: int,
) -> Dict[str, List[SourceRecord]]:
    """Shuffle source images and assign them to train/val/test."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    counts = split_counts_for_final_ratio(
        total=len(shuffled),
        final_ratios=final_ratios,
        augment_splits=augment_splits,
        augment_count=augment_count,
    )
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def rectangle_to_polygon(points: Sequence[Sequence[float]]) -> List[List[float]]:
    """Convert a LabelMe rectangle to four polygon points."""
    if len(points) != 2:
        return []
    x1, y1 = points[0]
    x2, y2 = points[1]
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def load_labelme_polygons(
    record: SourceRecord,
    width: int,
    height: int,
    class_name: str,
    report: Dict[str, object],
) -> List[Polygon]:
    """Read one LabelMe JSON file and return YOLO-normalized polygons."""
    if not record.json_path.exists():
        report["missing_json"].append(str(record.json_path))
        return []

    with record.json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    polygons: List[Polygon] = []
    ignored_labels: DefaultDict[str, int] = report["ignored_labels"]
    unsupported_shapes: DefaultDict[str, int] = report["unsupported_shapes"]
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if label != class_name:
            ignored_labels[label] += 1
            continue

        shape_type = str(shape.get("shape_type") or "polygon")
        points = shape.get("points") or []
        if shape_type == "rectangle":
            points = rectangle_to_polygon(points)
        elif shape_type != "polygon":
            unsupported_shapes[shape_type] += 1
            continue

        pixel_points: List[Tuple[float, float]] = []
        for point in points:
            if len(point) < 2:
                continue
            x = clip(float(point[0]), 0.0, float(width))
            y = clip(float(point[1]), 0.0, float(height))
            pixel_points.append((x, y))

        if len(pixel_points) < 3 or polygon_area_pixels(pixel_points) <= 0.5:
            report["degenerate_polygons"] += 1
            continue

        normalized: List[float] = []
        for x, y in pixel_points:
            normalized.extend((clip(x / width, 0.0, 1.0), clip(y / height, 0.0, 1.0)))
        polygons.append(Polygon(tuple(normalized)))
    return polygons


def yolo_label_text(polygons: Sequence[Polygon]) -> str:
    """Serialize polygons to YOLO segmentation text."""
    lines = []
    for polygon in polygons:
        coords = " ".join(f"{value:.6f}" for value in polygon.xy)
        lines.append(f"0 {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


def hflip_polygons(polygons: Sequence[Polygon]) -> List[Polygon]:
    """Horizontally flip normalized polygons."""
    flipped = []
    for polygon in polygons:
        coords = list(polygon.xy)
        for index in range(0, len(coords), 2):
            coords[index] = 1.0 - coords[index]
        flipped.append(Polygon(tuple(coords)))
    return flipped


def shifted_with_edge_fill(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift an image while filling exposed borders with edge pixels."""
    shifted = ImageChops.offset(image, dx, dy)
    width, height = image.size
    if dx > 0:
        fill = image.crop((0, 0, 1, height)).resize((dx, height))
        shifted.paste(fill, (0, 0))
    elif dx < 0:
        fill = image.crop((width - 1, 0, width, height)).resize((-dx, height))
        shifted.paste(fill, (width + dx, 0))
    if dy > 0:
        fill = image.crop((0, 0, width, 1)).resize((width, dy))
        shifted.paste(fill, (0, 0))
    elif dy < 0:
        fill = image.crop((0, height - 1, width, height)).resize((width, -dy))
        shifted.paste(fill, (0, height + dy))
    return shifted


def apply_motion_blur(image: Image.Image, seed: int) -> Image.Image:
    """Apply deterministic motion blur by averaging shifted frames."""
    rng = random.Random(seed)
    frames = rng.choice((5, 7, 9))
    step = rng.choice((2, 3, 4))
    direction = rng.choice(("horizontal", "vertical", "diag_down", "diag_up"))
    direction_xy = {
        "horizontal": (1, 0),
        "vertical": (0, 1),
        "diag_down": (1, 1),
        "diag_up": (1, -1),
    }[direction]
    radius = frames // 2
    blurred = image.copy()
    count = 1
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        dx = direction_xy[0] * offset * step
        dy = direction_xy[1] * offset * step
        shifted = shifted_with_edge_fill(image, dx, dy)
        blurred = Image.blend(blurred, shifted, 1.0 / (count + 1))
        count += 1
    return blurred


def apply_lighting(image: Image.Image, seed: int) -> Image.Image:
    """Apply deterministic brightness, contrast, and color perturbation."""
    rng = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.35))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.30))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))
    return image


def save_image(image: Image.Image, destination: Path) -> None:
    """Save an RGB image using stable settings."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(destination, quality=95)
    else:
        image.save(destination)


def write_sample(
    image: Image.Image,
    polygons: Sequence[Polygon],
    image_path: Path,
    label_path: Path,
) -> int:
    """Write one image-label pair and return its instance count."""
    save_image(image, image_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(yolo_label_text(polygons), encoding="utf-8")
    return len(polygons)


def write_augmented_sample(
    image: Image.Image,
    polygons: Sequence[Polygon],
    record: SourceRecord,
    split: str,
    augmentation: str,
    output_root: Path,
    seed: int,
) -> int:
    """Write one augmented image-label pair."""
    stem = Path(record.output_name).stem
    suffix = Path(record.output_name).suffix
    aug_name = f"{stem}__aug_{augmentation}{suffix}"
    aug_label = f"{stem}__aug_{augmentation}.txt"

    aug_image = image.copy()
    aug_polygons = list(polygons)
    aug_seed = stable_seed(seed, split, augmentation, record.output_name)
    if augmentation == "motion_blur":
        aug_image = apply_motion_blur(aug_image, aug_seed)
    elif augmentation == "hflip":
        aug_image = aug_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        aug_polygons = hflip_polygons(aug_polygons)
    elif augmentation == "lighting":
        aug_image = apply_lighting(aug_image, aug_seed)
    else:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    return write_sample(
        image=aug_image,
        polygons=aug_polygons,
        image_path=output_root / split / "images" / aug_name,
        label_path=output_root / split / "labels" / aug_label,
    )


def safe_overwrite_directory(output_root: Path) -> None:
    """Remove and recreate an output directory after basic path checks."""
    resolved = output_root.resolve()
    if resolved.anchor == str(resolved):
        raise ValueError(f"Refusing to overwrite drive root: {resolved}")
    if len(resolved.parts) < 3:
        raise ValueError(f"Refusing to overwrite a shallow path: {resolved}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_root: Path, class_name: str) -> None:
    """Write an Ultralytics data.yaml file."""
    text = (
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def convert_dataset(
    source_root: Path,
    output_root: Path,
    class_name: str,
    seed: int,
    final_ratios: Dict[str, float],
    augment_splits: Sequence[str],
    augmentations: Sequence[str],
) -> Dict[str, object]:
    """Convert LabelMe data into an augmented YOLO segmentation dataset."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Source and output directories must differ")

    records = find_records(source_root)
    if not records:
        raise FileNotFoundError(f"No images found under {source_root}")

    split_records = make_splits(
        records=records,
        seed=seed,
        final_ratios=final_ratios,
        augment_splits=augment_splits,
        augment_count=len(augmentations),
    )
    safe_overwrite_directory(output_root)

    report: Dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "class_name": class_name,
        "seed": seed,
        "target_final_ratio": final_ratios,
        "augment_splits": list(augment_splits),
        "augmentations": list(augmentations),
        "source_images": len(records),
        "source_split_counts": {split: len(items) for split, items in split_records.items()},
        "final_split_counts": {},
        "instances_by_split": {},
        "missing_json": [],
        "ignored_labels": defaultdict(int),
        "unsupported_shapes": defaultdict(int),
        "degenerate_polygons": 0,
    }

    for split, items in split_records.items():
        split_instances = 0
        final_images = 0
        for record in tqdm(
            items,
            desc=f"Write {split}",
            unit="image",
            dynamic_ncols=True,
        ):
            with Image.open(record.image_path) as image_handle:
                image = image_handle.convert("RGB")
            width, height = image.size
            polygons = load_labelme_polygons(record, width, height, class_name, report)

            output_image = output_root / split / "images" / record.output_name
            output_label = output_root / split / "labels" / f"{Path(record.output_name).stem}.txt"
            split_instances += write_sample(image, polygons, output_image, output_label)
            final_images += 1

            if split in augment_splits:
                for augmentation in augmentations:
                    split_instances += write_augmented_sample(
                        image=image,
                        polygons=polygons,
                        record=record,
                        split=split,
                        augmentation=augmentation,
                        output_root=output_root,
                        seed=seed,
                    )
                    final_images += 1

        report["instances_by_split"][split] = split_instances
        report["final_split_counts"][split] = final_images

    write_data_yaml(output_root, class_name)
    final_counts = report["final_split_counts"]
    test_count = max(int(final_counts.get("test", 0)), 1)
    report["actual_final_ratio_vs_test"] = {
        split: round(int(count) / test_count, 4) for split, count in final_counts.items()
    }

    report["ignored_labels"] = dict(report["ignored_labels"])
    report["unsupported_shapes"] = dict(report["unsupported_shapes"])
    report_path = output_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--augment-splits", nargs="+", default=list(DEFAULT_AUGMENT_SPLITS))
    parser.add_argument("--augmentations", nargs="+", default=list(DEFAULT_AUGMENTATIONS))
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    report = convert_dataset(
        source_root=args.source,
        output_root=args.output,
        class_name=args.class_name,
        seed=args.seed,
        final_ratios=DEFAULT_RATIOS,
        augment_splits=args.augment_splits,
        augmentations=args.augmentations,
    )
    summary = {
        "source_images": report["source_images"],
        "source_split_counts": report["source_split_counts"],
        "final_split_counts": report["final_split_counts"],
        "actual_final_ratio_vs_test": report["actual_final_ratio_vs_test"],
        "missing_json": len(report["missing_json"]),
        "degenerate_polygons": report["degenerate_polygons"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
