"""One-click visual audit for YOLO segmentation labels and predictions.

Green overlays are ground-truth polygons. Red overlays are model predictions.
Images are ranked by extra predictions first, which is useful for finding
possibly missing labels.
"""

from __future__ import annotations

import csv
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from ultralytics import YOLO


# =============================================================================
# USER SETTINGS: edit this block only
# =============================================================================

WEIGHTS = Path(r"Z:\001_1_yolov8nano-seg_adamw_yes\weights\best.pt")
DATA_YAML = Path(r"E:\mastercode\data\test\orange_wuxi_seg.yaml")
SPLIT = "all"  # train, val, test, or all
OUTPUT_DIR = Path(r"E:\mastercode\9_archive\low_confidence")

IMAGE_SIZE = 640
CONFIDENCE = 0.01
IOU = 0.70
MAX_DET = 300
DEVICE = "0"
RETINA_MASKS = False
INFERENCE_CHUNK_SIZE = 32

# Run inference on all images. Save every comparison by default for manual label review.
SAVE_ALL_IMAGES = True
SAVE_COUNT_MISMATCH_DIR = True
SAVE_TOP_N = 80
SAVE_RANDOM_N = 20
RANDOM_SEED = 42

# =============================================================================
# END USER SETTINGS
# =============================================================================


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides for one-click reuse across runs."""
    parser = argparse.ArgumentParser(description="Visual audit for YOLO segmentation labels and predictions.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--split", default=SPLIT, choices=["train", "val", "test", "all"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--retina-masks", action="store_true", default=RETINA_MASKS)
    parser.add_argument("--chunk-size", type=int, default=INFERENCE_CHUNK_SIZE)
    return parser.parse_args()


def resolve_split_paths(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Resolve image and label directories from an Ultralytics data YAML."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if split not in data:
        raise KeyError(f"Split '{split}' is not defined in {data_yaml}.")
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    image_dir = Path(data[split])
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if image_dir.parent.name == "images":
        label_dir = image_dir.parent.parent / "labels" / image_dir.name
    else:
        label_dir = Path(str(image_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
    return image_dir, label_dir


def image_files(image_dir: Path) -> list[Path]:
    """List image files in deterministic order."""
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def collect_samples(data_yaml: Path, split: str) -> list[dict[str, object]]:
    """Collect image paths and label directories for one split or all splits."""
    split = split.lower()
    if split == "all":
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        split_names = [name for name in ("train", "val", "test") if name in data]
    else:
        split_names = [split]

    samples = []
    for split_name in split_names:
        image_dir, label_dir = resolve_split_paths(data_yaml, split_name)
        for order, image_path in enumerate(image_files(image_dir), start=1):
            samples.append(
                {
                    "split": split_name,
                    "order": order,
                    "image_path": image_path,
                    "label_dir": label_dir,
                }
            )
    return samples


def read_yolo_segments(label_path: Path, width: int, height: int) -> list[list[tuple[float, float]]]:
    """Read normalized YOLO segmentation polygons as pixel coordinates."""
    if not label_path.is_file():
        return []
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 7:
            continue
        values = [float(value) for value in fields[1:]]
        points = []
        for index in range(0, len(values) - 1, 2):
            x = min(max(values[index] * width, 0.0), width - 1.0)
            y = min(max(values[index + 1] * height, 0.0), height - 1.0)
            points.append((x, y))
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def result_polygons(result) -> tuple[list[list[tuple[float, float]]], list[float]]:
    """Extract predicted mask polygons and confidence scores from one YOLO result."""
    if result.masks is None:
        return [], []
    polygons = []
    for segment in result.masks.xy:
        points = [(float(x), float(y)) for x, y in np.asarray(segment)]
        if len(points) >= 3:
            polygons.append(points)
    scores = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else [0.0] * len(polygons)
    return polygons, [float(score) for score in scores[: len(polygons)]]


def overlay_polygons(
    image: Image.Image,
    polygons: list[list[tuple[float, float]]],
    color: tuple[int, int, int],
    label: str,
) -> Image.Image:
    """Draw semi-transparent filled polygons and outlines."""
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = (*color, 80)
    outline = (*color, 255)
    for polygon in polygons:
        draw.polygon(polygon, fill=fill, outline=outline)
        x, y = polygon[0]
        draw.text((x, y), label, fill=outline)
    return Image.alpha_composite(canvas, layer).convert("RGB")


def add_title(image: Image.Image, title: str) -> Image.Image:
    """Add a compact title band above one panel."""
    band_height = 44
    output = Image.new("RGB", (image.width, image.height + band_height), (245, 245, 245))
    output.paste(image, (0, band_height))
    draw = ImageDraw.Draw(output)
    draw.text((12, 13), title, fill=(20, 20, 20), font=ImageFont.load_default())
    return output


def make_comparison(
    image_path: Path,
    label_dir: Path,
    pred_polygons: list[list[tuple[float, float]]],
    scores: list[float],
    output_path: Path,
) -> dict[str, object]:
    """Save original/ground-truth/prediction comparison image."""
    image = Image.open(image_path).convert("RGB")
    gt_polygons = read_yolo_segments(label_dir / f"{image_path.stem}.txt", image.width, image.height)
    count_diff = len(pred_polygons) - len(gt_polygons)
    gt_panel = overlay_polygons(image, gt_polygons, (0, 180, 0), "GT")
    pred_panel = overlay_polygons(image, pred_polygons, (220, 0, 0), "P")
    pred_text = f"Prediction red | count={len(pred_polygons)} | conf_max={max(scores, default=0.0):.2f}"
    panels = [
        add_title(image, f"Original | {image_path.name}"),
        add_title(gt_panel, f"Ground truth green | count={len(gt_polygons)}"),
        add_title(pred_panel, pred_text),
    ]
    target_height = 640
    resized = []
    for panel in panels:
        scale = target_height / panel.height
        resized.append(panel.resize((round(panel.width * scale), target_height)))
    comparison = Image.new("RGB", (sum(panel.width for panel in resized), target_height), (255, 255, 255))
    x_offset = 0
    for panel in resized:
        comparison.paste(panel, (x_offset, 0))
        x_offset += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, quality=92)
    return {
        "image": str(image_path),
        "output": str(output_path),
        "gt_count": len(gt_polygons),
        "pred_count": len(pred_polygons),
        "extra_predictions": count_diff,
        "missing_predictions": max(-count_diff, 0),
        "abs_count_diff": abs(count_diff),
        "max_conf": max(scores, default=0.0),
        "mean_conf": sum(scores) / max(len(scores), 1),
    }


def main() -> None:
    """Run inference and write ranked visual label-audit comparisons."""
    args = parse_args()
    samples = collect_samples(args.data_yaml, args.split)
    if not args.weights.is_file():
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not samples:
        raise FileNotFoundError(f"No images found for split setting: {args.split}")

    print("Label-audit settings")
    print("-" * 80)
    for key, value in {
        "weights": args.weights,
        "data_yaml": args.data_yaml,
        "split": args.split,
        "output_dir": args.output_dir,
        "images": len(samples),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "retina_masks": args.retina_masks,
        "chunk_size": args.chunk_size,
        "save_all_images": SAVE_ALL_IMAGES,
        "save_count_mismatch_dir": SAVE_COUNT_MISMATCH_DIR,
    }.items():
        print(f"{key:14s}: {value}")

    model = YOLO(str(args.weights))
    records = []
    with tqdm(total=len(samples), desc="Forward + rank", unit="image", dynamic_ncols=True) as progress:
        for start in range(0, len(samples), args.chunk_size):
            batch_samples = samples[start : start + args.chunk_size]
            predictions = model.predict(
                source=[str(sample["image_path"]) for sample in batch_samples],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                stream=True,
                verbose=False,
                retina_masks=args.retina_masks,
            )
            for batch_index, result in enumerate(predictions):
                sample = batch_samples[batch_index]
                image_path = sample["image_path"]
                label_dir = sample["label_dir"]
                height, width = result.orig_img.shape[:2]
                gt_count = len(read_yolo_segments(label_dir / f"{image_path.stem}.txt", width, height))
                pred_polygons, scores = result_polygons(result)
                records.append(
                    {
                        "split": sample["split"],
                        "order": sample["order"],
                        "image_path": image_path,
                        "label_dir": label_dir,
                        "pred_polygons": pred_polygons,
                        "scores": scores,
                        "gt_count": gt_count,
                        "pred_count": len(pred_polygons),
                        "extra_predictions": len(pred_polygons) - gt_count,
                        "max_conf": max(scores, default=0.0),
                    }
                )
                progress.update(1)

    ranked = sorted(
        records,
        key=lambda item: (item["extra_predictions"], item["pred_count"], item["max_conf"]),
        reverse=True,
    )
    if SAVE_ALL_IMAGES:
        selected = records
    else:
        random.seed(RANDOM_SEED)
        selected = ranked[:SAVE_TOP_N]
        remaining = [item for item in records if item not in selected]
        selected.extend(random.sample(remaining, min(SAVE_RANDOM_N, len(remaining))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for rank, item in enumerate(tqdm(selected, desc="Save comparisons", unit="image", dynamic_ncols=True), start=1):
        image_path = item["image_path"]
        prefix = f"{int(item['order']):06d}" if SAVE_ALL_IMAGES else f"rank{rank:03d}"
        subdir = str(item["split"]) if SAVE_ALL_IMAGES else "ranked"
        filename = (
            f"{prefix}_extra{int(item['extra_predictions']):+03d}_"
            f"gt{int(item['gt_count']):02d}_pred{int(item['pred_count']):02d}_{image_path.stem}.jpg"
        )
        row = make_comparison(
            image_path,
            item["label_dir"],
            item["pred_polygons"],
            item["scores"],
            args.output_dir / subdir / filename,
        )
        summary_rows.append({"split": item["split"], "order": item["order"], **row})

    fieldnames = [
        "split",
        "order",
        "image",
        "output",
        "gt_count",
        "pred_count",
        "extra_predictions",
        "missing_predictions",
        "abs_count_diff",
        "max_conf",
        "mean_conf",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (int(row["extra_predictions"]), int(row["abs_count_diff"]), float(row["max_conf"])),
        reverse=True,
    )
    with (args.output_dir / "summary_ranked.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)
    if SAVE_COUNT_MISMATCH_DIR:
        mismatch_dir = args.output_dir / "count_mismatch"
        mismatch_dir.mkdir(parents=True, exist_ok=True)
        mismatch_rows = sorted(
            (row for row in summary_rows if int(row["gt_count"]) != int(row["pred_count"])),
            key=lambda row: (Path(row["image"]).name.lower(), str(row["split"]), int(row["order"])),
        )
        copied_rows = []
        for order, row in enumerate(mismatch_rows, start=1):
            source = Path(row["output"])
            image_stem = Path(row["image"]).stem
            target = mismatch_dir / (
                f"{image_stem}_{row['split']}_gt{int(row['gt_count']):02d}_pred{int(row['pred_count']):02d}_"
                f"diff{int(row['abs_count_diff']):02d}_extra{int(row['extra_predictions']):+03d}.jpg"
            )
            shutil.copy2(source, target)
            copied_rows.append({**row, "mismatch_order": order, "mismatch_output": str(target)})
        with (mismatch_dir / "count_mismatch_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mismatch_order", *fieldnames, "mismatch_output"])
            writer.writeheader()
            writer.writerows(copied_rows)
    print(f"Saved {len(summary_rows)} comparisons: {args.output_dir}")
    print(f"Summary CSV: {args.output_dir / 'summary.csv'}")
    print(f"Ranked CSV : {args.output_dir / 'summary_ranked.csv'}")
    if SAVE_COUNT_MISMATCH_DIR:
        print(f"Count mismatch: {args.output_dir / 'count_mismatch'}")


if __name__ == "__main__":
    main()
