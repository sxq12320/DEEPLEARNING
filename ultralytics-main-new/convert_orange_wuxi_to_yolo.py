"""Convert LabelMe-style citrus annotations to a YOLO segmentation dataset.

Default input:
    E:/mastercode/data/orange_wuxi   (reads BOTH batches: annotions_x/img and annotion_x_2/img_2)

Default output:
    E:/mastercode/data/test

Run:
    python convert_orange_wuxi_to_yolo.py --overwrite

Notes:
- Reads every (annotation, image) batch listed in BATCHES; missing batches are skipped.
- An image with NO matching JSON (or a JSON with no kept shapes) is written as a
  background/negative sample: the image is copied and an EMPTY label .txt is emitted.
- Video files or other non-image extensions in the image dirs are ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

# (annotation_dir, image_dir) pairs, relative to --src. Add future batches here.
BATCHES = [("annotions_x", "img"), ("annotion_x_2", "img_2")]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert orange_wuxi LabelMe polygons to YOLO-seg format.")
    parser.add_argument("--src", default="E:/mastercode/data/orange_wuxi", help="Source dataset root.")
    parser.add_argument("--out", default="E:/mastercode/data/test", help="Output YOLO dataset root.")
    parser.add_argument("--class-name", default="orange_immature", help="Class label to keep.")
    parser.add_argument("--seed", type=int, default=20260708, help="Random split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated images/labels/yaml/report in output.")
    return parser.parse_args()


def clear_generated_outputs(out_root: Path) -> None:
    for name in ("images", "labels"):
        path = out_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("orange_wuxi_seg.yaml", "conversion_report.json"):
        path = out_root / name
        if path.exists():
            path.unlink()


def find_image(ann_dir: Path, img_dir: Path, ann_path: Path, image_path: str | None) -> Path:
    if image_path:
        candidate = (ann_dir / image_path).resolve()
        if candidate.exists():
            return candidate

    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = img_dir / f"{ann_path.stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No matching image found for {ann_path.name}")


def polygon_to_yolo(points: list[list[float]], width: int, height: int) -> list[float]:
    values: list[float] = []
    for point in points:
        if len(point) < 2:
            continue
        x = min(max(float(point[0]) / float(width), 0.0), 1.0)
        y = min(max(float(point[1]) / float(height), 0.0), 1.0)
        values.extend([x, y])
    return values


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    out_root = Path(args.out)

    present_batches: list[tuple[Path, Path]] = []
    for ann_name, img_name in BATCHES:
        ann_dir, img_dir = src_root / ann_name, src_root / img_name
        if ann_dir.exists() and img_dir.exists():
            present_batches.append((ann_dir, img_dir))
        elif ann_dir.exists() or img_dir.exists():
            print(f"[warn] incomplete batch skipped: {ann_name} / {img_name}")
    if not present_batches:
        raise FileNotFoundError(f"No (annotation, image) batch found under {src_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_generated_outputs(out_root)
    elif any(out_root.iterdir()):
        raise SystemExit(f"{out_root} is not empty. Use --overwrite if you want to rebuild generated files.")

    class_to_id = {args.class_name: 0}

    # item = (ann_path | None, image_path). ann_path is None => background/negative image
    # (no JSON) -> copied with an empty label file.
    items: list[tuple[Path | None, Path]] = []
    n_negatives = 0
    for ann_dir, img_dir in present_batches:
        annotated_stems: set[str] = set()
        for ann_path in sorted(ann_dir.glob("*.json")):
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            image = find_image(ann_dir, img_dir, ann_path, data.get("imagePath"))
            items.append((ann_path, image))
            annotated_stems.add(image.stem)
        # images with no JSON at all -> user-confirmed negatives (no orange_immature)
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS and img_path.stem not in annotated_stems:
                items.append((None, img_path))
                n_negatives += 1

    random.Random(args.seed).shuffle(items)
    n_total = len(items)
    n_train = round(n_total * args.train_ratio)
    n_val = round(n_total * args.val_ratio)
    assignments = {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }

    for split in assignments:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {split: {"images": 0, "instances": 0, "empty": 0, "skipped_shapes": 0} for split in assignments}

    for split, split_items in assignments.items():
        for ann_path, image_path in split_items:
            label_lines: list[str] = []
            if ann_path is not None:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
                width = int(data["imageWidth"])
                height = int(data["imageHeight"])
                for shape in data.get("shapes") or []:
                    label = shape.get("label")
                    if label not in class_to_id:
                        stats[split]["skipped_shapes"] += 1
                        continue
                    values = polygon_to_yolo(shape.get("points") or [], width, height)
                    if len(values) < 6:
                        stats[split]["skipped_shapes"] += 1
                        continue
                    label_lines.append(f"{class_to_id[label]} " + " ".join(f"{v:.6f}" for v in values))

            shutil.copy2(image_path, out_root / "images" / split / image_path.name)
            label_path = out_root / "labels" / split / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            stats[split]["images"] += 1
            stats[split]["instances"] += len(label_lines)
            stats[split]["empty"] += int(not label_lines)

    yaml_path = out_root / "orange_wuxi_seg.yaml"
    yaml_path.write_text(
        "path: E:/mastercode/data/test\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {args.class_name}\n",
        encoding="utf-8",
    )

    report = {
        "source": str(src_root),
        "output": str(out_root),
        "seed": args.seed,
        "class_name": args.class_name,
        "batches": [ann_dir.name for ann_dir, _ in present_batches],
        "total_images": n_total,
        "negative_images": n_negatives,
        "split_counts": {split: len(split_items) for split, split_items in assignments.items()},
        "stats": stats,
        "yaml": str(yaml_path),
    }
    (out_root / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
