"""Filter citrus mask instances by bounding-box short side after whole-image letterboxing.

The default filters train only. Pass --clean-splits train,val,test to define a
size-limited task on every split. The source and its split assignments are preserved.
"""
import argparse
import csv
import json
import shutil
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from PIL import Image
import yaml

SPLITS = ("train", "val", "test")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parent / "orange_yolo_grouped_dedup_20260820")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-side", type=Decimal, default=Decimal(8))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--clean-splits", default="train")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scaled_bbox(line, width, height, imgsz):
    fields = line.split()
    if len(fields) < 7 or len(fields) % 2 == 0:
        return None
    try:
        coordinates = [Decimal(value) for value in fields[1:]]
    except InvalidOperation:
        return None
    if not all(value.is_finite() and 0 <= value <= 1 for value in coordinates):
        return None
    xs, ys = coordinates[::2], coordinates[1::2]
    bw = (max(xs) - min(xs)) * width * imgsz / max(width, height)
    bh = (max(ys) - min(ys)) * height * imgsz / max(width, height)
    return (bw, bh) if min(bw, bh) > 0 else None


def main():
    args = parse_args()
    source = args.source.resolve()
    selected = {key.strip() for key in args.clean_splits.split(",")}
    if not selected <= set(SPLITS) or not args.min_side.is_finite() or args.min_side <= 0 or args.imgsz <= 0:
        raise ValueError("Invalid splits, threshold or image size")
    stamp = datetime.now().strftime("%Y%m%d")
    out = (args.out or source.with_name(f"{source.name}_min{args.min_side}px_{stamp}")).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Use a separate output directory")
    stage = out.with_name(out.name + ".building")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
    for split in SPLITS:
        if not (source / split / "images").is_dir() or not (source / split / "labels").is_dir():
            raise ValueError(f"Missing {split} images/labels directory in {source}")
    if not args.dry_run:
        if out.exists():
            raise FileExistsError(out)
        stage.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(source), imgsz=args.imgsz, threshold=float(args.min_side),
                  clean_splits=sorted(selected), split_reports={})
    removed = []
    for split in SPLITS:
        folder = source / split
        images = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in SUFFIXES)
        if not images:
            raise ValueError(f"Empty split {split}")
        stats = dict(images=len(images), before=0, removed=0, after=0, changed_images=0, newly_empty=0,
                     missing_labels=0, unmeasurable_preserved=0)
        if not args.dry_run:
            (stage / split / "images").mkdir(parents=True)
            (stage / split / "labels").mkdir()
        for image in images:
            label = folder / "labels" / (image.stem + ".txt")
            if not args.dry_run:
                shutil.copy2(image, stage / split / "images" / image.name)
            if not label.is_file():
                stats["missing_labels"] += 1
                continue
            with Image.open(image) as im:
                w, h = im.size
                if im.getexif().get(274, 1) in (5, 6, 7, 8):
                    w, h = h, w
            lines = label.read_bytes().splitlines(keepends=True)
            keep = []
            count = 0
            deleted = 0
            for num, raw in enumerate(lines, 1):
                line = raw.decode("utf-8-sig").strip()
                if line:
                    count += 1
                    box = scaled_bbox(line, w, h, args.imgsz)
                    if box is None:
                        stats["unmeasurable_preserved"] += 1
                    elif split in selected and min(box) < args.min_side:
                        deleted += 1
                        removed.append((split, image.name, num, w, h, float(box[0]), float(box[1]), line))
                        continue
                keep.append(raw)
            stats["before"] += count
            stats["after"] += count - deleted
            stats["removed"] += deleted
            stats["changed_images"] += int(deleted > 0)
            stats["newly_empty"] += int(deleted > 0 and count == deleted)
            if not args.dry_run:
                (stage / split / "labels" / label.name).write_bytes(b"".join(keep))
        report["split_reports"][split] = stats
    if not args.dry_run:
        config = {key: config[key] for key in ("names", "nc") if key in config}
        config.update({split: f"{split}/images" for split in SPLITS})
        (stage / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (stage / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (stage / "removed_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image", "line", "original_width", "original_height",
                             "width_at_input", "height_at_input", "original_polygon"])
            writer.writerows(removed)
        stage.rename(out)
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        print(out / "data.yaml")


if __name__ == "__main__":
    main()
