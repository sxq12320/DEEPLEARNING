"""Derive COCO/Roboflow/YOLO views of ONE data.yaml without changing its splits or labels."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image

from common import save_json

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def label_for(image):
    parts = list(image.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image path must have an images directory for YOLO label mapping: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def split_files(value, base):
    result = []
    for item in value if isinstance(value, list) else [value]:
        path = Path(item).expanduser()
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        if path.is_dir():
            result.extend(sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    image = Path(line.strip()).expanduser()
                    result.append(image.resolve() if image.is_absolute() else (path.parent / image).resolve())
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:
            result.append(path)
        else:
            raise FileNotFoundError(path)
    if not result or len(set(result)) != len(result):
        raise ValueError("Empty split or duplicate image paths inside split")
    if any(not image.is_file() for image in result):
        raise FileNotFoundError("An image listed in the split does not exist")
    return result


def inspect_source(data):
    data = Path(data).expanduser().resolve()
    content = data.read_text(encoding="utf-8-sig")
    cfg = yaml.safe_load(content)
    base = Path(cfg.get("path", ".")).expanduser()
    base = base.resolve() if base.is_absolute() else (data.parent / base).resolve()
    names = cfg["names"]
    if isinstance(names, dict):
        if sorted(int(i) for i in names) != list(range(len(names))):
            raise ValueError("Class IDs must be contiguous from zero")
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    splits = {split: split_files(cfg[split], base) for split in ("train", "val", "test") if cfg.get(split)}
    if "train" not in splits or "val" not in splits:
        raise ValueError("data.yaml requires train and val")
    seen = set()
    digest = hashlib.sha256(content.encode())
    for split, files in splits.items():
        if seen.intersection(files):
            raise ValueError("The provided data.yaml has overlapping image paths across splits")
        seen.update(files)
        for image in files:
            label = label_for(image)
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}; source data was NOT changed")
            stat = image.stat()
            digest.update(f"{split}|{image}|{stat.st_size}|{stat.st_mtime_ns}".encode())
            digest.update(label.read_bytes())
    return data, names, splits, digest.hexdigest()


def materialize(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # Only called after matching source metadata; restart of an interrupted preparation is safe.
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Incomplete/conflicting derived file: {destination}. Use a new project directory.")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare(data, output):
    from pycocotools import mask as mask_utils

    data, names, splits, signature = inspect_source(data)
    output = Path(output).resolve()
    marker = output / "source.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Dataset changed since preparation. Use a NEW PROJECT; no source files were modified.")
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise FileExistsError(f"Unrecognized prepared directory: {output}")
    save_json(marker, dict(data=str(data), signature=signature, note="provenance only; no confirmation prompt"))
    categories = [dict(id=i + 1, name=name, supercategory="citrus") for i, name in enumerate(names)]
    summary = {"source": str(data), "signature": signature, "names": names, "splits": {}}
    manifest = []
    for split, images in splits.items():
        coco = dict(
            info={"description": "Unmodified source split; polygon coordinates in original pixels"},
            licenses=[],
            images=[],
            annotations=[],
            categories=categories,
        )
        for image_id, image in enumerate(images, 1):
            label = label_for(image)
            file_name = f"{image_id:07d}_{image.name}"
            with Image.open(image) as source_image:
                width, height = source_image.size
            coco["images"].append(dict(id=image_id, file_name=file_name, width=width, height=height))
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                values = [float(v) for v in line.split()]
                if (
                    len(values) < 7
                    or len(values) % 2 != 1
                    or not all(math.isfinite(v) for v in values)
                    or not values[0].is_integer()
                    or not 0 <= values[0] < len(names)
                    or any(not 0 <= v <= 1 for v in values[1:])
                ):
                    raise ValueError(f"Invalid polygon at {label}:{line_number}; NOT repaired or deleted")
                points = [value * (width if i % 2 == 0 else height) for i, value in enumerate(values[1:])]
                xs, ys = points[::2], points[1::2]
                rle = mask_utils.merge(mask_utils.frPyObjects([points], height, width))
                area = float(mask_utils.area(rle))
                if area <= 0 or max(xs) <= min(xs) or max(ys) <= min(ys):
                    raise ValueError(f"Zero-area rasterized instance at {label}:{line_number}; NOT silently dropped")
                coco["annotations"].append(
                    dict(
                        id=len(coco["annotations"]) + 1,
                        image_id=image_id,
                        category_id=int(values[0]) + 1,
                        segmentation=[points],
                        area=area,
                        iscrowd=0,
                        bbox=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    )
                )
            rf_split = "valid" if split == "val" else split
            for target in (
                output / "coco/images" / split / file_name,
                output / "yolo" / split / "images" / file_name,
                output / "rfdetr" / rf_split / file_name,
            ):
                materialize(image, target)
            materialize(label, output / "yolo" / split / "labels" / Path(file_name).with_suffix(".txt"))
            manifest.append(
                dict(split=split, image_id=image_id, source=str(image), label=str(label), derived=file_name)
            )
        save_json(output / "coco/annotations" / f"instances_{split}.json", coco)
        # Keep all derived COCO views 1-based; post0 reserves classifier output 0.
        rf_coco = copy.deepcopy(coco)
        # RF-DETR 1.4.0.post0 reserves output 0; foreground uses canonical COCO IDs 1..N.
        save_json(output / "rfdetr" / rf_split / "_annotations.coco.json", rf_coco)
        summary["splits"][split] = dict(images=len(coco["images"]), instances=len(coco["annotations"]))
    yolo = dict(path=str(output / "yolo"), names=names)
    yolo.update({split: f"{split}/images" for split in splits})
    (output / "yolo/data.yaml").write_text(yaml.safe_dump(yolo, allow_unicode=True), encoding="utf-8")
    save_json(output / "manifest.json", manifest)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
