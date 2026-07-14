"""Draw COCO polygons and boxes to verify converted instance labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from baseline_common import resolve_path
from torchvision_maskrcnn_common import prepared_split_paths, validate_prepared_dataset


COLORS = (
    (235, 64, 52, 90),
    (52, 152, 219, 90),
    (46, 204, 113, 90),
    (241, 196, 15, 90),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    dataset_root = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_prepared_dataset(dataset_root, (args.split,), ["orange_immature"])
    annotation_path, image_dir = prepared_split_paths(dataset_root, args.split)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    images = list(data["images"])
    random.Random(args.seed).shuffle(images)
    selected = images[: max(0, args.limit)]
    font = ImageFont.load_default()

    for image_record in selected:
        image = Image.open(image_dir / str(image_record["file_name"])).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        records = annotations_by_image.get(int(image_record["id"]), [])
        for index, annotation in enumerate(records):
            color = COLORS[index % len(COLORS)]
            for polygon in annotation["segmentation"]:
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.polygon(points, fill=color, outline=color[:3] + (255,), width=2)
            x, y, width, height = (float(value) for value in annotation["bbox"])
            draw.rectangle((x, y, x + width, y + height), outline=(255, 255, 255, 255), width=2)
        draw.rectangle((4, 4, 145, 24), fill=(0, 0, 0, 180))
        draw.text((8, 7), f"instances: {len(records)}", fill=(255, 255, 255, 255), font=font)
        Image.alpha_composite(image, overlay).convert("RGB").save(
            output_dir / str(image_record["file_name"])
        )

    print(f"Rendered {len(selected)} images to {output_dir}")


if __name__ == "__main__":
    main()
