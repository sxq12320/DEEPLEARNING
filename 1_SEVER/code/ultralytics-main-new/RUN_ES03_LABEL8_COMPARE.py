"""ES03: train once per label policy, then evaluate each weight under both policies."""
from collections import Counter
from decimal import Decimal
from pathlib import Path

from RUN_LABEL8_COMPARE import ROOT, main as compare

OLD_DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
NEW_DATA = "/data/sxq/datasets/orange_yolo_grouped_min8px640_20261107/data.yaml"
PROJECT = "/data/sxq/results/ES03_LABEL8_20261108"
DEVICE = "1"
SEEDS = [42]
EPOCHS = 300
EVALUATE_TEST = False
SMOKE = False


def polygons(file):
    return [tuple(map(Decimal, line.split())) for line in file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def audit_labels(specs):
    """Confirm identical membership and the actual min8 rule before starting either fit."""
    from PIL import Image

    report = {}
    for split in (key for key in ("train", "val", "test") if key in specs["all"]):
        images = Path(specs["all"][split])
        old_labels = images.parent / "labels"
        new_labels = Path(specs["min8"][split]).parent / "labels"
        counts = dict(all=0, min8=0)
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            name = image.stem + ".txt"
            original = polygons(old_labels / name)
            filtered = polygons(new_labels / name)
            with Image.open(image) as im:
                w, h = im.size
                if im.getexif().get(274, 1) in (5, 6, 7, 8):
                    w, h = h, w
            expected = []
            for row in original:
                xs, ys = row[1::2], row[2::2]
                width = (max(xs) - min(xs)) * w
                height = (max(ys) - min(ys)) * h
                if min(width, height) * 640 >= 8 * max(w, h):
                    expected.append(row)
            if Counter(expected) != Counter(filtered):
                raise ValueError(f"{split}/{name}: expected {len(expected)} polygons after min8 filtering, "
                                 f"found {len(filtered)} or different polygons. Check OLD_DATA and NEW_DATA.")
            counts["all"] += len(original)
            counts["min8"] += len(filtered)
        counts["removed"] = counts["all"] - counts["min8"]
        report[split] = counts
        print(f"{split} targets | all={counts['all']} min8={counts['min8']} removed={counts['removed']}", flush=True)
    if report["train"]["removed"] == 0:
        raise ValueError("No training labels differ. Check the uploaded dataset pair.")
    return report


def main():
    return compare(model_yaml=ROOT / "0_orange_yaml/E12_SIMPLE/ES03_detail_semantic.yaml",
                   project_dir=PROJECT, old_data=OLD_DATA, new_data=NEW_DATA, device=DEVICE,
                   seeds=SEEDS, epochs=EPOCHS, evaluate_test=EVALUATE_TEST, smoke=SMOKE,
                   check_labels=audit_labels)


if __name__ == "__main__":
    main()
