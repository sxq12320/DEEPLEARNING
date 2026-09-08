"""Slicing-aided RGB training views with source-balanced sampling.

Inspired by SAHI (ICIP 2022), independently integrated with this YOLO fork.
Original files/splits are never changed. A disconnected clipped instance is
NOT split into multiple fruit labels: reject that entire tile and keep its
source's global view. This avoids unlabelled fragments and polygon bridges.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics.data.augment import Mosaic
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.models.yolo.segment import SegmentationTrainer


def source_files(data, split="train"):
    """Resolve the user's dataset using Ultralytics, excluding paired .npy caches."""
    config = check_det_dataset(str(data), autodownload=False)
    entries = config[split] if isinstance(config[split], list) else [config[split]]
    files = []
    for entry in entries:
        p = Path(entry)
        if p.is_dir():
            files.extend(x.resolve() for x in p.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        elif p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    q = Path(line.strip())
                    files.append((q if q.is_absolute() else p.parent / q).resolve())
        else:
            raise ValueError(f"Expected an image directory or manifest: {p}")
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No source images in {split}")
    return config, files


def view_windows(h, w, fraction=0.6):
    """Full image + four corner-aligned overlapping views; no GT in window choice."""
    if not 0.5 < fraction < 1.0:
        raise ValueError("Use .5 < fraction < 1 for four overlapping, covering tiles")
    ch, cw = max(1, math.ceil(h * fraction)), max(1, math.ceil(w * fraction))
    full = (0, 0, w, h)
    return list(dict.fromkeys([full] + [(x, y, x + cw, y + ch) for y in (0, h - ch) for x in (0, w - cw)]))


def read_polygons(file):
    if not Path(file).is_file():
        raise FileNotFoundError(f"Missing source label; not silently cleaned: {file}")
    rows = []
    for number, line in enumerate(Path(file).read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        values = np.array([float(x) for x in line.split()], dtype=np.float64)
        if len(values) < 7 or len(values) % 2 != 1 or not np.isfinite(values).all():
            raise ValueError(f"Expected YOLO polygon at {file}:{number + 1}; original not changed")
        rows.append((int(values[0]), values[1:].reshape(-1, 2)))
    return rows


def clip_instances(rows, size, window):
    """Return one polygon per visible instance or reject an unrepresentable tile.

    No min-size/min-area-ratio filtering. Only empty intersections disappear
    from a view. Global views retain original labels, including difficult ones.
    """
    try:
        from shapely.geometry import Polygon, box
    except ImportError as error:
        raise RuntimeError("Install slicing dependency: python -m pip install -r requirements-citrus-e.txt") from error
    h, w = size
    x0, y0, x1, y1 = window
    if tuple(window) == (0, 0, w, h):
        return rows, None
    boundary = box(x0, y0, x1, y1)
    clipped = []
    for cls, points in rows:
        polygon = Polygon(points * [w, h])
        if not polygon.is_valid:
            return [], "invalid_source_polygon_global_retained"
        intersection = polygon.intersection(boundary)
        if intersection.is_empty or intersection.area == 0:
            continue
        if intersection.geom_type != "Polygon" or len(intersection.interiors):
            return [], "disconnected_or_holed_intersection_global_retained"
        vertices = np.asarray(intersection.exterior.coords)[:-1]
        vertices = (vertices - [x0, y0]) / [x1 - x0, y1 - y0]
        clipped.append((cls, vertices.clip(0, 1)))
    return clipped, None


def _prepare_views(data, output, imgsz=640, fraction=0.6, window_selector=None):
    """One-time derived PNG/label cache; resize ONLY AFTER cropping original pixels."""
    try:
        import shapely  # noqa: F401 -- fail before creating a partial derived directory
    except ImportError as error:
        raise RuntimeError("Install: python -m pip install -r requirements-citrus-e.txt") from error
    cv2.setNumThreads(1)
    output = Path(output).resolve()
    config, files = source_files(data)
    _, val = source_files(data, "val")
    if set(files) & set(val):
        raise ValueError("Source train/val paths overlap; slicing must not leak views across splits")
    labels = [Path(x) for x in img2label_paths([str(p) for p in files])]
    signature = dict(
        version=1, data_yaml=Path(data).read_text(encoding="utf-8"), imgsz=imgsz, fraction=fraction,
        sources=[dict(image=str(p), label=str(q), image_size=p.stat().st_size,
                      image_mtime=p.stat().st_mtime_ns, label_text=q.read_text(encoding="utf-8"))
                 for p, q in zip(files, labels)],
    )
    if window_selector is not None:
        signature["guide_sha256"] = window_selector.sha256
    manifest = output / "views.json"
    if output.exists():
        if not manifest.is_file():
            raise FileExistsError(f"Partial prepared cache: {output}. Use a NEW PREPARED directory; do not overwrite")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise FileExistsError("Prepared source/config differs; choose a new prepared directory")
        for group in saved["groups"]:
            for view in group["views"]:
                if not (output / view["image"]).is_file() or not (output / view["label"]).is_file():
                    raise FileNotFoundError("Prepared view missing; do not train on an incomplete cache")
        return output / "data.yaml"
    output.mkdir(parents=True)
    (output / "train/images").mkdir(parents=True)
    (output / "train/labels").mkdir(parents=True)
    groups, skipped = [], []
    for i, (file, label) in enumerate(zip(files, labels)):
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError(f"Cannot read {file}")
        h, w = image.shape[:2]
        rows = read_polygons(label)
        views = []
        windows = (view_windows(h, w, fraction) if window_selector is None
                   else window_selector.windows(image, fraction))
        for j, window in enumerate(windows):
            targets, reason = clip_instances(rows, (h, w), window)
            if reason:
                skipped.append(dict(source=str(file), view=j, reason=reason))
                continue
            x0, y0, x1, y1 = window
            crop = image[y0:y1, x0:x1]
            ratio = imgsz / max(crop.shape[:2])
            crop = cv2.resize(crop, (min(imgsz, math.ceil(crop.shape[1] * ratio)),
                                     min(imgsz, math.ceil(crop.shape[0] * ratio))), interpolation=cv2.INTER_LINEAR)
            name = f"source{i:06d}_view{j}"
            im_path, lb_path = Path(f"train/images/{name}.png"), Path(f"train/labels/{name}.txt")
            if not cv2.imwrite(str(output / im_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Unable to save {im_path}")
            text = "\n".join(str(cls) + " " + " ".join(f"{v:.9f}" for v in p.reshape(-1)) for cls, p in targets)
            if j == 0:
                text = label.read_text(encoding="utf-8")  # full-view labels retain their exact source text
            else:
                text = text + ("\n" if text else "")
            (output / lb_path).write_text(text, encoding="utf-8")
            views.append(dict(image=im_path.as_posix(), label=lb_path.as_posix(), window=window,
                              global_view=j == 0, instances=len(targets)))
        groups.append(dict(source=str(file), original_shape=[h, w], instances=len(rows), views=views))
        if i % 25 == 0:
            print(f"PREPARE original RGB -> views: {i + 1}/{len(files)}", flush=True)
    derived = dict(path=str(output), train="train/images", val=config["val"], names=config["names"])
    if config.get("test"):
        derived["test"] = config["test"]
    (output / "data.yaml").write_text(yaml.safe_dump(derived, allow_unicode=True), encoding="utf-8")
    saved = dict(signature=signature, groups=groups, skipped_tiles=skipped,
                 training_sources=len(groups), prepared_views=sum(len(g["views"]) for g in groups),
                 note="Train only; original val/test untouched. Global view always retained. No minimum-size filtering.")
    manifest.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PREPARED {len(groups)} sources, {saved['prepared_views']} views, {len(skipped)} unrepresentable tiles skipped")
    return output / "data.yaml"


def prepare_views(data, output, imgsz=640, fraction=0.6, window_selector=None):
    """Original recipe is unchanged; optional guide is frozen and reads RGB only."""
    if window_selector is None:
        return _prepare_views(data, output, imgsz, fraction)
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        return _prepare_views(data, output, imgsz, fraction, window_selector)
    finally:
        torch.set_num_threads(previous)


class SourceBalancedViewDataset(YOLODataset):
    """Epoch length stays the number of original images, NOT five times larger.

    Each source draw chooses global with p=.5, otherwise one valid local view.
    Standard YOLO augmentation/close_mosaic remains unchanged across E arms.
    """

    def __init__(self, *args, tile_probability=0.5, **kwargs):
        self.tile_probability = tile_probability
        super().__init__(*args, **kwargs)
        if self.rect:
            raise ValueError("Source-balanced training requires rect=False")
        root = Path(self.data["path"])
        manifest = json.loads((root / "views.json").read_text(encoding="utf-8"))
        indices = {str(Path(p).resolve()): i for i, p in enumerate(self.im_files)}
        self.source_groups = [[indices[str((root / v["image"]).resolve())] for v in g["views"]
                               if str((root / v["image"]).resolve()) in indices]
                              for g in manifest["groups"]]
        self.source_files = [g["source"] for g in manifest["groups"]]
        self.source_instances = sum(g["instances"] for g in manifest["groups"])
        assert all(len(g) >= 1 for g in self.source_groups)

    def get_img_files(self, img_path):
        files = super().get_img_files(img_path)
        # Global controls do not need four unused tile images in RAM per source.
        return [p for p in files if Path(p).stem.endswith("_view0")] if self.tile_probability == 0 else files

    def __len__(self):
        return len(self.source_groups) if hasattr(self, "source_groups") else super().__len__()

    def get_image_and_label(self, index):
        group = self.source_groups[index]
        chosen = random.choice(group[1:]) if len(group) > 1 and random.random() < self.tile_probability else group[0]
        return super().get_image_and_label(chosen)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        # Mosaic's RAM path samples logical source indices. Never sample a raw
        # cached-view buffer index (also applies if RAM allocation is refused).
        def visit(t):
            if isinstance(t, Mosaic):
                t.buffer_enabled = False
            for child in getattr(t, "transforms", []):
                visit(child)
            pre = getattr(t, "pre_transform", None)
            if pre is not None:
                visit(pre)
        visit(transforms)
        return transforms


class SlicedTrainingTrainer(SegmentationTrainer):
    tile_probability = 0.5

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        return SourceBalancedViewDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
            rect=False, cache=self.args.cache, single_cls=self.args.single_cls, stride=32,
            pad=0.0, prefix="E train: ", task="segment", classes=self.args.classes, data=self.data,
            fraction=1.0, tile_probability=self.tile_probability,
        )


class GlobalTrainingTrainer(SlicedTrainingTrainer):
    tile_probability = 0.0


class ContextTrainingTrainer(SlicedTrainingTrainer):
    """E06 retains global context in 75% of source draws, with all other augmentation unchanged."""

    tile_probability = 0.25
