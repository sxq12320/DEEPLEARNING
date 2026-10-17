"""J V3 data path: inject teacher-mined instances into the unchanged multi-scale training views.

Teacher instances live in ORIGINAL-image normalized coordinates (``citrus_j3_teacher``). Each prepared view gets the
part of every instance inside its window, clipped exactly like ground truth, then encoded as a label-trust ``cls``
code (``ultralytics.utils.citrus_j3_loss``). The prepared PNG/label cache, val/test data and shared slicing files are
never modified; injection happens in memory after the label cache is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from citrus_e_v5_slicing import MultiScaleViewDataset
from citrus_e_v6_training import EV6TrainingTrainer
from ultralytics.data.augment import Format
from ultralytics.data.utils import polygon2mask
from ultralytics.utils.citrus_j3_loss import IGNORE_BASE, J3SegmentationLoss, decode_codes, ignore_code, pseudo_code

PSEUDO_MODES = ("none", "ignore", "hard", "soft", "naive")
MIN_VIEW_AREA = 2.0  # px^2 at the view's training resolution; smaller clipped slivers carry no supervision
SOFT_FLOOR = 0.05


def instance_code(tier, score, mode):
    """Label-trust code of one mined instance under an arm's policy, or None when the arm does not use it.

    none: nothing injected. ignore: every mined instance only stops background punishment. hard: corroborated
    high tier = full fruit, rest ignored. soft: high tier weighted by teacher confidence, rest ignored. naive: every
    mined instance as a full fruit (plain self-training control, no ignore band).
    """
    if mode not in PSEUDO_MODES:
        raise ValueError(f"pseudo mode must be one of {PSEUDO_MODES}")
    if tier not in ("high", "mid"):
        raise ValueError("teacher tier must be 'high' or 'mid'")
    if mode == "none":
        return None
    if mode == "naive":
        return pseudo_code(1.0)
    if mode == "ignore" or tier == "mid":
        return ignore_code(score)
    return pseudo_code(1.0 if mode == "hard" else min(max(float(score), SOFT_FLOOR), 1.0))


def clip_pseudo(points, shape, window):
    """Clip one normalized source polygon into a view.

    Returns ("polygon", normalized view vertices), ("box", normalized rectangle) when the visible part is
    disconnected or holed (only usable as an ignore region), or None when nothing is visible.
    """
    from shapely.geometry import Polygon, box

    h, w = shape
    x0, y0, x1, y1 = window
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None
    if tuple(window) == (0, 0, w, h):
        return "polygon", points.clip(0, 1)
    polygon = Polygon(points * [w, h])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    part = polygon.intersection(box(x0, y0, x1, y1))
    if part.is_empty or part.area == 0:
        return None
    if part.geom_type == "Polygon" and not len(part.interiors):
        vertices = np.asarray(part.exterior.coords)[:-1]
        return "polygon", ((vertices - [x0, y0]) / [x1 - x0, y1 - y0]).clip(0, 1)
    bx0, by0, bx1, by1 = part.bounds
    rect = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float64)
    return "box", ((rect - [x0, y0]) / [x1 - x0, y1 - y0]).clip(0, 1)


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def inject_pseudo(labels, im_files, root, groups, pseudo, mode, imgsz=640):
    """Append encoded teacher instances to each view's in-memory label dict. Returns an injection summary.

    Args:
        labels (list[dict]): Ultralytics label dicts (cls (n,1), bboxes (n,4) xywh normalized, segments list).
        im_files (list[str]): Image path of each label dict.
        root (Path): Prepared-view root holding ``views.json``.
        groups (list[dict]): ``views.json`` groups (source, original_shape, views with window/image).
        pseudo (dict): ``pseudo.json`` content from ``citrus_j3_teacher.mine``.
        mode (str): One of PSEUDO_MODES.
        imgsz (int): Training resolution of a view (long side), for the sliver filter.
    """
    summary = dict(mode=mode, sources_with_instances=0, views_touched=0, positive=0, ignore=0,
                   demoted_disconnected=0, skipped_sliver=0, by_scale={"global": [0, 0], "local": [0, 0]})
    if mode == "none":
        return summary
    lookup = {str(Path(p).resolve()): i for i, p in enumerate(im_files)}
    images = pseudo["images"]
    for group in groups:
        entry = images.get(str(Path(group["source"]).resolve()))
        if not entry or not entry["instances"]:
            continue
        summary["sources_with_instances"] += 1
        shape = tuple(group["original_shape"])
        for view in group["views"]:
            index = lookup.get(str((Path(root) / view["image"]).resolve()))
            if index is None:
                continue
            x0, y0, x1, y1 = view["window"]
            ratio = imgsz / max(x1 - x0, y1 - y0)
            rows, segments = [], []
            for instance in entry["instances"]:
                code = instance_code(instance["tier"], instance["score"], mode)
                if code is None:
                    continue
                clipped = clip_pseudo(instance["poly"], shape, view["window"])
                if clipped is None:
                    continue
                kind, vertices = clipped
                pixels = polygon_area(vertices * [(x1 - x0) * ratio, (y1 - y0) * ratio])
                if pixels < MIN_VIEW_AREA:
                    summary["skipped_sliver"] += 1
                    continue
                if kind == "box" and code < IGNORE_BASE:
                    code = ignore_code(instance["score"])
                    summary["demoted_disconnected"] += 1
                rows.append(code)
                segments.append(vertices.astype(np.float32))
            if not rows:
                continue
            label = labels[index]
            boxes = np.array([[(s[:, 0].min() + s[:, 0].max()) / 2, (s[:, 1].min() + s[:, 1].max()) / 2,
                               s[:, 0].max() - s[:, 0].min(), s[:, 1].max() - s[:, 1].min()] for s in segments],
                             dtype=np.float32)
            label["cls"] = np.concatenate([label["cls"].reshape(-1, 1), np.asarray(rows, np.float32)[:, None]], 0)
            label["bboxes"] = np.concatenate([label["bboxes"].reshape(-1, 4), boxes], 0)
            label["segments"] = list(label.get("segments") or []) + segments
            kinds, _ = decode_codes(np.asarray(rows))
            positives, ignores = int((kinds == 1).sum()), int((kinds == 2).sum())
            summary["views_touched"] += 1
            summary["positive"] += positives
            summary["ignore"] += ignores
            scale = summary["by_scale"]["global" if view.get("global_view") else "local"]
            scale[0] += positives
            scale[1] += ignores
    return summary


class J3Format(Format):
    """Overlap masks paint ignore regions FIRST so every real or pseudo fruit keeps its pixels where they overlap.

    Positives keep the stock rule among themselves (larger first, smaller painted on top). Only used when an arm
    injects instances; the J50 anchor keeps the stock ``Format`` bit-for-bit.
    """

    def _format_segments(self, instances, cls, w, h):
        if not self.mask_overlap:
            return super()._format_segments(instances, cls, w, h)
        segments = instances.segments
        masks = [polygon2mask((h, w), [s.reshape(-1)], color=1, downsample_ratio=self.mask_ratio) for s in segments]
        areas = np.asarray([m.sum() for m in masks])
        positive = decode_codes(cls.reshape(-1))[0] < 2
        order = np.lexsort((-areas, positive))  # ignore (False) first, then by decreasing area
        canvas = np.zeros((h // self.mask_ratio, w // self.mask_ratio),
                          dtype=np.int32 if len(segments) > 255 else np.uint8)
        for rank, k in enumerate(order):
            canvas[masks[k] > 0] = rank + 1
        return canvas[None], instances[order], cls[order]


def _swap_format(transforms):
    for i, t in enumerate(getattr(transforms, "transforms", [])):
        if type(t) is Format:
            t.__class__ = J3Format
            return True
    return False


class J3PseudoViewDataset(MultiScaleViewDataset):
    """E V5 multi-scale views (p=.5/.25/.25) + in-memory teacher instances; epoch length = source count."""

    def __init__(self, *args, pseudo=None, pseudo_mode="none", **kwargs):
        if pseudo_mode not in PSEUDO_MODES:
            raise ValueError(f"pseudo_mode must be one of {PSEUDO_MODES}")
        if (pseudo is None) != (pseudo_mode == "none"):
            raise ValueError("A pseudo-label file is required exactly when pseudo_mode != 'none'")
        self.pseudo_mode = pseudo_mode
        super().__init__(*args, **kwargs)
        root = Path(self.data["path"])
        groups = json.loads((root / "views.json").read_text(encoding="utf-8"))["groups"]
        if pseudo is not None:
            training = {str(Path(g["source"]).resolve()) for g in groups}
            foreign = set(pseudo["images"]) - training
            if foreign:
                raise ValueError(f"Pseudo labels contain {len(foreign)} non-training sources, e.g. {sorted(foreign)[0]}")
        self.pseudo_summary = inject_pseudo(self.labels, self.im_files, root, groups, pseudo or {"images": {}},
                                            pseudo_mode, self.imgsz)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.pseudo_mode != "none" and not _swap_format(transforms):
            raise RuntimeError("J3 could not locate the Format transform")
        return transforms


def load_pseudo(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def make_trainer(factors, pseudo_path=None):
    """EV6 trainer (I-series validator, input/2 mask raster) + J3 views + trust-aware loss. Graph stays YOLO11n-seg."""
    mode = factors["pseudo"]
    if (pseudo_path is None) != (mode == "none"):
        raise ValueError("pseudo_path is required exactly for arms that inject teacher instances")

    class J3Trainer(EV6TrainingTrainer):
        j3_factors = dict(factors)
        j3_pseudo_path = str(pseudo_path) if pseudo_path else None

        def build_dataset(self, img_path, mode_name="train", batch=None):
            if mode_name != "train":
                return super().build_dataset(img_path, mode_name, batch)
            dataset = J3PseudoViewDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=self.args,
                rect=False,
                cache=self.args.cache,
                single_cls=self.args.single_cls,
                stride=32,
                pad=0.0,
                prefix="J3 views: ",
                task="segment",
                classes=self.args.classes,
                data=self.data,
                fraction=1.0,
                tile_probability=0.5,
                pseudo=load_pseudo(self.j3_pseudo_path),
                pseudo_mode=mode,
            )
            print("J3 TEACHER INJECTION:", json.dumps(dataset.pseudo_summary), flush=True)
            return dataset

        def _setup_train(self):
            super()._setup_train()
            from ultralytics.utils.torch_utils import unwrap_model

            # Installed AFTER the EMA copy: checkpoints stay plain YOLO11n-seg (stock-loadable, no J3 classes).
            model = unwrap_model(self.model)
            model.criterion = J3SegmentationLoss(model)

        def plot_training_labels(self):
            from ultralytics.utils.plotting import plot_labels

            labels = self.train_loader.dataset.labels
            boxes = np.concatenate([lb["bboxes"][lb["cls"].reshape(-1) < 0.5] for lb in labels], 0)
            plot_labels(boxes, np.zeros(len(boxes)), names=self.data["names"], save_dir=self.save_dir,
                        on_plot=self.on_plot)

        def plot_training_samples(self, batch, ni):
            from ultralytics.utils.plotting import plot_images

            shown = dict(batch)
            shown["cls"] = decode_codes(batch["cls"].view(-1))[0].float().view(-1, 1)  # 0 GT, 1 pseudo, 2 ignore
            plot_images(labels=shown, paths=batch["im_file"], fname=self.save_dir / f"train_batch{ni}.jpg",
                        on_plot=self.on_plot)

    J3Trainer.__name__ = f"J3Trainer_{mode}"
    return J3Trainer
