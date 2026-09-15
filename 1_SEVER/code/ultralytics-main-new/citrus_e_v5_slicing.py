"""Uniform scale-stratified sampling, never RGB/GT-guided crop selection.

Derived crops retain the existing strict polygon clipping contract. Original
data and val/test are untouched; no size filtering. A source still contributes
one logical training draw, regardless of how many views were cached.
"""

import hashlib
import json
import math
import random
from pathlib import Path

from citrus_slicing import SourceBalancedViewDataset, SlicedTrainingTrainer, prepare_views, view_windows
from ultralytics.data.dataset import YOLODataset


def uniform_windows(h, w, fraction=0.4):
    """Global plus overlapping grid covering every original pixel (9 tiles at .4)."""
    if not 0.25 <= fraction < 1 or min(h, w) < 1:
        raise ValueError("Uniform fraction must be in [0.25, 1), image dimensions positive")
    th, tw = max(1, round(h * fraction)), max(1, round(w * fraction))
    # At least 20% overlap for normal-sized images, including final border tiles.
    ny = math.ceil((h - th) / max(1, 0.8 * th)) + 1
    nx = math.ceil((w - tw) / max(1, 0.8 * tw)) + 1
    ys = [round(i * (h - th) / max(ny - 1, 1)) for i in range(ny)]
    xs = [round(i * (w - tw) / max(nx - 1, 1)) for i in range(nx)]
    return list(dict.fromkeys([(0, 0, w, h)] + [(x, y, x + tw, y + th) for y in ys for x in xs]))


class UniformMultiScale:
    """Uses the shared selector INTERFACE, not a learned guide or image content."""

    sha256 = hashlib.sha256(b"E_V5_uniform_v1:global+corners.6+grid.4:overlap.2").hexdigest()

    def windows(self, image, fraction=0.6):
        h, w = image.shape[:2]
        return list(dict.fromkeys(view_windows(h, w, fraction) + uniform_windows(h, w, 0.4)[1:]))


def prepare_multiscale_views(data, output, imgsz=640):
    result = prepare_views(data, output, imgsz, 0.6, UniformMultiScale())
    saved = json.loads((Path(output) / "views.json").read_text(encoding="utf-8"))
    upper_gib = saved["prepared_views"] * imgsz * imgsz * 3 / 1024**3
    print(
        f"UNIFORM MULTISCALE: {saved['prepared_views']} views; RGB RAM upper estimate {upper_gib:.2f} GiB. "
        "Epoch draw count remains original source count. No learned guide.",
        flush=True,
    )
    return result


class MultiScaleViewDataset(SourceBalancedViewDataset):
    """p(global)=.5, p(coarse .6)=.25, p(fine .4)=.25, uniformly within scale.

    Equal scale probability avoids oversampling the nine-view stratum merely
    because it has more cached windows. Invalid/disconnected clips fall back to
    a representable stratum; source-level fallback counts are recorded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(self.data["path"])
        groups = json.loads((root / "views.json").read_text(encoding="utf-8"))["groups"]
        lookup = {str(Path(p).resolve()): i for i, p in enumerate(self.im_files)}
        self.scale_groups = []
        for group in groups:
            h, w = group["original_shape"]
            pools = [[], [], []]
            for view in group["views"]:
                key = str((root / view["image"]).resolve())
                if key not in lookup:
                    continue
                x0, y0, x1, y1 = view["window"]
                scale = max((x1 - x0) / w, (y1 - y0) / h)
                bucket = 0 if view["global_view"] else 1 if scale > 0.5 else 2
                pools[bucket].append(lookup[key])
            if len(pools[0]) != 1:
                raise ValueError("Each source requires exactly one original full view")
            self.scale_groups.append(pools)
        self.scale_summary = dict(
            probability=[0.5, 0.25, 0.25],
            fractions=[1.0, 0.6, 0.4],
            no_coarse_sources=sum(not g[1] for g in self.scale_groups),
            no_fine_sources=sum(not g[2] for g in self.scale_groups),
            windows_per_scale=[sum(len(g[i]) for g in self.scale_groups) for i in range(3)],
        )

    def get_image_and_label(self, index):
        pools = self.scale_groups[index]
        draw = random.random()
        bucket = 0 if draw < 0.5 else 1 if draw < 0.75 else 2
        candidates = pools[bucket] or pools[1] or pools[0]
        return YOLODataset.get_image_and_label(self, random.choice(candidates))


class MultiScaleTrainingTrainer(SlicedTrainingTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        return MultiScaleViewDataset(
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
            prefix="E V5 uniform: ",
            task="segment",
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            tile_probability=0.5,
        )
