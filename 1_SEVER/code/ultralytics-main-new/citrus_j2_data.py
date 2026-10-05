"""J V2 data path: resampling choice, gaze-biased fine views, anti-aliased validation.

Shared slicing files are NOT modified (their bytes are part of earlier series' provenance digests).
Resampling is applied by a scoped wrapper around the unchanged view preparation.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

import citrus_slicing
from citrus_e_v5_slicing import MultiScaleViewDataset, UniformMultiScale
from citrus_e_v6_training import EV6TrainingTrainer
from ultralytics.data import build as data_build
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.patches import imread

GLANCE_IMGSZ = 640  # "small" is defined on the 640 glance raster for every arm
SMALL_AREA = 32 * 32
RESAMPLE = ("linear", "area")


class _AreaDownscale:
    """cv2 proxy: INTER_AREA whenever the target is smaller than the source; otherwise unchanged."""

    def __getattr__(self, name):
        return getattr(cv2, name)

    @staticmethod
    def resize(src, dsize, *args, **kwargs):
        if dsize[0] < src.shape[1] or dsize[1] < src.shape[0]:
            kwargs["interpolation"] = cv2.INTER_AREA
            args = ()
        return cv2.resize(src, dsize, *args, **kwargs)


def prepare_resampled_views(data, output, imgsz=640, resample="linear"):
    """Global + 4 coarse (.6) + 9 fine (.4) views, identical to E V5/J V1 except optional INTER_AREA shrink."""
    if resample not in RESAMPLE:
        raise ValueError(f"resample must be one of {RESAMPLE}")
    output = Path(output)
    marker = output / "j2_resample.json"
    record = dict(resample=resample, imgsz=int(imgsz), windows=UniformMultiScale.sha256)
    if output.exists() and (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != record):
        raise FileExistsError(f"Prepared views at {output} were not made with {record}; use a new directory")
    context = (
        contextlib.nullcontext()
        if resample == "linear"
        else _patched(citrus_slicing, "cv2", _AreaDownscale())
    )
    with context:
        result = citrus_slicing.prepare_views(data, output, imgsz, 0.6, UniformMultiScale())
    if not marker.is_file():
        marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    saved = json.loads((output / "views.json").read_text(encoding="utf-8"))
    gib = saved["prepared_views"] * imgsz * imgsz * 3 / 1024**3
    print(f"J2 VIEWS {resample} @{imgsz}: {saved['prepared_views']} views; RGB RAM upper estimate {gib:.1f} GiB", flush=True)
    return result


@contextlib.contextmanager
def _patched(owner, name, value):
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def small_count(label_file, window, original_shape, glance=GLANCE_IMGSZ):
    """Instances in a view whose visible area on the GLOBAL glance raster is < 32x32 px."""
    h, w = original_shape
    x0, y0, x1, y1 = window
    r = glance / max(h, w)
    scale = (x1 - x0) * (y1 - y0) * r * r
    count = 0
    for line in Path(label_file).read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 7:
            area = polygon_area(np.asarray(values[1:], float).reshape(-1, 2)) * scale
            count += 0 < area < SMALL_AREA
    return count


class GazeViewDataset(MultiScaleViewDataset):
    """Same .5/.25/.25 global/coarse/fine strata. Within FINE: half uniform, half proportional to small fruits.

    Uniform half keeps leaf-only negatives; the weighted half matches test-time gaze windows.
    """

    gaze_fraction = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(self.data["path"])
        groups = json.loads((root / "views.json").read_text(encoding="utf-8"))["groups"]
        lookup = {str(Path(p).resolve()): i for i, p in enumerate(self.im_files)}
        self.view_small = np.zeros(len(self.im_files), dtype=np.float64)
        for group in groups:
            for view in group["views"]:
                key = str((root / view["image"]).resolve())
                if key in lookup:
                    self.view_small[lookup[key]] = small_count(root / view["label"], view["window"], group["original_shape"])
        fine = [i for pools in self.scale_groups for i in pools[2]]
        self.gaze_summary = dict(
            gaze_fraction=self.gaze_fraction,
            fine_views=len(fine),
            fine_views_with_small=int((self.view_small[fine] > 0).sum()) if fine else 0,
            small_in_fine_views=int(self.view_small[fine].sum()) if fine else 0,
        )

    def get_image_and_label(self, index):
        pools = self.scale_groups[index]
        draw = random.random()
        bucket = 0 if draw < 0.5 else 1 if draw < 0.75 else 2
        candidates = pools[bucket] or pools[1] or pools[0]
        if bucket == 2 and pools[2] and random.random() < self.gaze_fraction:
            weights = self.view_small[candidates]
            if weights.sum() > 0:
                return YOLODataset.get_image_and_label(self, random.choices(candidates, weights=weights.tolist())[0])
        return YOLODataset.get_image_and_label(self, random.choice(candidates))


class AntialiasYOLODataset(YOLODataset):
    """Validation images shrunk with INTER_AREA (box prefilter) instead of INTER_LINEAR point sampling."""

    def load_image(self, i, rect_mode=True, resize_short=False):
        f, fn = str(self.im_files[i]), self.npy_files[i]
        if self.augment or self.ims[i] is not None or fn.exists() or f.lower().endswith(".npy") or resize_short:
            return super().load_image(i, rect_mode, resize_short)
        im = imread(f, flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        h0, w0 = im.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                size = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, size, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            interp = cv2.INTER_AREA if self.imgsz < max(h0, w0) else cv2.INTER_LINEAR
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=interp)
        if im.ndim == 2:
            im = im[..., None]
        return im, (h0, w0), im.shape[:2]


def make_trainer(factors):
    """EV6 trainer (I-series validator, input/2 mask raster) with J V2 data factors."""
    gaze_train, antialias = bool(factors["gaze_train"]), bool(factors["antialias"])

    class J2Trainer(EV6TrainingTrainer):
        j2_factors = dict(factors)

        def build_dataset(self, img_path, mode="train", batch=None):
            if mode == "train":
                if not gaze_train:
                    return super().build_dataset(img_path, mode, batch)
                dataset = GazeViewDataset(
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
                    prefix="J2 gaze views: ",
                    task="segment",
                    classes=self.args.classes,
                    data=self.data,
                    fraction=1.0,
                    tile_probability=0.5,
                )
                print("J2 GAZE TRAINING:", dataset.gaze_summary, flush=True)
                return dataset
            if not antialias:
                return super().build_dataset(img_path, mode, batch)
            with _patched(data_build, "YOLODataset", AntialiasYOLODataset):
                return super().build_dataset(img_path, mode, batch)

    J2Trainer.__name__ = "J2Trainer_gaze{}_aa{}".format(int(gaze_train), int(antialias))
    return J2Trainer
