"""J6 paired-resolution data without changing the EV6 validation GT raster.

Training applies one 1280 augmentation and derives the 640 image from that view.
The internal training polygon raster ratio is four, yielding the external
640/2 supervision grid. Validation uses the untouched 640 YOLODataset sample;
only its additional high image is reconstructed directly from the source image.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imread


def aligned_high_view(raw, global_shape, ratio_pad, scale=2):
    """Rebuild a high view in the exact final global-canvas coordinate system.

    ``ratio_pad`` is the unchanged YOLO sample metadata: the first pair is the
    base loader's (height gain, width gain), the second is integer (left, top).
    BaseDataset uses ceil for the first resize, then LetterBox uses round for a
    possible second resize. Reproducing those two *sizes* avoids the one-pixel
    drift introduced by independently letterboxing at 1280. The source is resized
    directly once at high resolution; it is never interpolated from the 640 view.

    Returned affine matrices act on pixel-boundary coordinates (not pixel-centre
    indices). High and low image interpolation values need not be identical.
    """
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if raw is None or raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        raise ValueError("raw must be an HWC uint8 three-channel image")
    gh, gw = map(int, global_shape)
    if min(gh, gw) < 1:
        raise ValueError("global_shape must be positive")
    (gain_h, gain_w), (left, top) = ratio_pad
    values = np.asarray((gain_h, gain_w, left, top), dtype=float)
    if not np.isfinite(values).all() or min(gain_h, gain_w) <= 0:
        raise ValueError("ratio_pad must contain finite positive gains")
    if min(left, top) < 0 or int(left) != left or int(top) != top:
        raise ValueError("ratio_pad padding must contain non-negative integer left/top")
    left, top = int(left), int(top)
    h0, w0 = raw.shape[:2]
    # First resize was already rounded by BaseDataset; recover those exact sizes.
    h1, w1 = round(h0 * gain_h), round(w0 * gain_w)
    if min(h1, w1) < 1:
        raise ValueError("ratio_pad yields an empty pre-letterbox image")
    ratio = min(gh / h1, gw / w1, 1.0)  # standard validation LetterBox(scaleup=False)
    ch, cw = round(h1 * ratio), round(w1 * ratio)
    if min(ch, cw) < 1 or top + ch > gh or left + cw > gw:
        raise ValueError("ratio_pad/content does not fit the global canvas")
    if left != round((gw - cw) / 2 - 0.1) or top != round((gh - ch) / 2 - 0.1):
        raise ValueError("ratio_pad does not match centred standard validation letterboxing")
    high = np.full((gh * scale, gw * scale, 3), 114, dtype=np.uint8)
    resized = cv2.resize(raw, (cw * scale, ch * scale), interpolation=cv2.INTER_LINEAR)
    high[top * scale : (top + ch) * scale, left * scale : (left + cw) * scale] = resized
    affine = np.asarray([[cw / w0, 0.0, left], [0.0, ch / h0, top], [0.0, 0.0, 1.0]], dtype=np.float32)
    geometry = {
        "raw_to_global": affine,
        "raw_to_high": np.diag([scale, scale, 1.0]).astype(np.float32) @ affine,
        "global_content_xyxy": (left, top, left + cw, top + ch),
        "original_shape": (h0, w0),
        "global_shape": (gh, gw),
    }
    return high, geometry


class J6Dataset(YOLODataset):
    """Standard YOLO sample plus an aligned, genuinely higher-resolution image.

    The optional validation high-view cache is a bounded per-process uint8 LRU,
    not a cache of float tensors. Its cap is 128 MiB by default; DataLoader
    workers have separate caches, so multiply this bound by worker count when
    budgeting host RAM. The parent's requested cache policy is retained.
    """

    def __init__(self, *args, high_scale=2, high_cache_mb=128, **kwargs):
        self.j6_global_size = int(kwargs.get("imgsz", 640))
        self.j6_high_scale = int(high_scale)
        if high_scale != self.j6_high_scale or self.j6_high_scale != 2:
            raise ValueError("J6 currently fixes the high/global ratio to two")
        if self.j6_global_size != 640:
            raise ValueError("J6 fixes the global image size to 640")
        if not math.isfinite(float(high_cache_mb)) or high_cache_mb < 0:
            raise ValueError("high_cache_mb must be finite and non-negative")
        self.j6_high_cache_limit = int(high_cache_mb * 1024 * 1024)
        self._j6_high_cache = OrderedDict()
        self._j6_high_cache_bytes = 0
        augment = bool(kwargs.get("augment", False))
        hyp = deepcopy(kwargs.get("hyp"))
        if hyp is None or getattr(hyp, "task", "segment") != "segment" or kwargs.get("task") != "segment":
            raise ValueError("J6Dataset requires segmentation hyperparameters")
        if int(hyp.mask_ratio) != 2:
            raise ValueError("J6 external supervision must use mask_ratio=2")
        if augment and (kwargs.get("rect", False) or getattr(hyp, "multi_scale", 0)):
            raise ValueError("J6 training currently requires rect=False and multi_scale=0")
        kwargs["hyp"] = hyp
        kwargs["imgsz"] = self.j6_global_size * self.j6_high_scale if augment else self.j6_global_size
        super().__init__(*args, **kwargs)
        self.j6_protocol = {
            "global_imgsz": 640,
            "high_imgsz": 1280,
            "external_mask_ratio": 2,
            "internal_mask_ratio": 4 if augment else 2,
            "validation_gt": "unchanged standard 640 YOLODataset raster",
            "training_observation": "single high-resolution augmentation, AREA downsample to global",
            "high_cache_mb_per_worker": high_cache_mb if self.cache == "ram" and not augment else 0,
        }
        LOGGER.info(f"J6 paired views: {self.j6_protocol}")

    def build_transforms(self, hyp=None):
        """Keep close_mosaic rebuilds on the correct internal mask raster."""
        internal_hyp = deepcopy(hyp)
        internal_hyp.mask_ratio = 4 if self.augment else 2
        return super().build_transforms(internal_hyp)

    def _validation_high(self, index, sample):
        cached = self._j6_high_cache.pop(index, None)
        if cached is not None:
            self._j6_high_cache[index] = cached
            return cached
        path = self.im_files[index]
        raw = np.load(path, allow_pickle=False) if Path(path).suffix.lower() == ".npy" else imread(path)
        high_bgr, geometry = aligned_high_view(
            raw, sample["img"].shape[-2:], sample["ratio_pad"], self.j6_high_scale
        )
        high_rgb = np.ascontiguousarray(high_bgr[..., ::-1].transpose(2, 0, 1))
        payload = high_rgb, geometry
        size = high_rgb.nbytes
        if self.cache == "ram" and size <= self.j6_high_cache_limit:
            while self._j6_high_cache and self._j6_high_cache_bytes + size > self.j6_high_cache_limit:
                _, evicted = self._j6_high_cache.popitem(last=False)
                self._j6_high_cache_bytes -= evicted[0].nbytes
            self._j6_high_cache[index] = payload
            self._j6_high_cache_bytes += size
        return payload

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.augment:
            high = sample["img"]
            if tuple(high.shape[-2:]) != (1280, 1280):
                raise RuntimeError("J6 high training augmentation must return a 1280 square")
            image_hwc = high.numpy().transpose(1, 2, 0)
            low_hwc = cv2.resize(image_hwc, (640, 640), interpolation=cv2.INTER_AREA)
            sample["high_img"] = high
            sample["img"] = torch.from_numpy(np.ascontiguousarray(low_hwc.transpose(2, 0, 1)))
            sample["resized_shape"] = (640, 640)
            # RandomPerspective normally removes ratio_pad. If present it is
            # training metadata only; do not expose a misleading high-size pad.
            if "ratio_pad" in sample:
                sample.pop("ratio_pad")
        else:
            high, geometry = self._validation_high(index, sample)
            sample["high_img"] = torch.from_numpy(high)
            sample["j6_raw_to_global"] = torch.from_numpy(geometry["raw_to_global"])
            sample["j6_raw_to_high"] = torch.from_numpy(geometry["raw_to_high"])
        sample["j6_high_to_global"] = torch.diag(torch.tensor([0.5, 0.5, 1.0]))
        return sample

    @staticmethod
    def collate_fn(batch):
        """Preserve YOLO's instance indexing and stack only the added tensors."""
        if not batch:
            raise ValueError("Cannot collate an empty J6 batch")
        extra = {"high_img", "j6_high_to_global", "j6_raw_to_global", "j6_raw_to_high"}
        plain = [{key: value for key, value in sample.items() if key not in extra} for sample in batch]
        result = YOLODataset.collate_fn(plain)
        for key in extra & batch[0].keys():
            if not all(key in sample for sample in batch):
                raise ValueError(f"Inconsistent J6 sample metadata: {key}")
            result[key] = torch.stack([sample[key] for sample in batch])
        return result


def build_j6_dataset(cfg, img_path, batch, data, mode="train", stride=32, rect=None, high_cache_mb=128):
    """Construct a paired dataset with the ordinary YOLO builder's policies."""
    if mode not in {"train", "val", "test"}:
        raise ValueError("J6 mode must be train, val, or test")
    cfg = deepcopy(cfg)
    training = mode == "train"
    use_rect = bool(getattr(cfg, "rect", False) or (not training if rect is None else rect))
    return J6Dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=training,
        hyp=cfg,
        rect=use_rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if training else 0.5,
        prefix=f"J6 {mode}: ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if training else 1.0,
        high_cache_mb=high_cache_mb,
    )


def prepare_j6_batch(batch, device=None, half=False):
    """Move/normalise only high_img after the trainer's ordinary preprocessing.

    uint8 inputs become [0,1] floats. Already-normalised float tensors are not
    divided again, making repeated calls safe. Other keys are left untouched.
    """
    if "high_img" not in batch:
        raise KeyError("J6 batches require high_img")
    high = batch["high_img"]
    if not isinstance(high, torch.Tensor) or high.ndim != 4 or high.shape[1] != 3:
        raise ValueError("high_img must be a BCHW three-channel tensor")
    if high.dtype != torch.uint8 and not high.is_floating_point():
        raise ValueError("high_img must be uint8 or an already-normalised float tensor")
    was_uint8 = high.dtype == torch.uint8
    target = torch.device(device) if device is not None else high.device
    high = high.to(
        device=target, dtype=torch.float16 if half else torch.float32, non_blocking=target.type == "cuda"
    )
    batch["high_img"] = high / 255.0 if was_uint8 else high
    return batch
