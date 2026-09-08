"""Train-only tiny RGB proposal heatmap, then frozen content-guided crop placement.

Inspired by ClusDet/DMNet coarse proposal routing, not a reproduction of their
networks or their density estimator. No validation labels, detector oracle,
extra sparse CUDA package, colour threshold or GT-dependent inference crop.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GUIDE_SIZE = 384
GUIDE_EPOCHS = 20
GUIDE_SEED = 20260908
GUIDE_VERSION = "citrus_rgb_heatmap_v1"


def _guide_device(device):
    """Resolve a physical request against an optional single-GPU visibility mask."""
    value = str(device).lower().replace("cuda:", "").strip()
    if value == "cpu":
        return torch.device("cpu")
    if not value.isdigit():
        return torch.device(device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        physical = [item.strip() for item in visible.split(",") if item.strip()]
        if value in physical:
            return torch.device("cuda", physical.index(value))
        if len(physical) == 1:
            raise RuntimeError(f"Requested physical GPU {value} is not visible: CUDA_VISIBLE_DEVICES={visible}")
    return torch.device("cuda", int(value))


class TinyCropGuide(nn.Module):
    """Four ordinary convolutions plus a one-channel proposal logit map at stride8."""

    def __init__(self):
        super().__init__()
        layers = []
        for c1, c2, stride, dilation in ((3, 8, 2, 1), (8, 16, 2, 1), (16, 32, 2, 1), (32, 32, 1, 2)):
            layers.extend([nn.Conv2d(c1, c2, 3, stride, dilation, dilation=dilation, bias=False),
                           nn.BatchNorm2d(c2), nn.SiLU()])
        self.features = nn.Sequential(*layers)
        self.predictor = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        return self.predictor(self.features(image))


def guide_image(image):
    """Aspect-preserving BGR->RGB letterbox, identical during training and inference."""
    h, w = image.shape[:2]
    ratio = GUIDE_SIZE / max(h, w)
    rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
    top, left = (GUIDE_SIZE - rh) // 2, (GUIDE_SIZE - rw) // 2
    canvas = np.full((GUIDE_SIZE, GUIDE_SIZE, 3), 114, dtype=np.uint8)
    canvas[top:top + rh, left:left + rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)), (top, left, rh, rw)


def guide_target(rows, geometry):
    """Equal-width centre Gaussians: a proposal heatmap, NOT calibrated object counts."""
    top, left, rh, rw = geometry
    side = GUIDE_SIZE // 8
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    target = np.zeros((side, side), np.float32)
    for _, points in rows:
        centre = (points.min(0) + points.max(0)) / 2
        x, y = (centre * [rw, rh] + [left, top]) / 8
        target = np.maximum(target, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2)))
    return target[None]


def heatmap_loss(logits, target):
    """Balance foreground/background loss mass; guard empty images without dropping them."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return .5 * ((bce * target).sum() / target.sum().clamp_min(1)
                 + (bce * (1 - target)).sum() / (1 - target).sum().clamp_min(1))


def train_crop_guide(data, output, device="cpu", epochs=GUIDE_EPOCHS):
    """Fixed, separate preprocessing experiment; train split only, last epoch, no val selection."""
    from citrus_slicing import read_polygons, source_files
    from ultralytics.data.utils import img2label_paths

    if epochs < 1:
        raise ValueError("Guide requires positive epochs")
    _, files = source_files(data, "train")
    labels = [Path(p) for p in img2label_paths([str(p) for p in files])]
    sources = [dict(image=str(p), size=p.stat().st_size, mtime=p.stat().st_mtime_ns,
                    label_sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for p, q in zip(files, labels)]
    identity = dict(version=GUIDE_VERSION, size=GUIDE_SIZE, epochs=epochs, seed=GUIDE_SEED, sources=sources,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    if output.is_file():
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if saved.get("source_identity") != fingerprint:
            raise FileExistsError("Guide source/protocol changed: use a new E project; never silently reuse")
        return output
    if output.parent.exists():
        raise FileExistsError(f"Partial guide directory: {output.parent}; choose a new project")
    output.parent.mkdir(parents=True)
    cv2.setNumThreads(1)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.perf_counter()
    target_device = _guide_device(device)
    try:
        images, targets = [], []
        for i, (file, label) in enumerate(zip(files, labels)):
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError(f"Unreadable guide training RGB: {file}")
            pixels, geometry = guide_image(image)
            images.append(pixels)
            targets.append(guide_target(read_polygons(label), geometry))
            if i % 100 == 0:
                print(f"GUIDE train-only cache {i + 1}/{len(files)}", flush=True)
        images = torch.from_numpy(np.stack(images))
        targets = torch.from_numpy(np.stack(targets))
        cached_at = time.perf_counter()
        # Do not change the RNG state used by the later segmentation runs.
        devices = [target_device.index or 0] if target_device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(GUIDE_SEED)
            model = TinyCropGuide().to(target_device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
            generator = torch.Generator().manual_seed(GUIDE_SEED)
            losses = []
            for epoch in range(epochs):
                order = torch.randperm(len(images), generator=generator)
                loss_sum = 0.0
                for start in range(0, len(order), 16):
                    indices = order[start:start + 16]
                    x = images[indices].to(target_device).float().div_(255)
                    y = targets[indices].to(target_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = heatmap_loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite guide loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu()) * len(indices)
                losses.append(loss_sum / len(images))
                print(f"GUIDE train-only epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}", flush=True)
            report = dict(identity=identity, source_identity=fingerprint, train_images=len(files),
                          parameters=sum(p.numel() for p in model.parameters()), losses=losses,
                          cache_seconds=cached_at - started, training_seconds=time.perf_counter() - cached_at,
                          device=str(target_device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                          batch=16, optimizer="AdamW", lr=.001, weight_decay=.0005,
                          amp=False, selection="last fixed epoch; no validation/test labels or metrics")
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(dict(state_dict=state, source_identity=fingerprint, report=report), output)
            output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            del model, optimizer, images, targets
            if target_device.type == "cuda":
                import gc

                gc.collect()
                torch.cuda.empty_cache()
    finally:
        if target_device.type == "cuda":
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        torch.set_num_threads(previous_threads)
    return output


def windows_from_heatmap(shape, heatmap, fraction=.6):
    """Three heat-guided positions + one uncovered-area position; full view always first.

    Budget and size equal fixed slicing. All positions are derived from RGB
    predictions; no ground-truth argument. Flat/invalid maps fall back to fixed.
    """
    from citrus_slicing import view_windows

    h, w = shape
    fallback = view_windows(h, w, fraction)
    heatmap = np.asarray(heatmap, np.float32)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or np.ptp(heatmap) < 1e-5:
        return fallback
    work = np.maximum(heatmap - np.median(heatmap), 0)
    work /= max(float(work.max()), 1e-8)
    gh, gw = work.shape
    ch, cw = math.ceil(h * fraction), math.ceil(w * fraction)
    choices = list(dict.fromkeys((int(x), int(y), int(x) + cw, int(y) + ch)
                                 for y in np.linspace(0, h - ch, 5).round()
                                 for x in np.linspace(0, w - cw, 5).round()))
    if len(choices) < 4:
        return fallback
    uncovered = np.ones_like(work)
    selected = []
    for step in range(4):
        scores, boxes = [], []
        for x0, y0, x1, y1 in choices:
            left, right = round(x0 * gw / w), round(x1 * gw / w)
            top, bottom = round(y0 * gh / h), round(y1 * gh / h)
            boxes.append((left, top, right, bottom))
            if step == 3:  # coverage exploration, not another confident-peak gate
                scores.append(float(uncovered[top:bottom, left:right].sum()))
            else:
                mx, my = max(1, (right - left) // 10), max(1, (bottom - top) // 10)
                inner = work[top + my:bottom - my, left + mx:right - mx].sum()
                scores.append(float(inner + .25 * work[top:bottom, left:right].sum()))
        best = int(np.argmax(scores))
        selected.append(choices.pop(best))
        left, top, right, bottom = boxes[best]
        work[top:bottom, left:right] *= .15  # discourage duplicate coverage, not a hard discard
        uncovered[top:bottom, left:right] = 0
    return [(0, 0, w, h), *selected]


class CropGuide:
    """Frozen CPU guide for offline preparation and optional original-image inference."""

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint).resolve()
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if saved["report"]["identity"]["version"] != GUIDE_VERSION:
            raise ValueError("Unsupported guide version")
        self.model = TinyCropGuide().eval().requires_grad_(False)
        self.model.load_state_dict(saved["state_dict"])

    @torch.inference_mode()
    def heatmap(self, image):
        pixels, (top, left, rh, rw) = guide_image(image)
        logits = self.model(torch.from_numpy(pixels[None]).float() / 255)
        full = F.interpolate(logits.sigmoid(), (GUIDE_SIZE, GUIDE_SIZE), mode="bilinear", align_corners=False)
        valid = full[0, 0, top:top + rh, left:left + rw].numpy()
        return cv2.resize(valid, (96, 96), interpolation=cv2.INTER_AREA)

    def windows(self, image, fraction=.6):
        return windows_from_heatmap(image.shape[:2], self.heatmap(image), fraction)
