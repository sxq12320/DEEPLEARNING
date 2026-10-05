"""J V2 Glance-Gaze evaluation on frozen weights: one forward cache, many tile-selection policies.

Glance: whole image. Gaze: fine tiles (fraction .4, >=20% overlap, the I-series "fine" grid) cropped from the
ORIGINAL image. Policies differ only in WHICH tiles are merged, so every number below comes from the same
predictions. Metrics use the I-series common 640 raster, mask-aware merge (trusted border weight .5).

Policies
  global                    glance only (1 view)
  all9                      every tile (= historical fine slicing)
  self_greedy_t             greedy coverage of glance candidates with raster area < 32^2, weight = score
  density_greedy_t          greedy coverage of density peaks (p >= .05), weight = p      [density heads only]
  density_greedy_t_gate_b   + gaze-only small candidates scored s * g^b, g = max density in the candidate box
  oracle_small              tiles containing a GT fruit < 32^2 (CEILING; uses GT, never a method)
  random_match              random tiles, same per-image count as the main adaptive policy (control)
Greedy coverage maximises a monotone submodular weighted coverage F(S); for any fixed K it attains
>= (1 - 1/e) OPT_K (Nemhauser et al., 1978). The threshold t stops when the marginal gain < t.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from citrus_e_v5_slicing import uniform_windows
from citrus_slicing import source_files
from eval_citrus_e_v5 import add_view_metadata, merge_views
from eval_citrus_sliced import ground_truth, mask_overlaps, merge_candidates, raster_shape
from scripts.review_sage_v4r import greedy_pairs
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics, box_iou
from ultralytics.utils.ops import scale_masks

SMALL = 32 * 32
TINY = 16 * 16
PEAK_FLOOR = 0.05


def _sync(device):
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_view(model, image, window, eval_shape, device, imgsz=640, antialias=False, conf=0.001, iou=0.7, max_det=300):
    """Same contract as eval_citrus_sliced.predict_view; antialias pre-shrinks the crop with INTER_AREA."""
    x0, y0, x1, y1 = window
    h, w = image.shape[:2]
    eh, ew = eval_shape
    gx0, gx1 = round(x0 * ew / w), round(x1 * ew / w)
    gy0, gy1 = round(y0 * eh / h), round(y1 * eh / h)
    crop = image[y0:y1, x0:x1]
    factor = 1.0
    if antialias:
        r = imgsz / max(crop.shape[:2])
        if r < 1:
            size = (min(math.ceil(crop.shape[1] * r), imgsz), min(math.ceil(crop.shape[0] * r), imgsz))
            factor = crop.shape[1] / size[0]
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    result = model.predict(crop, imgsz=imgsz, device=device, conf=conf, iou=iou, max_det=max_det, verbose=False,
                           save=False, retina_masks=False)[0]
    if result.masks is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy() * factor + [x0, y0, x0, y0]
    masks = scale_masks(result.masks.data[None].float(), (gy1 - gy0, gx1 - gx0))[0].gt(0.5).cpu().numpy()
    candidates = []
    for box, score, cls, mask in zip(boxes, result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), masks):
        yy, xx = np.nonzero(mask)
        if len(xx):
            left, top, right, bottom = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
            patch, offset = mask[top:bottom, left:right].copy(), (gx0 + int(left), gy0 + int(top))
        else:
            patch, offset = np.zeros((0, 0), dtype=bool), (0, 0)
        candidates.append(dict(box=box, score=score, cls=int(cls), patch=patch, offset=offset))
    return candidates


def letterbox_geometry(h, w, density_shape, stride, imgsz):
    """Map original pixels to the predictor's auto-letterboxed input that produced ``density_shape``."""
    gh, gw = density_shape
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = gw * stride - new_w, gh * stride - new_h
    return r, round(dw / 2 - 0.1), round(dh / 2 - 0.1)


def density_peaks(prob, h, w, stride, imgsz, floor=PEAK_FLOOR):
    """3x3 local maxima >= floor, returned as original-image (x, y) points and probabilities."""
    if prob is None:
        return np.zeros((0, 2)), np.zeros(0)
    t = torch.as_tensor(prob)[None, None].float()
    keep = (t == torch.nn.functional.max_pool2d(t, 3, 1, 1)) & (t >= floor)
    ys, xs = np.nonzero(keep[0, 0].numpy())
    r, left, top = letterbox_geometry(h, w, prob.shape, stride, imgsz)
    px = ((xs + 0.5) * stride - left) / r
    py = ((ys + 0.5) * stride - top) / r
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return np.stack([px, py], 1)[inside], prob[ys, xs][inside]


def box_density(prob, box, h, w, stride, imgsz):
    """Max density probability over grid cells covered by an original-coordinate box (>= 1 cell)."""
    r, left, top = letterbox_geometry(h, w, prob.shape, stride, imgsz)
    gh, gw = prob.shape
    x0 = int(np.clip(math.floor((box[0] * r + left) / stride), 0, gw - 1))
    y0 = int(np.clip(math.floor((box[1] * r + top) / stride), 0, gh - 1))
    x1 = int(np.clip(math.ceil((box[2] * r + left) / stride), x0 + 1, gw))
    y1 = int(np.clip(math.ceil((box[3] * r + top) / stride), y0 + 1, gh))
    return float(prob[y0:y1, x0:x1].max())


def greedy_windows(points, weights, windows, threshold, max_windows=None):
    """Greedy weighted max-coverage; stop when the best marginal gain < threshold (or zero)."""
    if not len(points) or not len(windows):
        return []
    px, py = points[:, 0], points[:, 1]
    inside = np.stack([(px >= x0) & (px < x1) & (py >= y0) & (py < y1) for x0, y0, x1, y1 in windows])
    covered, chosen = np.zeros(len(points), bool), []
    while max_windows is None or len(chosen) < max_windows:
        gains = (inside & ~covered) @ weights
        k = int(np.argmax(gains))
        if gains[k] <= 0 or gains[k] < threshold:
            break
        chosen.append(k)
        covered |= inside[k]
    return chosen


def fit_s50(sizes, hits):
    """Logistic recall model r(s) = sigmoid((ln s - ln s50) / tau), maximum-likelihood grid fit."""
    if len(sizes) < 10 or hits.all() or not hits.any():
        return None, None
    best, lx = None, np.log(np.maximum(sizes, 1))
    for s50 in np.linspace(3, 40, 149):
        for tau in np.linspace(0.05, 1.5, 59):
            p = np.clip(1 / (1 + np.exp(-(lx - math.log(s50)) / tau)), 1e-6, 1 - 1e-6)
            ll = float((hits * np.log(p) + (1 - hits) * np.log(1 - p)).sum())
            if best is None or ll > best[0]:
                best = (ll, s50, tau)
    return float(best[1]), float(best[2])


class DensityHook:
    def __init__(self, model):
        head = model.model.model[-1]
        self.stride = int(head.stride[0]) if hasattr(head, "stride") else 8
        self.module = getattr(head, "density", None) if getattr(head, "density_enabled", False) else None
        self.value = None
        if self.module is not None:
            self.handle = self.module.register_forward_hook(self._store)

    def _store(self, module, inputs, output):
        self.value = torch.sigmoid(output[0, 0].detach().float()).cpu().numpy()

    def pop(self):
        value, self.value = self.value, None
        return value


def evaluate(weights, data, output, device="cpu", imgsz=640, antialias=False, limit=0, fraction=0.4,
             thresholds=(0.25, 0.5, 1.0), betas=(0.5, 1.0), seed=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config, files = source_files(data, "val")
    files = files[:limit] if limit else files
    model = YOLO(str(weights))
    hook = DensityHook(model)
    has_density = hook.module is not None
    model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)
    hook.pop()
    matcher = SegmentationValidator(args=dict(plots=False))
    main = "density_greedy_0.5" if has_density else "self_greedy_0.5"
    policies = ["global", "all9", "oracle_small"] + [f"self_greedy_{t}" for t in thresholds]
    if has_density:
        policies += [f"density_greedy_{t}" for t in thresholds]
        policies += [f"density_greedy_0.5_gate_{b}" for b in betas]
    policies.append("random_match")
    metrics = {p: SegmentMetrics(config["names"]) for p in policies}
    tally = {p: dict(tiles=0, tp=0, background=0, tiny_n=0, tiny_hit=0, small_n=0, small_hit=0) for p in policies}
    timing = {p: [] for p in policies}
    sizes = {p: [] for p in policies}
    hits = {p: [] for p in policies}
    rng = random.Random(seed)
    for index, file in enumerate(files):
        image = cv2.imread(str(file))
        if image is None:
            raise ValueError(f"Unreadable validation image: {file}")
        h, w = image.shape[:2]
        shape = raster_shape(h, w, 640)
        windows = uniform_windows(h, w, fraction)
        tiles = windows[1:]
        _sync(device)
        start = time.perf_counter()
        glance = add_view_metadata(predict_view(model, image, windows[0], shape, device, imgsz, antialias),
                                   windows[0], (h, w), 0)
        _sync(device)
        glance_s = time.perf_counter() - start
        prob = hook.pop()
        views, tile_s = [glance], []
        for v, window in enumerate(tiles, 1):
            _sync(device)
            start = time.perf_counter()
            views.append(add_view_metadata(predict_view(model, image, window, shape, device, imgsz, antialias),
                                           window, (h, w), v))
            _sync(device)
            tile_s.append(time.perf_counter() - start)
            hook.pop()
        gt = ground_truth(file, (h, w), shape)
        gt_area = gt["masks"].sum((1, 2)) if len(gt["masks"]) else np.zeros(0)
        gt_box = gt["boxes"].numpy()
        small_glance = [c for c in glance if c["patch"].size and c["patch"].sum() < SMALL]
        self_pts = np.array([[0.5 * (c["box"][0] + c["box"][2]), 0.5 * (c["box"][1] + c["box"][3])] for c in small_glance])
        self_w = np.array([c["score"] for c in small_glance])
        gt_small = np.array([[0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])] for a, b in zip(gt_area, gt_box) if a < SMALL])
        oracle = [k for k, (x0, y0, x1, y1) in enumerate(tiles)
                  if len(gt_small) and ((gt_small[:, 0] >= x0) & (gt_small[:, 0] < x1)
                                        & (gt_small[:, 1] >= y0) & (gt_small[:, 1] < y1)).any()]
        choice = {"global": [], "all9": list(range(len(tiles))), "oracle_small": oracle}
        for t in thresholds:
            choice[f"self_greedy_{t}"] = greedy_windows(self_pts, self_w, tiles, t) if len(self_pts) else []
        if has_density:
            points, weights_ = density_peaks(prob, h, w, hook.stride, imgsz)
            for t in thresholds:
                choice[f"density_greedy_{t}"] = greedy_windows(points, weights_, tiles, t)
            for b in betas:
                choice[f"density_greedy_0.5_gate_{b}"] = choice["density_greedy_0.5"]
        choice["random_match"] = rng.sample(range(len(tiles)), len(choice[main]))
        for policy in policies:
            selected = choice[policy]
            candidates = list(glance)
            beta = float(policy.rsplit("_", 1)[1]) if "_gate_" in policy else 0.0
            for k in selected:
                for c in views[k + 1]:
                    if beta and c["patch"].size and c["patch"].sum() < SMALL:
                        g = max(box_density(prob, c["box"], h, w, hook.stride, imgsz), 1e-3)
                        c = {**c, "score": c["score"] * g**beta}
                    candidates.append(c)
            start = time.perf_counter()
            pred = merge_candidates(candidates, shape) if policy == "global" else merge_views(candidates, shape,
                                                                                               border_weight=0.5)
            timing[policy].append(glance_s + sum(tile_s[k] for k in selected) + time.perf_counter() - start)
            overlap = mask_overlaps(gt["masks"], pred["masks"])
            boxes = box_iou(gt["boxes"], pred["bboxes"])
            metrics[policy].update_stats(dict(
                tp=matcher.match_predictions(pred["cls"], gt["cls"], boxes).numpy(),
                tp_m=matcher.match_predictions(pred["cls"], gt["cls"], overlap).numpy(),
                conf=pred["conf"].numpy(), pred_cls=pred["cls"].numpy(), target_cls=gt["cls"].numpy(),
                target_img=np.unique(gt["cls"].numpy()), im_name=file.name))
            keep = pred["conf"] >= 0.25
            ov, bo = overlap[:, keep].numpy(), boxes[:, keep].numpy()
            pairs = greedy_pairs(ov, 0.5)
            matched_gt, matched_pred = set(pairs[:, 0].tolist()), set(pairs[:, 1].tolist())
            t_ = tally[policy]
            t_["tiles"] += len(selected)
            t_["tp"] += len(matched_pred)
            t_["background"] += sum(1 for j in range(ov.shape[1]) if j not in matched_pred
                                    and ov[:, j].max(initial=0) < 0.5 and bo[:, j].max(initial=0) < 0.5)
            for j, area in enumerate(gt_area):
                hit = j in matched_gt
                sizes[policy].append(math.sqrt(max(float(area), 1.0)))
                hits[policy].append(hit)
                if area < TINY:
                    t_["tiny_n"] += 1
                    t_["tiny_hit"] += hit
                elif area < SMALL:
                    t_["small_n"] += 1
                    t_["small_hit"] += hit
        if index % 20 == 0 or index == len(files) - 1:
            print(f"J2 gaze evaluation {index + 1}/{len(files)}", flush=True)
    summary = {}
    for policy in policies:
        metrics[policy].process(plot=False)
        res = metrics[policy].results_dict
        s50, tau = fit_s50(np.asarray(sizes[policy]), np.asarray(hits[policy], float))
        t_ = tally[policy]
        summary[policy] = dict(
            views_per_image=1 + t_["tiles"] / max(len(files), 1),
            mask_ap50_95=res["metrics/mAP50-95(M)"], mask_ap50=res["metrics/mAP50(M)"],
            box_ap50_95=res["metrics/mAP50-95(B)"],
            recall25=float(np.mean(hits[policy])) if hits[policy] else 0.0,
            precision25=t_["tp"] / max(t_["tp"] + t_["background"], 1) if t_["tp"] else 0.0,
            background25=t_["background"], tiny=f"{t_['tiny_hit']}/{t_['tiny_n']}", small=f"{t_['small_hit']}/{t_['small_n']}",
            s50_px=s50, tau=tau, median_ms=float(np.median(timing[policy]) * 1000), mean_ms=float(np.mean(timing[policy]) * 1000),
        )
    payload = dict(
        weights=str(weights), data=str(data), images=len(files), main_policy=main, summary=summary,
        protocol=dict(imgsz=imgsz, antialias=antialias, mask_raster=640, conf=0.001, nms_iou=0.7, max_det=300,
                      fraction=fraction, merge="mask-aware, trusted border weight .5", thresholds=list(thresholds),
                      betas=list(betas), density_peak_floor=PEAK_FLOOR, small_area=SMALL, random_seed=seed,
                      device=str(device)),
        limits=("Frozen weights; all policies reuse one prediction cache. oracle_small uses GT and is a ceiling only. "
                "Timing = glance + selected tiles (sequential, batch 1) + merge; tiles could be batched. "
                "Common-raster AP is not the training-validator AP. precision25/background25 use IoU .5 at conf .25."),
    )
    (output / "gaze_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{'policy':28s} views  AP50-95  AP50   recall  tiny     bgFP   s50   ms")
    for policy, s in summary.items():
        s50 = f"{s['s50_px']:.1f}" if s["s50_px"] else "-"
        print(f"{policy:28s} {s['views_per_image']:5.2f}  {s['mask_ap50_95'] * 100:6.2f}  {s['mask_ap50'] * 100:5.2f}"
              f"  {s['recall25']:.3f}  {s['tiny']:>7s}  {s['background25']:5d}  {s50:>4s}  {s['median_ms']:.0f}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ("weights", "data", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--antialias", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    evaluate(**vars(parser.parse_args()))
