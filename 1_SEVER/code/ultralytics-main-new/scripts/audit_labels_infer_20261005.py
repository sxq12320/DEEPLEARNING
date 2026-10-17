"""Full-dataset label audit by model disagreement (read-only, no training).

Runs existing segmentation weights (default: two diverse I-series checkpoints)
over EVERY source image of the formal dataset, in global and fine views, and
reconciles predictions against the original X-AnyLabeling JSON polygons:

- ``candidates``: detections that match no annotation -> candidate missing labels.
- per-GT fields: whether anything was proposed there at all (conf floor 0.001),
  whether a merged prediction matched at 0.25, and the best mask IoU -> a label
  the model can never see is a suspicious label (or an invisible fruit).

Candidates are clustered across models/views so each unique location appears
once with a vote list. Writes one JSON per image under ``--output/infer``;
a separate script renders the HTML. Nothing on disk is modified.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from citrus_e_v5_slicing import uniform_windows
from eval_citrus_sliced import raster_shape
from diagnose_citrus_20261005 import infer_view, traced_merge
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


def gt_raster_masks(points_list, shape, wh):
    """Rasterize original-image polygons onto the common raster canvas."""
    w, h = wh
    masks = []
    for pts in points_list:
        canvas = np.zeros(shape, np.uint8)
        scaled = np.asarray(pts, dtype=np.float64)
        scaled[:, 0] *= shape[1] / w
        scaled[:, 1] *= shape[0] / h
        if len(scaled) >= 3:
            cv2.fillPoly(canvas, [np.round(scaled).astype(np.int32)], 1)
        masks.append(canvas.astype(bool))
    return masks


def det_polygon(mask, shape, wh):
    """Simplified outer contour of a raster mask, in original-image coords."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
    w, h = wh
    return [[float(x) * w / shape[1], float(y) * h / shape[0]] for x, y in approx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True,
                        help="comma separated name=path pairs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wuxi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)

    shard_k, shard_n = args.shard
    out = args.output / "infer"
    out.mkdir(parents=True, exist_ok=True)

    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    images = {p.name: p for d in ("img", "img_2") for p in (args.wuxi_root / d).iterdir() if p.is_file()}
    jsons = {p.stem: p for d in ("annotions_x", "annotion_x_2") for p in (args.wuxi_root / d).glob("*.json")}

    models = {}
    for spec in args.weights.split(","):
        name, path = spec.split("=", 1)
        models[name] = YOLO(path).model.float().eval()
        for p in models[name].parameters():
            p.requires_grad_(False)

    elapsed = time.perf_counter()
    todo = [(i, r) for i, r in enumerate(manifest) if i % shard_n == shard_k]
    if args.limit:
        todo = todo[:args.limit]
    for done, (i, r) in enumerate(todo):
        destination = out / f"audit_{i:04d}.json"
        if destination.exists():
            continue
        source_image = images.get(r["source_name"])
        source_json = jsons.get(Path(r["source_name"]).stem)
        if source_image is None:
            continue
        image = cv2.imread(str(source_image))
        if image is None:
            continue
        h, w = image.shape[:2]
        shape = raster_shape(h, w, 640)
        windows = uniform_windows(h, w, 0.4)

        shapes, points_list = [], []
        if source_json is not None:
            shapes = json.loads(source_json.read_text(encoding="utf-8")).get("shapes", [])
            points_list = [s["points"] for s in shapes if s.get("shape_type") == "polygon" and len(s["points"]) >= 3]
        gt_masks = gt_raster_masks(points_list, shape, (w, h))
        gt_boxes = torch.tensor([[min(p[0] for p in pts), min(p[1] for p in pts),
                                  max(p[0] for p in pts), max(p[1] for p in pts)] for pts in points_list],
                                dtype=torch.float32) if points_list else torch.zeros((0, 4))

        detections, raw_boxes_all, raw_scores_all = [], [], []
        for mname, model in models.items():
            global_view = infer_view(model, image, windows[0], (h, w), shape, 0)
            views = [global_view]
            for view, win in enumerate(windows[1:], 1):
                x0, y0, x1, y1 = win
                views.append(infer_view(model, image[y0:y1, x0:x1], win, (h, w), shape, view))
            for view in views:
                raw_boxes_all.append(view["boxes"])
                raw_scores_all.append(view["scores"])
            for mode, single, sel in (("global", True, [global_view]), ("fine", False, views)):
                final, masks, _removed = traced_merge(sum((v["candidates"] for v in sel), []), shape, single)
                for k, c in enumerate(final):
                    if c["score"] >= 0.25:
                        detections.append(dict(model=mname, mode=mode, box=np.asarray(c["box"], float),
                                               score=float(c["score"]), mask=masks[k]))

        if len(detections):
            det_masks = np.stack([d["mask"] for d in detections])
        else:
            det_masks = np.zeros((0, *shape), bool)
        raw_boxes = (torch.tensor(np.concatenate(raw_boxes_all), dtype=torch.float32)
                     if raw_boxes_all else torch.zeros((0, 4)))
        raw_scores = np.concatenate(raw_scores_all) if raw_scores_all else np.zeros(0)

        # GT side
        gts = []
        if len(gt_masks):
            gt_stack = np.stack(gt_masks)
            iou_gd = np.zeros((len(gt_masks), len(detections)))
            if len(detections):
                inter = (gt_stack[:, None] & det_masks[None]).sum((2, 3))
                union = gt_stack.sum((1, 2))[:, None] + det_masks.sum((1, 2))[None] - inter
                iou_gd = inter / np.maximum(union, 1)
            geom = box_iou(gt_boxes, raw_boxes).numpy() if len(raw_boxes) else np.zeros((len(gt_masks), 0))
            for g in range(len(gt_masks)):
                best = float(iou_gd[g].max()) if len(detections) else 0.0
                matched = [f"{d['model']}:{d['mode']}" for d, v in zip(detections, iou_gd[g] >= 0.5) if v]
                eligible = geom[g] >= 0.5
                gts.append(dict(index=g, box=[float(x) for x in gt_boxes[g]],
                                area_raster=int(gt_stack[g].sum()),
                                matched25=matched, best_iou=best,
                                proposed_001=bool((raw_scores[eligible] > 0.001).any()) if eligible.any() else False,
                                proposed_025=bool((raw_scores[eligible] > 0.25).any()) if eligible.any() else False))

        # Candidate missing labels: merged detections matching nothing
        cand = []
        if len(detections):
            if len(gt_masks):
                inter = (np.stack(gt_masks)[:, None] & det_masks[None]).sum((2, 3))
                union = np.stack(gt_masks).sum((1, 2))[:, None] + det_masks.sum((1, 2))[None] - inter
                iou_dg = inter / np.maximum(union, 1)
                best_per_det = iou_dg.max(0)
            else:
                best_per_det = np.zeros(len(detections))
            order = sorted(range(len(detections)),
                           key=lambda k: -detections[k]["score"])
            clusters = []  # each: {members:[det idx], box: rep box}
            for k in order:
                if best_per_det[k] >= 0.5:
                    continue
                b = torch.tensor(detections[k]["box"][None])
                hit = None
                for cl in clusters:
                    if float(box_iou(b, torch.tensor(np.asarray([cl["box"]])))[0, 0]) >= 0.3:
                        hit = cl
                        break
                if hit is None:
                    clusters.append({"box": detections[k]["box"], "members": [k]})
                else:
                    hit["members"].append(k)
            for cl in clusters:
                members = [detections[m] for m in cl["members"]]
                votes = sorted({f"{d['model']}:{d['mode']}" for d in members})
                rep = members[0]
                cand.append(dict(box=[float(x) for x in rep["box"]], score=float(rep["score"]),
                                 votes=votes, n_dets=len(members),
                                 best_iou=float(max(best_per_det[m] for m in cl["members"])),
                                 poly=det_polygon(rep["mask"], shape, (w, h))))

        packet = dict(stem=r["stem"], image=r["source_name"], json=str(source_json) if source_json else "",
                      split=r["new_split"], width=w, height=h, raster=list(shape), n_gt=len(gts),
                      gts=gts, candidates=cand)
        destination.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
        print(f"shard{shard_k} {done + 1}/{len(todo)} {r['source_name']} "
              f"gts={len(gts)} cand={len(cand)} elapsed={time.perf_counter() - elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
