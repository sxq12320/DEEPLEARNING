"""Read-only D1/D2/D4 probes on historical weights; never replaces EV6 validation.

Square 640 inputs, EV6 input/2 prototype interpolation/crop, then a common
640-long-side image raster for paired multi-view diagnostics. This extra raster
is NOT the training CSV metric. Test data and model parameters are not changed.
"""

# ruff: noqa: E402
import argparse
import csv
import hashlib
import html
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from citrus_e_v5_slicing import uniform_windows
from citrus_slicing import source_files
from eval_citrus_e_v5 import add_view_metadata, patch_overlap
from eval_citrus_sliced import ground_truth, mask_overlaps, raster_shape
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import process_mask, scale_boxes, scale_masks, xywh2xyxy

MODES = ("global", "fine", "upsample_linear", "upsample_area")
STAGES = ("S0", "S1", "S2", "S3", "S4", "matching_competition", "unresolved")


def ev6_pairs(iou, threshold=0.5):
    """Return GT/prediction witnesses using EV6's default IoU-sorted matching."""
    iou = np.asarray(iou)
    matches = np.array(np.nonzero(iou >= threshold)).T
    if len(matches) > 1:
        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches.astype(int).reshape(-1, 2)


def tp_matrix(overlap):
    """Compute the same ten-IoU TP flags as the EV6 matcher, single class."""
    tp = np.zeros((overlap.shape[1], 10), bool)
    for k, threshold in enumerate(torch.linspace(0.5, 0.95, 10).tolist()):
        pairs = ev6_pairs(overlap, threshold)
        tp[pairs[:, 1], k] = True
    return tp


def metric_rows(rows):
    """Aggregate globally ranked AP, never the average of per-image AP."""
    count = sum(r["targets"] for r in rows)
    if not count:
        return dict(ap50=None, ap5095=None, recall25=None, tiny_recall25=None)
    conf = np.concatenate([np.asarray(r["conf"], float) for r in rows])
    tp = np.concatenate([np.asarray(r["tp_m"], bool).reshape(-1, 10) for r in rows])
    if len(conf):
        result = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(count), plot=False)
        ap = result[5]
        ap50, ap5095 = float(ap[:, 0].mean()), float(ap.mean())
    else:
        ap50 = ap5095 = 0.0
    tiny_n = sum(r["tiny_targets"] for r in rows)
    return dict(ap50=ap50, ap5095=ap5095, recall25=sum(r["hits25"] for r in rows) / count,
                tiny_recall25=sum(r["tiny_hits25"] for r in rows) / tiny_n if tiny_n else None)


def traced_merge(candidates, shape, single_view=False):
    """Replay existing multi-view suppression while retaining candidate identity."""
    ordered = []
    for c in candidates:
        weight = 1.0 if single_view or not c["internal_border"] else 0.5
        ordered.append({**c, "score": c["score"] * weight, "area": int(c["patch"].sum())})
    ordered.sort(key=lambda c: c["score"], reverse=True)
    keep, removed = [], {}
    for c in ordered:
        suppressor = None
        if not single_view:
            for chosen in keep:
                if chosen["cls"] == c["cls"] and chosen["view"] != c["view"]:
                    iou, ios = patch_overlap(chosen, c)
                    if iou >= 0.5 or (ios >= 0.8 and (chosen["internal_border"] or c["internal_border"])):
                        suppressor = chosen["uid"]
                        break
        if suppressor is not None:
            removed[c["uid"]] = ("cross_view_mask", suppressor)
        elif len(keep) >= 300:
            removed[c["uid"]] = ("merged_max_det", None)
        else:
            keep.append(c)
    masks = np.zeros((len(keep), *shape), bool)
    for i, c in enumerate(keep):
        x, y = c["offset"]
        h, w = c["patch"].shape
        masks[i, y:y + h, x:x + w] = c["patch"]
    return keep, masks, removed


def infer_view(model, crop, window, original_shape, shape, view):
    """Capture raw candidate IDs and decode kept masks with EV6's operation order."""
    boxed = LetterBox((640, 640), auto=False, scaleup=True)(image=crop)
    tensor = torch.from_numpy(np.ascontiguousarray(boxed[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
    with torch.inference_mode():
        output = model(tensor)
        raw, proto = output[0]
        assert raw.shape[1] == 37 and proto.shape[1] == 32, "Probe requires single-class 32-proto head"
        raw_boxes = xywh2xyxy(raw[0, :4].T).clone()
        raw_scores = raw[0, 4].clone()
        # Unlimited survivors permit max_det loss to be distinguished from box suppression.
        full, ids = non_max_suppression(raw.clone(), 0.001, 0.7, nc=1, max_det=raw.shape[-1], return_idxs=True)
        all_ids = ids[0]
        pred, kept_ids = full[0][:300], all_ids[:300]
        resized_proto = F.interpolate(proto, (320, 320), mode="bilinear", align_corners=False)
        masks = (process_mask(resized_proto[0], pred[:, 6:], pred[:, :4], (640, 640))
                 if len(pred) else torch.zeros((0, 320, 320), dtype=torch.uint8))
        h, w = original_shape
        eh, ew = shape
        x0, y0, x1, y1 = window
        gx0, gx1 = round(x0 * ew / w), round(x1 * ew / w)
        gy0, gy1 = round(y0 * eh / h), round(y1 * eh / h)
        mapped = (scale_masks(masks[None].float(), (gy1 - gy0, gx1 - gx0))[0].gt(0.5).numpy()
                  if len(pred) else np.zeros((0, gy1 - gy0, gx1 - gx0), bool))
        raw_boxes = scale_boxes((640, 640), raw_boxes, crop.shape[:2]).numpy() + [x0, y0, x0, y0]
        boxes = scale_boxes((640, 640), pred[:, :4].clone(), crop.shape[:2]).numpy() + [x0, y0, x0, y0]
    candidates = []
    for box, p, idx, mask in zip(boxes, pred, kept_ids.tolist(), mapped):
        yy, xx = np.nonzero(mask)
        if len(xx):
            left, top, right, bottom = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
            patch, offset = mask[top:bottom, left:right].copy(), (gx0 + int(left), gy0 + int(top))
        else:
            patch, offset = np.zeros((0, 0), bool), (0, 0)
        candidates.append(dict(box=box, score=float(p[4]), cls=0, patch=patch, offset=offset,
                               uid=f"{view}:{idx}"))
    add_view_metadata(candidates, window, original_shape, view)
    valid = raw_scores.numpy() > 0.001
    raw_ids = np.flatnonzero(valid)
    return dict(candidates=candidates, boxes=raw_boxes[valid], scores=raw_scores.numpy()[valid],
                uids=[f"{view}:{i}" for i in raw_ids], survivors=set(all_ids.tolist()),
                capped=set(all_ids[300:].tolist()), view=view, candidates_at_floor=int(valid.sum()))


def properties(gt, image, shape):
    """Measure visible-mask shape, spacing and contrast on the common image raster."""
    eh, ew = shape
    gray = cv2.cvtColor(cv2.resize(image, (ew, eh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(float)
    masks = gt["masks"]
    distances = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in masks]
    h, w = image.shape[:2]
    boxes = gt["boxes"].numpy() * [ew / w, eh / h, ew / w, eh / h]
    result = []
    kernel = np.ones((7, 7), np.uint8)
    allmask = masks.any(0) if len(masks) else np.zeros(shape, bool)
    for i, mask in enumerate(masks):
        area = int(mask.sum())
        pts = cv2.findNonZero(mask.astype(np.uint8))
        hull = np.zeros(shape, np.uint8)
        if pts is not None:
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~allmask
        inside, outside = gray[mask], gray[ring]
        cnr = (abs(inside.mean() - outside.mean()) / max(math.sqrt(inside.var() + outside.var()), 1e-9)
               if len(inside) and len(outside) else None)
        gap = min((float(distances[j][mask].min()) for j in range(len(masks)) if i != j and area), default=None)
        box = boxes[i]
        result.append(dict(gt_index=i, area640=area, size640=math.sqrt(area),
                           short640=float(min(box[2] - box[0], box[3] - box[1])),
                           solidity=area / max(int(hull.sum()), 1), gap640=gap, cnr=cnr))
    return result


def classify(gt, props, views, final, masks, removed, threshold):
    """Separate actual mask matches from box funnel proxies and capacity limits."""
    raw_boxes = torch.tensor(np.concatenate([v["boxes"] for v in views]), dtype=torch.float32)
    raw_scores = np.concatenate([v["scores"] for v in views])
    uids = sum((v["uids"] for v in views), [])
    lookup = {v["view"]: v for v in views}
    boxes = torch.tensor(np.asarray([c["box"] for c in final]).reshape(-1, 4), dtype=torch.float32)
    scores = np.asarray([c["score"] for c in final])
    overlap = mask_overlaps(gt["masks"], masks).numpy()
    geom = box_iou(gt["boxes"], raw_boxes).numpy()
    post_geom = box_iou(gt["boxes"], boxes).numpy()
    work = np.flatnonzero(scores >= threshold)
    pairs = ev6_pairs(overlap[:, work], 0.5)
    matched = {int(g): int(work[p]) for g, p in pairs}
    pred_gt = {p: g for g, p in matched.items()}
    kept_uids = {c["uid"] for c in final}
    records = []
    for j, prop in enumerate(props):
        raw_eligible = np.flatnonzero(geom[j] >= 0.5)
        passing = raw_eligible[raw_scores[raw_eligible] >= threshold]
        post = work[post_geom[j, work] >= 0.5]
        subtype, witness, suppressor = None, None, None
        if j in matched:
            stage, witness = "S4", final[matched[j]]["uid"]
        elif not len(raw_eligible):
            stage = "S0"
        elif not len(passing):
            stage = "S1"
        elif not len(post):
            stage = "S2"
            # Every passing geometrical candidate is absent; track the best one.
            k = passing[np.argmax(raw_scores[passing])]
            witness = uids[k]
            view_id, idx = map(int, witness.split(":"))
            view = lookup[view_id]
            if idx in view["capped"]:
                subtype = "per_view_max_det"
            elif idx not in view["survivors"]:
                subtype = "box_nms"
                survivors = [c for c in view["candidates"] if c["score"] >= raw_scores[k]]
                if survivors:
                    ious = box_iou(raw_boxes[k:k + 1], torch.tensor(np.asarray([c["box"] for c in survivors])))[0]
                    ids = torch.where(ious > 0.7)[0]
                    if len(ids):
                        suppressor = survivors[int(ids[0])]["uid"]
            elif witness in removed:
                subtype, suppressor = removed[witness]
            elif witness in kept_uids:
                subtype = "border_score_penalty"
            else:
                stage, subtype = "unresolved", "untracked_removal"
        elif not (overlap[j, work] >= 0.5).any():
            stage = "S3"
            witness = final[int(post[np.argmax(overlap[j, post])])]["uid"]
        else:
            stage = "matching_competition"
        suppression_kind = None
        if suppressor is not None:
            final_index = next((k for k, c in enumerate(final) if c["uid"] == suppressor), None)
            suppression_kind = ("another_gt_tp" if final_index in pred_gt else "fp_or_removed")
        records.append({**prop, "threshold": threshold, "stage": stage, "subtype": subtype,
                        "witness": witness, "suppressor": suppressor, "suppressor_kind": suppression_kind,
                        "raw_box_score_max": float(raw_scores[raw_eligible].max()) if len(raw_eligible) else None,
                        "best_post_mask_iou": float(overlap[j, work].max()) if len(work) else 0.0,
                        "matched_mask_iou": float(overlap[j, matched[j]]) if j in matched else None,
                        "matched_box_iou": float(post_geom[j, matched[j]]) if j in matched else None,
                        "box_gate_exception": bool(j in matched and post_geom[j, matched[j]] < 0.5),
                        "matched25": j in matched})
    flags = tp_matrix(overlap)
    hit = set(matched)
    tiny = {p["gt_index"] for p in props if p["area640"] < 256}
    stats = dict(targets=len(props), conf=scores.tolist(), tp_m=flags.tolist(),
                 hits25=len(hit), tiny_targets=len(tiny), tiny_hits25=len(tiny & hit))
    return records, stats, overlap, pairs, work


def export_crop(image, box, output, mask=None, score=None):
    """Save a read-only contextual thumbnail; no label or source edits."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    margin = max(24, int(max(x1 - x0, y1 - y0) * 0.6))
    left, top = max(0, int(x0) - margin), max(0, int(y0) - margin)
    right, bottom = min(w, int(x1) + margin), min(h, int(y1) + margin)
    view = image[top:bottom, left:right].copy()
    if not view.size:
        return
    if mask is not None:
        native = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[top:bottom, left:right] > 0
        overlay = np.zeros_like(view)
        overlay[:, :, 2] = 255
        view[native] = (0.55 * view[native] + 0.45 * overlay[native]).astype(np.uint8)
    view = cv2.resize(view, (256, 256), interpolation=cv2.INTER_AREA)
    if score is not None:
        cv2.putText(view, f"score {score:.3f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(output), view)


def rho_bootstrap(rows, groups, repetitions=1000):
    """Paired cluster intervals with undefined denominator replicates retained."""
    names = list(rows["global"])
    units = sorted(set(groups[n] for n in names))
    members = {u: [n for n in names if groups[n] == u] for u in units}
    rng = np.random.default_rng(42)
    estimates = {m: [] for m in ("upsample_linear", "upsample_area")}
    invalid = {m: 0 for m in estimates}
    for _ in range(repetitions):
        chosen = sum((members[u] for u in rng.choice(units, len(units), replace=True)), [])
        recalls = {}
        for mode in MODES:
            recs = [rows[mode][n] for n in chosen]
            n = sum(r["tiny_targets"] for r in recs)
            recalls[mode] = sum(r["tiny_hits25"] for r in recs) / n if n else None
        denominator = (recalls["fine"] - recalls["global"]
                       if recalls["fine"] is not None and recalls["global"] is not None else 0)
        for mode in estimates:
            if denominator <= 1e-8:
                invalid[mode] += 1
            else:
                estimates[mode].append((recalls[mode] - recalls["global"]) / denominator)
    return {m: dict(ci95=np.quantile(v, [0.025, 0.975]).tolist() if v else None,
                    invalid_denominator_replicates=invalid[m], repetitions=repetitions)
            for m, v in estimates.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Only reuse this run's completed per-image files")
    parser.add_argument("--group-manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume for identical inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(exist_ok=True)
    config, files = source_files(args.data, "val")
    if args.limit:
        files = files[:args.limit]
    fingerprint = dict(weights=str(args.weights.resolve()), sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                       data=str(args.data.resolve()), images=[str(f) for f in files],
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       protocol="extra_common640_ev6decode_square640_v1", torch=torch.__version__)
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != fingerprint:
        raise ValueError("Cannot resume diagnostic with changed input or protocol")
    manifest.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    model = YOLO(str(args.weights)).model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    group_lookup = {}
    if args.group_manifest:
        with args.group_manifest.open(encoding="utf-8-sig") as stream:
            group_lookup = {r["stem"]: r["group_id"] for r in csv.DictReader(stream)}
    groups = {f.name: group_lookup.get(f.stem, f.name) for f in files}
    rows, all_records, fp_entries = {m: {} for m in MODES}, [], {m: [] for m in ("global", "fine")}
    elapsed = time.perf_counter()
    for image_number, file in enumerate(files):
        destination = args.output / f"image_{image_number:04d}.json"
        if destination.exists() and args.resume:
            packet = json.loads(destination.read_text(encoding="utf-8"))
        else:
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError(file)
            h, w = image.shape[:2]
            shape = raster_shape(h, w, 640)
            windows = uniform_windows(h, w, 0.4)
            gt = ground_truth(file, (h, w), shape)
            props = properties(gt, image, shape)
            # The geometrical crop windows are identical. Only the RGB observations change.
            low_linear = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            low_area = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            recon_linear = cv2.resize(low_linear, (w, h), interpolation=cv2.INTER_CUBIC)
            recon_area = cv2.resize(low_area, (w, h), interpolation=cv2.INTER_CUBIC)
            global_view = infer_view(model, image, windows[0], (h, w), shape, 0)
            packet = dict(image=file.name, group=groups[file.name], stats={}, records=[], fp={})
            for mode in MODES:
                source = image if mode in ("global", "fine") else recon_linear if mode == "upsample_linear" else recon_area
                views = [global_view]
                start = time.perf_counter()
                if mode != "global":
                    for view, win in enumerate(windows[1:], 1):
                        x0, y0, x1, y1 = win
                        views.append(infer_view(model, source[y0:y1, x0:x1], win, (h, w), shape, view))
                final, masks, removed = traced_merge(sum((v["candidates"] for v in views), []), shape, mode == "global")
                for threshold in (0.001, 0.25):
                    records, stats, overlap, pairs, work = classify(gt, props, views, final, masks, removed, threshold)
                    for record in records:
                        record.update(image=file.name, mode=mode, group=groups[file.name])
                        if mode in ("global", "fine") and threshold == 0.25:
                            example = assets / f"stage_{mode}_{record['stage']}_{image_number}_{record['gt_index']}.jpg"
                            # Keep up to twelve reproducible earliest image examples per stage.
                            if image_number < 30:
                                export_crop(image, gt["boxes"][record["gt_index"]].numpy(), example)
                                record["example"] = example.relative_to(args.output).as_posix()
                    packet["records"].extend(records)
                    if threshold == 0.25:
                        stats.update(image=file.name, group=groups[file.name], views=len(views),
                                     per_image_seconds=time.perf_counter() - start)
                        packet["stats"][mode] = stats
                        if mode in ("global", "fine"):
                            matched_pred = set(work[pairs[:, 1]].tolist())
                            fp = []
                            for k in work:
                                if int(k) in matched_pred:
                                    continue
                                c = final[int(k)]
                                fp.append(dict(image=file.name, index=int(k), score=c["score"],
                                               box=np.asarray(c["box"]).tolist(),
                                               max_mask_iou=float(overlap[:, k].max()) if len(props) else 0.0,
                                               human_label="", semantic_status="unverified"))
                            for c in sorted(fp, key=lambda r: -r["score"])[:8]:
                                asset = assets / f"fp_{mode}_{image_number}_{c['index']}.jpg"
                                export_crop(image, c["box"], asset, masks[c["index"]], c["score"])
                                c["asset"] = asset.relative_to(args.output).as_posix()
                            packet["fp"][mode] = [r for r in fp if "asset" in r]
            destination.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
        for mode in MODES:
            rows[mode][file.name] = packet["stats"][mode]
        all_records.extend(packet["records"])
        for mode in fp_entries:
            fp_entries[mode].extend(packet["fp"].get(mode, []))
        print(f"{image_number + 1}/{len(files)} {file.name}, elapsed={time.perf_counter() - elapsed:.1f}s", flush=True)
    metrics = {m: metric_rows(list(r.values())) for m, r in rows.items()}
    rho = {}
    denominator = (metrics["fine"]["tiny_recall25"] - metrics["global"]["tiny_recall25"]
                   if metrics["fine"]["tiny_recall25"] is not None else 0)
    for mode in ("upsample_linear", "upsample_area"):
        rho[mode] = ((metrics[mode]["tiny_recall25"] - metrics["global"]["tiny_recall25"]) / denominator
                     if denominator > 1e-8 else None)
    summary = dict(evidence="旧划分探索", images=len(files), instances=sum(r["targets"] for r in rows["global"].values()),
                   params=sum(p.numel() for p in model.parameters()), metrics=metrics, rho=rho,
                   rho_intervals=rho_bootstrap(rows, groups),
                   group_mapping_coverage=sum(f.stem in group_lookup for f in files),
                   stages={m: dict(Counter(r["stage"] for r in all_records if r["mode"] == m and r["threshold"] == 0.25))
                           for m in MODES},
                   limits=["Additional common640 paired raster, not native training EV6 AP or COCO AP.",
                           "Low rho is not proof of information loss; blur/domain shift/finite model remain confounders.",
                           "No semantic FP claims until human labels. One historical model, not new training evidence.",
                           "Square inference, FP32 CPU. Timing is not comparable to historical GPU speed."])
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = sorted(set().union(*(r.keys() for r in all_records)))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_records)
    with (args.output / "prediction_stats.jsonl").open("w", encoding="utf-8") as stream:
        for mode in MODES:
            for row in rows[mode].values():
                stream.write(json.dumps({**row, "mode": mode}) + "\n")
    for mode, entries in fp_entries.items():
        entries.sort(key=lambda r: -r["score"])
        # Fifty highest-scored plus fifty spread over the remaining score range.
        chosen = entries[:50]
        rest = entries[50:]
        if rest:
            chosen += [rest[i] for i in np.unique(np.linspace(0, len(rest) - 1, min(50, len(rest))).astype(int))]
        if chosen:
            with (args.output / f"fp_{mode}_human_labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(chosen[0]))
                writer.writeheader()
                writer.writerows(chosen)
        table = "\n".join(f"<tr><td>{html.escape(r['image'])}</td><td>{r['score']:.4f}</td>"
                           f"<td><img src='{r['asset']}' width='256'></td><td contenteditable='true'></td></tr>" for r in chosen)
        (args.output / f"fp_{mode}.html").write_text("<meta charset='utf-8'><h1>Unverified false positives</h1>"
                                                    "<p>Manual labels must be saved to the CSV. Leaf/branch/missing fruit/"
                                                    "highlight/duplicate/partial fruit/other.</p><table>" + table + "</table>",
                                                    encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
