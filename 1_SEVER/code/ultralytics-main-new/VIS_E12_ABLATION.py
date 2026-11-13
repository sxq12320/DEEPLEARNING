"""Run in the foreground. Compare V12_04 to a frozen intervention or a retrained control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

import torch
import cv2
import numpy as np

from citrus_paper_suite import configurations
from citrus_slicing import source_files
from eval_citrus_e_v5 import boundary_iou_pairs, empirical_mask_pr, greedy_pairs
from eval_citrus_sliced import ground_truth, mask_overlaps, merge_candidates, predict_view, raster_shape
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.citrus_e_v11 import EV11DetailExchange, EV11DetailInject
from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import scale_masks

# Set these on your server, then use VS Code -> Run Python File.
PROJECT = Path("/data/sxq/results/PAPER_V12_ABLATION_20261025")
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
FULL_WEIGHTS = PROJECT / "P01_v12_full_seed42_amp0/weights/best_mask.pt"
CONTROL_WEIGHTS = None  # Or the retrained P02 (detail), P03 (route), P06 (phase) best_mask.pt.
MECHANISM = "detail"  # detail / route / phase
ALL_RETRAINED = False  # True compares P01 to the three trained controls in its experiment directory.
DEVICE = "1"
OUTPUT = None  # A fresh directory is allocated next to the project by default.

CONTROLS = {"detail": "P02_no_persistent_detail", "route": "P03_no_recognition", "phase": "P06_bilinear_only"}
CONF = 0.25
MASK_IOU = 0.50
RADIUS = 2
SIZE_GROUPS = (("all", 0, np.inf), ("tiny", 0, 256), ("small", 256, 1024),
               ("medium", 1024, 4096), ("large", 4096, np.inf))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_arm(path, expected):
    """Compare the actual graph in the checkpoint with the predeclared YAML, ignoring source paths."""
    model = YOLO(str(path), task="segment")
    cfg, ref = model.model.yaml, configurations()[expected]
    if any(cfg[key] != ref[key] for key in ("backbone", "head")):
        raise ValueError("Checkpoint graph is not %s: %s" % (expected, path))
    head = model.model.model[-1]
    if head.nc != 1 or not head.quality_calibration:
        raise ValueError("Expected a one-class E12 head with trained quality calibration.")
    head.quality_floor = 0.5
    return model


def disable_exits(model, mechanism):
    """Modify exclusively the in-memory control instance before predictor warmup."""
    stages = model.model.model
    head = stages[-1]
    if mechanism == "detail":
        inject, = [m for m in stages if isinstance(m, EV11DetailInject)]
        gains = {"C4_injection": inject.gain, "head_persistent_blend": head.persistent_gain}
    elif mechanism == "route":
        gains = {"P3_correction": head.recognition[0].gain, "P4_correction": head.recognition[1].gain}
    else:
        gains = {"learned_phase_residual": head.fine_scale}
    before = {k: float(v.detach().tanh().abs().mean()) for k, v in gains.items()}
    with torch.no_grad():
        for v in gains.values():
            v.zero_()
    return before


def all_predictions(model, image, shape, device):
    h, w = image.shape[:2]
    candidates = predict_view(model, image, (0, 0, w, h), shape, device, conf=.001, iou=.7, max_det=300)
    return merge_candidates(candidates, shape)


def predictions(model, image, shape, device):
    pred = all_predictions(model, image, shape, device)
    return pred["masks"][pred["conf"].numpy() >= CONF]


def collect(model, image, shape, device, gt, matcher, ranking_samples, eligible_candidates=None):
    pred = all_predictions(model, image, shape, device)
    overlap = mask_overlaps(gt["masks"], pred["masks"])
    tp = matcher.match_predictions(pred["cls"], gt["cls"], overlap).numpy()
    scores = pred["conf"].numpy()
    ranking_samples.append((tp, scores, len(gt["cls"])))
    if eligible_candidates is not None:
        eligible = (overlap.numpy() >= MASK_IOU).any(axis=0)
        eligible_candidates.append((overlap.numpy()[:, eligible], scores[eligible]))
    keep = scores >= CONF
    return pred["masks"][keep], overlap.numpy()[:, keep]


def ranking(samples):
    flags = np.concatenate([s[0] for s in samples])
    scores = np.concatenate([s[1] for s in samples])
    gt = np.zeros(sum(s[2] for s in samples), dtype=int)
    pred = np.zeros(len(scores), dtype=int)
    ap = ap_per_class(flags, scores, pred, gt)[5] if len(scores) else np.zeros((1, 10))
    pr = empirical_mask_pr(dict(tp_m=flags, conf=scores, pred_cls=pred, target_cls=gt), max_points=None)["0"]
    eligible = [i for i, p in enumerate(pr["precision"]) if p >= .90]
    i = max(eligible, key=lambda j: pr["recall"][j]) if eligible else None
    summary = dict(mask_ap50=float(ap[:, 0].mean()), mask_ap5095=float(ap.mean()), rmax=pr["rmax"],
                   r_at_precision90=pr["recall"][i] if i is not None else None,
                   confidence_at_precision90=pr["confidence"][i] if i is not None else None,
                   precision_at_selected_threshold=pr["precision"][i] if i is not None else None)
    return summary, pr


def write_ranking(directory, sample_sets):
    summary, curve = {}, []
    for arm, samples in sample_sets.items():
        summary[arm], pr = ranking(samples)
        curve.extend(dict(arm=arm, score=c, precision=p, recall=r) for c, p, r in
                     zip(pr["confidence"], pr["precision"], pr["recall"]))
    write_json(directory/"ranking.json", summary)
    if curve:
        write_csv(directory/"observed_pr.csv", curve)
    return summary


def precision90_by_size(directory, instances, image_names, candidates, rankings):
    """Rematch GT at each arm's observed P>=90% threshold. Keep identical area bins and GT IDs."""
    thresholds = {arm: record["confidence_at_precision90"] for arm, record in rankings.items()}
    if any(value is None for value in thresholds.values()):
        write_json(directory/"precision90_by_size.json", dict(thresholds=thresholds, groups=None))
        return
    hits = {}
    for arm, per_image in candidates.items():
        hits[arm] = {}
        for image, (overlap, scores) in zip(image_names, per_image):
            pairs = greedy_pairs(overlap[:, scores >= thresholds[arm]], MASK_IOU)
            hits[arm][image] = {g for g, _ in pairs}
    paired = []
    for row in instances:
        image, idx = row["image"], row["gt_index"]
        paired.append(dict(image=image, gt_index=idx, area640=row["area640"],
                           full_hit=int(idx in hits["full"][image]), control_hit=int(idx in hits["control"][image])))
    groups = {}
    for name, lo, hi in SIZE_GROUPS:
        rows = [r for r in paired if lo <= r["area640"] < hi]
        full, control = (sum(r[k+"_hit"] for r in rows) for k in ("full", "control"))
        groups[name] = dict(n=len(rows), full=full, control=control,
                            gained=sum(r["full_hit"] > r["control_hit"] for r in rows),
                            lost=sum(r["full_hit"] < r["control_hit"] for r in rows),
                            delta_pp=100*(full-control)/len(rows) if rows else None)
    write_json(directory/"precision90_by_size.json", dict(thresholds=thresholds, groups=groups,
               method="Confidence chosen on PR sweep, then one-to-one mask matching repeated at that threshold."))
    write_csv(directory/"precision90_instances.csv", paired)


def measure(gt, pred, overlaps=None):
    if overlaps is None:
        overlaps = mask_overlaps(gt, pred).numpy()
    pairs = greedy_pairs(overlaps, MASK_IOU)
    assigned = {int(g): int(p) for g, p in pairs}
    band = {}
    for g, p in assigned.items():
        values = boundary_iou_pairs(gt, pred, [(g, p)], RADIUS)
        band[g] = values[0] if values else None
    return assigned, band, overlaps


def stats_rows(image, masks, full, control, full_overlap, control_overlap):
    f, fb, _ = measure(masks, full, full_overlap)
    c, cb, _ = measure(masks, control, control_overlap)
    instances = []
    for i, mask in enumerate(masks):
        instances.append(dict(image=image, gt_index=i, area640=int(mask.sum()), full_hit=int(i in f),
                              control_hit=int(i in c), full_band=fb.get(i), control_band=cb.get(i)))
    image_row = dict(image=image, n_gt=len(masks), full_pred=len(full), control_pred=len(control),
                     full_tp=len(f), control_tp=len(c), full_unmatched_pred=len(full)-len(f),
                     control_unmatched_pred=len(control)-len(c))
    return instances, image_row


def paired_summary(records, image_names):
    """Pair by GT. Bootstrap images (including images without eligible GT), conditional on fitted weights."""
    index = {name: i for i, name in enumerate(image_names)}
    rng = np.random.default_rng(20261103)
    draws = rng.integers(len(index), size=(2000, len(index)))
    results = {}
    for name, lo, hi in SIZE_GROUPS:
        rows = [r for r in records if lo <= r["area640"] < hi]
        totals = np.zeros((len(index), 4))
        full_hits = control_hits = gain = loss = 0
        bf, bc = [], []
        for r in rows:
            f, c = r["full_hit"], r["control_hit"]
            full_hits += f
            control_hits += c
            gain += int(f > c)
            loss += int(c > f)
            counts = totals[index[r["image"]]]
            counts[:2] += (f-c, 1)
            if f and c and r["full_band"] is not None and r["control_band"] is not None:
                bf.append(r["full_band"])
                bc.append(r["control_band"])
                counts[2:] += (bf[-1]-bc[-1], 1)
        sampled = totals[draws].sum(1)

        def interval(num, denom):
            valid = sampled[:, denom] > 0
            if not valid.any():
                return None
            return (100 * np.percentile(sampled[valid, num] / sampled[valid, denom], [2.5, 97.5])).tolist()

        results[name] = dict(n=len(rows), full_hits=full_hits, control_hits=control_hits, gained=gain,
                             lost=loss, both=full_hits-gain, neither=len(rows)-control_hits-gain,
                             delta_recall_pp=100*(gain-loss)/len(rows) if rows else None,
                             delta_recall_image_bootstrap_ci95_pp=interval(0, 1),
                             common_boundary_n=len(bf),
                             full_common_band=float(np.mean(bf)) if bf else None,
                             control_common_band=float(np.mean(bc)) if bc else None,
                             delta_common_band_pp=float(100*np.mean(np.subtract(bf, bc))) if bf else None,
                             delta_common_band_image_bootstrap_ci95_pp=interval(2, 3))
    return results


def example_cases(records):
    """One median-size recovered fruit, one typical boundary improvement, and one counterexample."""
    valid = [r for r in records if r["area640"] > 0]
    common = [r for r in valid if r["full_band"] is not None and r["control_band"] is not None]
    gained = [r for r in valid if r["full_hit"] > r["control_hit"]]
    lost = [r for r in valid if r["full_hit"] < r["control_hit"]]
    delta = lambda r: r["full_band"] - r["control_band"]
    area = lambda r: r["area640"]
    selections = []
    used = set()
    choices = [
        ("Recovered", [r for r in gained if r["area640"] < 256] or gained, area),
        ("Contour gain", [r for r in common if delta(r) >= .05], delta),
        ("Lost", [r for r in lost if r["area640"] < 256] or lost, area),
        ("Contour loss", [r for r in common if delta(r) <= -.05], delta),
        ("Both miss", [r for r in valid if not r["full_hit"] and not r["control_hit"]], area),
        ("Both match", common, area),
    ]
    for label, candidates, value in choices:
        if len(selections) == 3:
            break
        if label == "Contour loss" and any(s["category"] == "Lost" for s in selections):
            continue
        candidates = [r for r in candidates if r["image"] not in used]
        if candidates:
            median = np.median([value(r) for r in candidates])
            selected = min(candidates, key=lambda r: (abs(value(r)-median), r["image"], r["gt_index"]))
            selections.append(dict(selected, category=label))
            used.add(selected["image"])
    return selections


def correction_view(model, image, shape, device, mechanism):
    """Capture the actual additive update from the same forward pass as the reported full prediction."""
    stages = model.model.model
    head = stages[-1]
    features = {}
    with ExitStack() as hooks:
        def watch(module, key, transform):
            def capture(_module, arguments, out):
                features[key] = transform(arguments, out).detach()
            hooks.callback(module.register_forward_hook(capture).remove)
        if mechanism == "detail":
            exchanges = [m for m in stages if isinstance(m, EV11DetailExchange)]
            watch(exchanges[-1], "persistent", lambda args, out: out)
            watch(head.refiner, "base", lambda args, out: out)
        elif mechanism == "route":
            watch(head.recognition[0], "residual", lambda args, out: out-args[0])
        else:
            watch(head.fine_to_proto, "phase", lambda args, out: out)
        pred = predictions(model, image, shape, device)
    if mechanism == "detail":
        update = head.persistent_gain.detach().tanh() * (features["persistent"]-features["base"])
    elif mechanism == "route":
        update = features["residual"]
    else:
        update = head.fine_scale.detach().tanh()*torch.nn.functional.pixel_shuffle(features["phase"], 2)
    energy = update.abs().mean(1, keepdim=True)
    return pred, scale_masks(energy, shape)[0, 0].cpu().numpy()


def square_crop(mask):
    ys, xs = np.where(mask)
    h, w = mask.shape
    side = min(h, w, max(56, int(2.3*max(np.ptp(ys)+1, np.ptp(xs)+1))))
    x = int(np.clip((xs.min()+xs.max()-side+1)//2, 0, w-side))
    y = int(np.clip((ys.min()+ys.max()-side+1)//2, 0, h-side))
    return x, y, side


def case_arrays(selected, path, full, control, device, mechanism, destination):
    image = cv2.imread(str(path))
    shape = raster_shape(*image.shape[:2], 640)
    masks = ground_truth(path, image.shape[:2], shape)["masks"]
    pf, heat = correction_view(full, image, shape, device, mechanism)
    pc = predictions(control, image, shape, device)
    g = selected["gt_index"]
    x, y, side = square_crop(masks[g])
    crop = np.s_[y:y+side, x:x+side]
    arrays = {"rgb": cv2.resize(image, shape[::-1], interpolation=cv2.INTER_AREA)[crop][:, :, ::-1],
              "gt": masks[g][crop], "update": heat[crop]}
    for tag, pred in (("full", pf), ("control", pc)):
        pairs, _, overlap = measure(masks, pred)
        if (g in pairs) != bool(selected[tag+"_hit"]):
            raise RuntimeError("Example rerun differs from census: %s / %s" % (path, tag))
        # If unmatched, show the best-overlapping retained prediction, even if IoU < .5.
        p = pairs.get(g)
        if p is None and len(pred) and overlap[g].max() > 0:
            p = int(overlap[g].argmax())
        arrays[tag] = pred[p][crop] if p is not None else np.zeros((side, side), bool)
        selected[tag+"_shown_iou"] = float(overlap[g, p]) if p is not None else 0.0
    selected.update(crop_xywh=[x, y, side, side], raster_hw=list(shape), file=destination.name)
    np.savez_compressed(destination, **arrays)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--full", type=Path, default=FULL_WEIGHTS)
    p.add_argument("--control", type=Path, default=CONTROL_WEIGHTS)
    p.add_argument("--mechanism", choices=tuple(CONTROLS), default=MECHANISM)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--all-retrained", action="store_true", default=ALL_RETRAINED,
                   help="Compare P01 against trained P02/P03/P06 from the same project and seed")
    return p.parse_args()


def main(args):
    from datetime import datetime

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    for path in (args.data, args.full, args.control):
        if path is not None and not path.is_file():
            raise FileNotFoundError("Set DATA / FULL_WEIGHTS / CONTROL_WEIGHTS to your files: %s" % path)
    config, files = source_files(args.data, args.split)
    if len(config["names"]) != 1:
        raise ValueError("This diagnostic requires a single foreground class.")
    data_root = Path(args.data).resolve().parent
    keyed = {Path(os.path.relpath(f, data_root)).as_posix(): f for f in files}
    full = load_arm(args.full, "P01_v12_full")
    control = load_arm(args.control or args.full, CONTROLS[args.mechanism] if args.control else "P01_v12_full")
    gains = {} if args.control else disable_exits(control, args.mechanism)
    control_type = "retrained" if args.control else "frozen"
    target = args.output or (PROJECT.parent / ("E12_%s_%s_%s" % (
        args.mechanism, control_type, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))))
    target.mkdir(parents=True, exist_ok=False)
    protocol = dict(split=args.split, data=str(args.data.resolve()), full=str(args.full.resolve()),
                    control=str((args.control or args.full).resolve()), control_type=control_type,
                    mechanism=args.mechanism, full_sha256=hashlib.sha256(args.full.read_bytes()).hexdigest(),
                    control_sha256=hashlib.sha256((args.control or args.full).read_bytes()).hexdigest(),
                    source_images=len(files), imgsz=640, views="one whole image", conf_prefilter=.001,
                    conf_operating=CONF, mask_iou=MASK_IOU, boundary_band_radius_px=RADIUS, gain_before=gains,
                    evaluation="public640, one-to-one IoU-greedy, two-sided contour band on common matches",
                    bootstrap="2000 paired image draws, fixed weights; no training-seed or acquisition-group interval",
                    software=dict(torch=str(torch.__version__), device=str(args.device)))
    write_json(target/"protocol.json", protocol)
    for model in (full, control):
        model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, device=args.device, verbose=False, save=False)
    rows, images = [], []
    matcher = SegmentationValidator(args=dict(plots=False))
    samples = {"full": [], "control": []}
    eligible = {"full": [], "control": []}
    for n, (key, path) in enumerate(keyed.items(), 1):
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError("Unreadable image: %s" % path)
        shape = raster_shape(*im.shape[:2], 640)
        gt = ground_truth(path, im.shape[:2], shape)
        pf, of = collect(full, im, shape, args.device, gt, matcher, samples["full"], eligible["full"])
        pc, oc = collect(control, im, shape, args.device, gt, matcher, samples["control"], eligible["control"])
        inst, img = stats_rows(key, gt["masks"], pf, pc, of, oc)
        rows.extend(inst)
        images.append(img)
        if n % 20 == 0 or n == len(keyed):
            print("%d/%d %s" % (n, len(keyed), key), flush=True)
    if not rows:
        raise ValueError("No GT instances in selected split.")
    write_csv(target/"instances.csv", rows)
    write_csv(target/"images.csv", images)
    rankings = write_ranking(target, samples)
    precision90_by_size(target, rows, list(keyed), eligible, rankings)
    summary = paired_summary(rows, list(keyed))
    write_json(target/"summary.json", summary)
    cases = example_cases(rows)
    for i, case in enumerate(cases, 1):
        case_arrays(case, keyed[case["image"]], full, control, args.device, args.mechanism,
                    target/("case_%02d.npz" % i))
    write_json(target/"cases.json", cases)
    from scripts.plot_e12_ablation import render
    render(target)
    print("Figures and per-GT data:", target, flush=True)


def run(args):
    if not args.all_retrained:
        return main(args)
    if args.control:
        raise ValueError("Specify one --control or use --all-retrained.")
    run_dir = args.full.parent.parent
    prefix = "P01_v12_full_"
    if not run_dir.name.startswith(prefix):
        raise ValueError("For --all-retrained, use a P01_v12_full_seed*_amp* run as --full.")
    suffix = run_dir.name[len(prefix):]
    controls = {key: run_dir.parent / (name + "_" + suffix) / "weights" / args.full.name
                for key, name in CONTROLS.items()}
    missing = [str(p) for p in (args.full, *controls.values()) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained checkpoint(s): " + ", ".join(missing))
    if args.output and args.output.exists():
        raise FileExistsError("Choose a fresh output directory: %s" % args.output)
    for mechanism, weight in controls.items():
        options = vars(args).copy()
        options.update(mechanism=mechanism, control=weight, output=args.output/mechanism if args.output else None)
        main(argparse.Namespace(**options))


if __name__ == "__main__":
    run(arguments())
