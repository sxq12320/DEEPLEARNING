"""Audit uploaded V4R records and optionally revalidate checkpoints on LOCAL validation data.

Never writes labels or replaces server metrics. Diagnostic subset recall is NOT COCO AP_small.
"""

# ruff: noqa: E402 -- local checkout takes precedence for direct script execution
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(handle)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(results, output):
    rows = []
    for path in sorted(results.glob("*/results.csv")):
        history = read_rows(path)
        peak = max(history, key=lambda row: row["metrics/mAP50-95(M)"])
        run = path.parent
        args = yaml.safe_load((run / "args.yaml").read_text())
        init = json.loads((run / "initialization_transfer.json").read_text())
        row = dict(
            name=run.name,
            path=str(path),
            epochs=len(history),
            peak=peak,
            tail20_ap=statistics.mean(r["metrics/mAP50-95(M)"] for r in history[-20:]),
            epoch_median_seconds=statistics.median(b["time"] - a["time"] for a, b in zip(history, history[1:])),
            args=args,
            params=init["total_parameter_numel"],
            initialization_fraction=init["equal_fraction"],
            loaded=json.loads((run / "loaded_data_summary.json").read_text()),
            train_list_sha256=sha(run / "train_loaded_files.txt"),
            val_list_sha256=sha(run / "val_loaded_files.txt"),
        )
        rows.append(row)
    baseline = next(r for r in rows if r["name"].startswith("SAGE30_"))
    ignore = {"model", "name", "project", "save_dir"}
    for row in rows:
        row["argument_differences"] = {
            k: [baseline["args"].get(k), row["args"].get(k)]
            for k in baseline["args"].keys() | row["args"].keys()
            if k not in ignore and baseline["args"].get(k) != row["args"].get(k)
        }
    recorded = json.loads((results / "_protocol/implementation_sha256.json").read_text())
    hashes = []
    for name, value in recorded.items():
        path = ROOT / ("yolo11n-seg.pt" if name == "initialization_checkpoint" else name)
        hashes.append(dict(file=name, exists=path.is_file(), exact_match=path.is_file() and sha(path) == value))
    payload = dict(
        runs=rows,
        source_hash_comparison=hashes,
        limits="Single seed; epoch timing is not a controlled latency benchmark; list hashes do not prove image bytes.",
    )
    (output / "v4r_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# SAGE V4R 结果核对",
        "",
        "全部 AP 为百分数，AP50 与 AP50–95 取同一峰值轮。尾20均值不是多种子统计。",
        "",
        "| 模型 | 轮数 | Mask AP50–95 | 同轮 AP50 | 对基线差值/百分点 | 尾20 AP | 参数/M | 初始化相等比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        p = row["peak"]
        lines.append(
            f"| {row['name']} | {row['epochs']} | {100 * p['metrics/mAP50-95(M)']:.3f} | "
            f"{100 * p['metrics/mAP50(M)']:.3f} | "
            f"{100 * (p['metrics/mAP50-95(M)'] - baseline['peak']['metrics/mAP50-95(M)']):+.3f} | "
            f"{100 * row['tail20_ap']:.3f} | {row['params'] / 1e6:.3f} | {100 * row['initialization_fraction']:.2f}% |"
        )
    lines += ["", "本表来自逐轮 CSV，不替换原始结果。完整参数、源代码校验及文件清单校验见同目录 v4r_audit.json。"]
    (output / "V4R_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                runs=len(rows),
                argument_differences={r["name"]: r["argument_differences"] for r in rows},
                unequal_sources=[r for r in hashes if not r["exact_match"]],
            ),
            indent=2,
        ),
        flush=True,
    )


def greedy_pairs(iou, threshold):
    """IoU-ordered one-to-one matching; mirrors the validator's non-SciPy convention."""
    g, p = np.where(iou >= threshold)
    if not len(g):
        return np.empty((0, 2), dtype=int)
    matches = np.stack((g, p, iou[g, p]), 1)
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches[:, :2].astype(int)


def validate(results, dataset, output, names):
    import cv2
    import torch
    import torch.nn.functional as F

    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils.metrics import box_iou

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    local_yaml = output / "local_validation.yaml"
    local_yaml.write_text(
        yaml.safe_dump(
            dict(path=str(dataset.resolve()), train="train/images", val="val/images", names={0: "orange_immature"})
        ),
        encoding="utf-8",
    )

    class ChallengeValidator(SegmentationValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.diagnostic = []

        def _prepare_batch(self, si, batch):
            prepared = super()._prepare_batch(si, batch)
            rgb = F.interpolate(batch["img"][si : si + 1], size=prepared["masks"].shape[-2:], mode="area")
            prepared["rgb_small"] = rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            return prepared

        def _process_batch(self, preds, batch):
            result = super()._process_batch(preds, batch)
            gt = batch["masks"].float()
            pred = preds["masks"].float()
            if not len(gt):
                return result
            intersection = (gt.flatten(1) @ pred.flatten(1).T).cpu().numpy()
            area = gt.sum((1, 2)).cpu().numpy()
            pred_area = pred.sum((1, 2)).cpu().numpy()
            iou = intersection / np.maximum(area[:, None] + pred_area[None] - intersection, 1e-7)
            boxes = box_iou(batch["bboxes"], preds["bboxes"]).cpu().numpy()
            confidence = preds["conf"].cpu().numpy()
            sy, sx = batch["imgsz"][0] / gt.shape[1], batch["imgsz"][1] / gt.shape[2]
            masks = gt.cpu().numpy().astype(np.uint8)
            lab = cv2.cvtColor(batch["rgb_small"], cv2.COLOR_RGB2LAB)
            union = masks.any(0)
            matched = {}
            for conf in (0.001, 0.25):
                kept = confidence >= conf
                for label, overlaps in (("mask", iou), ("box", boxes)):
                    ids = greedy_pairs(overlaps[:, kept], 0.5)[:, 0]
                    flags = np.zeros(len(gt), dtype=bool)
                    flags[ids] = True
                    matched[f"{label}_matched_{conf}"] = flags
            kept = confidence >= 0.25
            coverage = intersection[:, kept] / np.maximum(area[:, None], 1e-7)
            purity = intersection[:, kept] / np.maximum(pred_area[None, kept], 1e-7)
            split = ((coverage >= 0.2) & (purity >= 0.5)).sum(1) >= 2
            merge_preds = (coverage >= 0.3).sum(0) >= 2
            merged = (coverage[:, merge_preds] >= 0.3).any(1)
            for i, mask in enumerate(masks):
                points = cv2.findNonZero(mask)
                hull = np.zeros_like(mask)
                if points is not None:
                    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
                outer = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
                ring = outer & ~union
                delta = (
                    float(np.linalg.norm(lab[mask > 0].mean(0) - lab[ring].mean(0))) if area[i] and ring.any() else None
                )
                near = bool((outer & union & ~(mask > 0)).any())
                self.diagnostic.append(
                    dict(
                        image=Path(batch["im_file"]).name,
                        instance_index=i,
                        mask_area_at_input=float(area[i] * sx * sy),
                        solidity=float(area[i] / max(hull.sum(), 1)),
                        nearby_instance=near,
                        lab_mean_delta=delta,
                        split_proxy=bool(split[i]),
                        merge_proxy=bool(merged[i]),
                        **{key: bool(flags[i]) for key, flags in matched.items()},
                    )
                )
            return result

    for name in names:
        run = results / name
        expected = {Path(line).name for line in (run / "val_loaded_files.txt").read_text().splitlines()}
        actual = {p.name for p in (dataset / "val/images").glob("*.jpg")}
        if expected != actual:
            raise ValueError("Local validation filenames differ from server; choose matching local dataset")
        save_dir = output / (name + "_local_eval")
        validator = ChallengeValidator(
            args=dict(
                data=str(local_yaml),
                device="cpu",
                imgsz=640,
                batch=1,
                workers=0,
                plots=False,
                half=False,
                conf=0.001,
                iou=0.7,
                max_det=300,
                rect=False,
                overlap_mask=True,
                mask_ratio=4,
            ),
            save_dir=save_dir,
        )
        model = YOLO(str(run / "weights/best_mask.pt"), verbose=False)
        metrics = validator(model=model.model)
        records = validator.diagnostic
        groups = dict(
            all=records,
            tiny=[r for r in records if r["mask_area_at_input"] < 256],
            small=[r for r in records if 256 <= r["mask_area_at_input"] < 1024],
            larger=[r for r in records if r["mask_area_at_input"] >= 1024],
            concave=[r for r in records if r["solidity"] < 0.9],
            touching=[r for r in records if r["nearby_instance"]],
            low_color_contrast=[r for r in records if r["lab_mean_delta"] is not None and r["lab_mean_delta"] < 10],
        )
        summary = {
            key: dict(
                n=len(values),
                **{
                    metric: sum(r[metric] for r in values) / len(values) if values else None
                    for metric in (
                        "mask_matched_0.001",
                        "mask_matched_0.25",
                        "box_matched_0.001",
                        "box_matched_0.25",
                        "split_proxy",
                        "merge_proxy",
                    )
                },
            )
            for key, values in groups.items()
        }
        payload = dict(
            checkpoint=str(run / "weights/best_mask.pt"),
            metrics=metrics,
            summary=summary,
            instances=records,
            protocol="LOCAL CPU FP32 batch1 rectFalse best_mask; diagnostics stride4, no test data. Filename parity only.",
            limitations="Subset recall, not AP_small; color distance is an appearance proxy, not proof of color dependence.",
        )
        (output / (name + "_diagnostics.json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(dict(name=name, metrics=metrics, summary=summary), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--only",
        default="SAGE30_official_control_seed42,SAGE42_asym_semantic_detail_seed42,SAGE46_asym_geometry_seed42",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit(args.results, args.output)
    if args.validate:
        if args.dataset is None:
            parser.error("--validate requires --dataset")
        validate(args.results, args.dataset, args.output, args.only.split(","))


if __name__ == "__main__":
    main()
