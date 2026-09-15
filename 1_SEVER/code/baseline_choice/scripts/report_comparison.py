"""Aggregate every COCO mask evaluation into the paper-1 comparison table.

Scans <evaluation>/<run>_<split>/metrics.json, maps run names back to baselines
(new unified names from run_comparison_batch.py plus legacy launcher names),
aggregates mean±std across seeds, and writes comparison_table.csv/.md.

Example:
    python scripts/report_comparison.py --evaluation runs/evaluation --registry configs/baselines.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_common import load_registry  # noqa: E402

METRIC_KEYS = (
    "mask_ap_50_95",
    "mask_ap_50",
    "mask_ap_75",
    "mask_ap_small",
    "mask_ap_medium",
    "mask_ap_large",
    "mask_precision",
    "mask_recall",
    "mask_f1",
    "box_ap_50_95",
)
LATENCY_KEY = "latency_ms_per_image_end_to_end"

TIER_ORDER = {"primary": 0, "core": 1, "journal": 2, "auxiliary": 3, "optional": 4, "optional_accuracy_reference": 4}
TIER_LABEL = {
    "primary": "核心(主基线)",
    "core": "核心",
    "journal": "期刊增强",
    "auxiliary": "辅助",
    "optional": "可选",
    "optional_accuracy_reference": "可选",
}

# Legacy run-name fragments from the per-family launchers -> baseline ID.
LEGACY_ALIASES = {
    "001_6_maskrcnn_r50_fpn": "mask_rcnn_r50_torchvision",
    "MRCNN_R50_FPN": "mask_rcnn_r50_torchvision",
    "E_unet_r18_watershed": "unet",
    "UNET_R18_WATERSHED": "unet",
}

FOOTNOTES = (
    "注1：RF-DETR 使用官方包暴露的 Seg Preview @312 分辨率，延迟不与 640 输入模型直接比较。",
    "注2：U-Net 行为语义分割经 marker-controlled watershed 转实例后的 Mask AP，表中记为 U-Net + Watershed。",
    "注3：多种子单元格为 均值±标准差；单种子为单次结果。全部数值来自同一份 COCO mask 评估器。",
)


def parse_args() -> argparse.Namespace:
    """Parse report options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="Evaluation root with <run>_<split> dirs.")
    parser.add_argument("--registry", type=Path, default=None, help="configs/baselines.yaml path.")
    parser.add_argument("--split", default="test", help="Evaluation split suffix to collect (default test).")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def match_baseline(dir_name: str, baseline_ids: list[str]) -> str | None:
    """Map one evaluation directory name to a registry baseline ID."""
    for fragment, baseline in LEGACY_ALIASES.items():
        if fragment in dir_name:
            return baseline
    # New unified names: <prefix>_<family>_<baseline>_seed<N>_<split>.
    match = re.match(r"^[A-Za-z0-9]+_(?P<family>yolo|mmdet|torchvision|rfdetr|unet)_(?P<rest>.+)$", dir_name)
    if match:
        family = match.group("family")
        rest = match.group("rest")
        candidates = [b for b in baseline_ids if rest.startswith(b)]
        if candidates:
            best = max(candidates, key=len)
            # mmdet Mask R-CNN and torchvision Mask R-CNN share a prefix; the family token disambiguates.
            if best == "mask_rcnn_r50" and family == "torchvision":
                return "mask_rcnn_r50_torchvision"
            return best
    for baseline in sorted(baseline_ids, key=len, reverse=True):
        if baseline in dir_name:
            return baseline
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample std); std is 0 for a single value."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def format_percent(values: list[float]) -> str:
    """Format AP-like fractions as percentage mean or mean±std."""
    if not values:
        return "—"
    mean, std = mean_std(values)
    if std > 0:
        return f"{mean * 100:.2f}±{std * 100:.2f}"
    return f"{mean * 100:.2f}"


def collect(evaluation: Path, split: str, baseline_ids: list[str]) -> dict[str, dict]:
    """Collect per-baseline metric series from all matching evaluation directories."""
    collected: dict[str, dict] = {}
    if not evaluation.is_dir():
        return collected
    for directory in sorted(evaluation.iterdir()):
        metrics_path = directory / "metrics.json"
        if not directory.is_dir() or not metrics_path.is_file() or not directory.name.endswith(f"_{split}"):
            continue
        baseline = match_baseline(directory.name, baseline_ids)
        if baseline is None:
            continue
        seed_match = re.search(r"seed(\d+)", directory.name)
        seed = int(seed_match.group(1)) if seed_match else 42
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        bucket = collected.setdefault(baseline, {"seeds": [], "runs": [], "metrics": {}, "latency": []})
        bucket["seeds"].append(seed)
        bucket["runs"].append(directory.name)
        for key in METRIC_KEYS:
            if key in metrics:
                bucket["metrics"].setdefault(key, []).append(float(metrics[key]))
        if LATENCY_KEY in metrics:
            bucket["latency"].append(float(metrics[LATENCY_KEY]))
    return collected


def build_rows(collected: dict[str, dict], registry: dict) -> list[dict]:
    """Build ordered table rows with registry display names and tiers."""
    baselines = registry.get("baselines", {})
    rows = []
    for baseline, bucket in collected.items():
        info = baselines.get(baseline, {})
        tier = info.get("tier", "optional")
        display = info.get("display_name", baseline)
        if baseline == "unet":
            display = "U-Net + Watershed"
        row = {
            "baseline": baseline,
            "display": display,
            "family": info.get("family", "?"),
            "tier": tier,
            "tier_label": TIER_LABEL.get(tier, tier),
            "seeds": ",".join(str(s) for s in sorted(set(bucket["seeds"]))),
            "n_runs": len(bucket["runs"]),
        }
        for key in METRIC_KEYS:
            row[key] = format_percent(bucket["metrics"].get(key, []))
        row["primary_sort"] = -float(bucket["metrics"].get("mask_ap_50_95", [0.0])[0])
        if bucket["latency"]:
            mean, std = mean_std(bucket["latency"])
            row["latency"] = f"{mean:.2f}±{std:.2f}" if std > 0 else f"{mean:.2f}"
        else:
            row["latency"] = "—"
        rows.append(row)
    rows.sort(key=lambda r: (TIER_ORDER.get(r["tier"], 9), r["primary_sort"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the machine-readable comparison table."""
    header = ["display", "baseline", "family", "tier_label", "seeds", *METRIC_KEYS, "latency"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[key] for key in header])


def write_markdown(path: Path, rows: list[dict], split: str) -> None:
    """Write the paper-ready comparison table with protocol footnotes."""
    columns = [
        ("模型", "display"), ("族", "family"), ("层级", "tier_label"), ("种子", "seeds"),
        ("mAP50-95", "mask_ap_50_95"), ("mAP50", "mask_ap_50"), ("mAP75", "mask_ap_75"),
        ("APs", "mask_ap_small"), ("APm", "mask_ap_medium"), ("APl", "mask_ap_large"),
        ("P", "mask_precision"), ("R", "mask_recall"), ("F1", "mask_f1"), ("延迟/ms", "latency"),
    ]
    lines = [f"# 柑橘幼果实例分割跨范式基线对比表（{split} 集，COCO mask 指标，单位 %）", ""]
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in rows:
        lines.append("| " + " | ".join(row[key] for _, key in columns) + " |")
    lines.extend(["", *[f"- {note}" for note in FOOTNOTES], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Aggregate and write both table formats."""
    args = parse_args()
    registry = load_registry(args.registry) if args.registry else load_registry()
    baseline_ids = list(registry.get("baselines", {}))
    collected = collect(args.evaluation, args.split, baseline_ids)
    if not collected:
        print(f"No metrics.json found under {args.evaluation} for split '{args.split}'.")
        return
    rows = build_rows(collected, registry)
    output_csv = args.output_csv or args.evaluation / f"comparison_table_{args.split}.csv"
    output_md = args.output_md or args.evaluation / f"comparison_table_{args.split}.md"
    write_csv(output_csv, rows)
    write_markdown(output_md, rows, args.split)
    print(f"Baselines aggregated: {len(rows)}")
    for row in rows:
        print(
            f"  {row['display']:<28} [{row['tier_label']:<8}] seeds={row['seeds']:<14} "
            f"mAP50-95={row['mask_ap_50_95']}  mAP50={row['mask_ap_50']}"
        )
    print(f"CSV: {output_csv}\nMD : {output_md}")


if __name__ == "__main__":
    main()
