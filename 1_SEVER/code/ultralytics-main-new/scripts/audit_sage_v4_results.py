"""Reproducible CSV/checkpoint audit; never edits experimental inputs or datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml


def rows_from(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(handle)]


def audit_run(path):
    rows = rows_from(path)
    if not rows or "metrics/mAP50-95(M)" not in rows[0]:
        return None
    best = max(rows, key=lambda row: row["metrics/mAP50-95(M)"])
    args_path = path.parent / "args.yaml"
    args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
    durations = np.diff([0, *[row.get("time", 0) for row in rows]])
    result = {
        "name": path.parent.name,
        "path": str(path),
        "epochs_observed": len(rows),
        "mask_peak_row": best,
        "last_mask_ap": rows[-1]["metrics/mAP50-95(M)"],
        "max_mask_ap50": max(r["metrics/mAP50(M)"] for r in rows),
        "tail20_mask_ap_mean": np.mean([r["metrics/mAP50-95(M)"] for r in rows[-20:]]).item(),
        "time_hours": durations.sum().item() / 3600,
        "median_epoch_seconds": float(np.median(durations)),
        "p95_epoch_seconds": float(np.quantile(durations, 0.95)),
        "max_epoch_seconds": float(durations.max()),
        "args": args,
    }
    if "SAGE" in path.parts and (path.parent / "weights" / "best.pt").exists():
        try:
            checkpoint = torch.load(path.parent / "weights" / "best.pt", map_location="cpu", weights_only=False)
            result["official_best_checkpoint_metrics"] = checkpoint.get("train_metrics", {})
        except Exception as error:
            result["checkpoint_read_error"] = repr(error)
            return result
        target = result["official_best_checkpoint_metrics"].get("metrics/mAP50-95(M)")
        result["official_best_candidate_epochs"] = [
            r["epoch"] for r in rows if target is not None and abs(r["metrics/mAP50-95(M)"] - target) < 1e-6
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audited, errors = [], []
    for path in sorted(args.results.rglob("results.csv")):
        try:
            result = audit_run(path)
            if result:
                audited.append(result)
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"scope": str(args.results), "mask_runs": len(audited), "errors": errors, "runs": audited}
    (args.output / "all_results_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sage = [r for r in audited if "SAGE" in Path(r["path"]).parts]
    lines = [
        "# SAGE 结果核对表（自动生成）",
        "",
        "数值为百分数；峰值行和官方 best.pt 不能混写。尾20均值是跨epoch描述，不是跨seed统计。",
        "",
        "| 模型 | 轮数 | Mask AP50-95峰值 | 峰值epoch | 同epoch AP50 | 官方best AP50-95 | 官方best AP50 | 尾20均值 | epoch中位秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sage:
        best, checkpoint = result["mask_peak_row"], result.get("official_best_checkpoint_metrics", {})
        lines.append(
            f"| {result['name']} | {result['epochs_observed']} | {best['metrics/mAP50-95(M)'] * 100:.3f} | "
            f"{best['epoch']:.0f} | {best['metrics/mAP50(M)'] * 100:.3f} | "
            f"{checkpoint.get('metrics/mAP50-95(M)', float('nan')) * 100:.3f} | "
            f"{checkpoint.get('metrics/mAP50(M)', float('nan')) * 100:.3f} | "
            f"{result['tail20_mask_ap_mean'] * 100:.3f} | {result['median_epoch_seconds']:.2f} |"
        )
    lines += [
        "",
        f"全目录读取到 {len(audited)} 个含Mask指标的CSV；读取失败 {len(errors)} 个。",
        "",
        "完整模型、训练参数、路径和逐运行统计见同目录 all_results_inventory.json。",
        "",
    ]
    (args.output / "SAGE_METRICS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
