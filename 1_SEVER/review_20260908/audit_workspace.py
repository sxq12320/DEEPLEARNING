"""Read saved experimental evidence without loading or changing checkpoints."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MASK_AP = "metrics/mAP50-95(M)"


def main():
    records, errors = [], []
    for path in sorted((ROOT / "results").rglob("results.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
            valid = [r for r in rows if MASK_AP in r and math.isfinite(float(r[MASK_AP]))]
            if not valid:
                raise ValueError("No finite mask AP rows")
            best = max(valid, key=lambda r: float(r[MASK_AP]))
            args_path = path.with_name("args.yaml")
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
            records.append({
                "path": path.relative_to(ROOT).as_posix(),
                "name": path.parent.name,
                "family": path.relative_to(ROOT / "results").parts[0],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(rows),
                "last_epoch": float(valid[-1]["epoch"]),
                "best_epoch": float(best["epoch"]),
                "best_metrics": {k: float(v) for k, v in best.items() if k.startswith("metrics/")},
                "tail20_mask_ap": statistics.mean(float(r[MASK_AP]) for r in valid[-20:]),
                "best_mask_exists": (path.parent / "weights" / "best_mask.pt").exists(),
                "best_exists": (path.parent / "weights" / "best.pt").exists(),
                "args": args,
            })
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    duplicates = defaultdict(list)
    for record in records:
        duplicates[record["sha256"]].append(record["path"])
    representatives = [next(r for r in records if r["sha256"] == sha) for sha in duplicates]
    sage_v8 = [r for r in records if "/CITRUS_SAGE_V8_ALL_300EP/" in r["path"]]
    excluded = {"model", "name", "project", "save_dir"}
    arg_keys = set().union(*(r["args"] for r in sage_v8))
    differences = {
        key: {r["name"]: r["args"].get(key) for r in sage_v8}
        for key in sorted(arg_keys - excluded)
        if len({json.dumps(r["args"].get(key), sort_keys=True) for r in sage_v8}) > 1
    }
    summary = {
        "csv_count": len(records) + len(errors),
        "parsed_csv": len(records),
        "unique_csv_content": len(duplicates),
        "at_least_300_epochs_unique": sum(r["last_epoch"] >= 300 for r in representatives),
        "under_300_epochs_unique": sum(r["last_epoch"] < 300 for r in representatives),
        "family_csv_counts": dict(Counter(r["family"] for r in records)),
        "seeds_unique_content": dict(Counter(str(r["args"].get("seed")) for r in representatives)),
        "dataset_paths_unique_content": dict(Counter(str(r["args"].get("data")) for r in representatives)),
        "v8_non_identity_arg_differences": differences,
        "duplicate_content_groups": [v for v in duplicates.values() if len(v) > 1],
        "errors": errors,
    }
    (OUT / "results_audit.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# 原始 results.csv 核对表", "", "仅为验证集同轮峰值索引；跨协议不可按此表归因。",
             "", "| 路径 | 记录轮数 | 最佳轮 | Mask AP50–95 (%) | 同轮 AP50 (%) | 尾20 AP均值 (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['path']} | {r['last_epoch']:g} | {r['best_epoch']:g} | "
                     f"{r['best_metrics'][MASK_AP]*100:.3f} | "
                     f"{r['best_metrics']['metrics/mAP50(M)']*100:.3f} | {r['tail20_mask_ap']*100:.3f} |")
    (OUT / "原始结果核对表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in records:
        if r["family"] == "SAGE" or r["name"].startswith(("T00", "T01", "T04", "G00")):
            print(r["name"], r["last_epoch"], r["best_epoch"], r["best_metrics"], r["tail20_mask_ap"])


if __name__ == "__main__":
    main()
