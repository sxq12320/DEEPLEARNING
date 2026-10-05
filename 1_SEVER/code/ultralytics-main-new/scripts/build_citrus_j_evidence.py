"""Extract bounded, protocol-aware historical evidence; never modify original runs."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows, paired = [], []
    seen = set()
    for run in inventory["runs"]:
        if run["csv_hash"] in seen:
            continue
        seen.add(run["csv_hash"])
        peak, cfg = run["peak"], run["args"]
        signature = {
            k: cfg.get(k)
            for k in (
                "amp",
                "imgsz",
                "batch",
                "optimizer",
                "lr0",
                "lrf",
                "mask_ratio",
                "copy_paste",
                "seed",
                "pretrained",
            )
        }
        loaded = Path(run["path"]).parent / "val_loaded_files.txt"
        members = []
        if loaded.is_file():
            members = sorted(Path(s.replace("\\", "/")).name for s in loaded.read_text().splitlines() if s.strip())
        member_hash = hashlib.sha256("\n".join(members).encode()).hexdigest() if members else "unknown"
        row = dict(
            series=run["series"],
            name=run["name"],
            epochs=run["epochs_observed"],
            peak_epoch=peak["epoch"],
            mask_ap=peak["metrics/mAP50-95(M)"],
            mask_ap50=peak["metrics/mAP50(M)"],
            mask_recall=peak["metrics/recall(M)"],
            mask_precision=peak["metrics/precision(M)"],
            tail20_ap=run["tail20_mean"],
            **signature,
            val_members_hash=member_hash,
            data=cfg.get("data"),
            yaml=" | ".join(run["current_yaml_candidates"]),
            source_status="current basename match; not historical source revision proof",
            path=run["path"],
        )
        rows.append(row)
        for mode in ("fine", "coarse"):
            path = Path(run["path"]).parent / ("paired_" + mode + "_eval") / "paired_metrics.json"
            if path.is_file():
                content = json.loads(path.read_text())
                summary = content.get("summary", {})
                compact = {}
                for method, values in summary.items():
                    if isinstance(values, dict):
                        compact[method] = {
                            k: v
                            for k, v in values.items()
                            if k.startswith("metrics/")
                            or k in {"tiny_n", "tiny_matched", "tiny_recall25", "size_bins25", "speed", "median_ms"}
                        }
                paired.append(
                    dict(
                        name=run["name"],
                        mode=mode,
                        path=str(path),
                        summary=compact,
                        note="paired inference on same weights; not retraining evidence",
                    )
                )
    with (args.output / "unique_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        csv_files=len(inventory["runs"]),
        unique_runs=len(rows),
        series_counts=dict(Counter(r["series"] for r in rows)),
        historical_yaml_graphs=len(inventory["configs"]),
        errors=inventory["errors"],
        paired=paired,
    )
    (args.output / "bounded_evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "paired"}, ensure_ascii=False))
    for series in sorted({r["series"] for r in rows}):
        top = sorted([r for r in rows if r["series"] == series], key=lambda r: r["mask_ap"], reverse=True)[:2]
        print(series, [(r["name"], round(r["mask_ap"], 5), round(r["mask_ap50"], 5), r["epochs"]) for r in top])
    for item in paired:
        if "I20" in item["name"] or "V12_04" in item["name"]:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
