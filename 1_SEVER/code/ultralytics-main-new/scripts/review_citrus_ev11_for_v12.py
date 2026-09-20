"""Read all historical CSV/source mappings and compare returned V11 completed runs."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import review_citrus_ev7_for_v8 as history

OUT = history.ROOT / "docs/E_V12_REVIEW_20260918"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true", help="Reuse this review's completed CSV scan; no new results")
    args = parser.parse_args()
    history.OUT = OUT
    if not args.cached:
        with contextlib.redirect_stdout(io.StringIO()):
            history.main()
    data = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    rows = [r for r in data["history"] if "/E/V11_/" in r["path"].replace("\\", "/")]
    table = []
    pr_evidence = {}
    for r in rows:
        row = dict(
            model=r["name"],
            epochs=r["epochs"],
            completed=bool(r.get("completed")),
            recall=100 * r["best"]["metrics/recall(M)"],
            ap50=100 * r["best"]["metrics/mAP50(M)"],
            ap=100 * r["best"][history.AP],
            last20=100 * r["last20"],
            epoch_s=r["median_epoch_s"],
        )
        for budget in ("coarse", "fine"):
            report = Path(r["path"]).parent / f"paired_{budget}_eval/paired_metrics.json"
            if not report.exists():
                continue
            raw = json.loads(report.read_text(encoding="utf-8"))
            for mode in ("global", "trustedmask"):
                s = raw["summary"][mode]
                c = raw["empirical_mask_pr"][mode]["0"]
                pr_evidence[f"{r['name']}/{budget}/{mode}"] = dict(
                    rmax=c["rmax"],
                    last_real_precision=c["precision"][-1] if c["precision"] else None,
                    errors25=s["errors25"],
                    tiny_n=s["tiny_n"],
                    tiny_matched=s["tiny_matched"],
                    summary=s,
                )
                prefix = "global" if mode == "global" else budget
                row.update(
                    {
                        prefix + "_ap": 100 * s[history.AP],
                        prefix + "_ap50": 100 * s["metrics/mAP50(M)"],
                        prefix + "_tiny": s["tiny_matched"],
                        prefix + "_tiny_n": s["tiny_n"],
                        prefix + "_background": s["errors25"]["background"],
                        prefix + "_rmax": c["rmax"],
                        prefix + "_r90": 100
                        * max((rec for prec, rec in zip(c["precision"], c["recall"]) if prec >= 0.9), default=0),
                        prefix + "_ms": s["median_ms"],
                    }
                )
        table.append(row)
    fields = list(dict.fromkeys(k for row in table for k in row))
    with (OUT / "V11_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    (OUT / "PR_evidence.json").write_text(json.dumps(pr_evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    checks = dict(
        csv_count=len(data["history"]),
        unique=data["unique_csv"],
        errors=data["errors"],
        yaml_mapped=sum(bool(r["sources"]) for r in data["history"]),
        train_lists=len({r.get("train_loaded_files_sha256") for r in rows}),
        val_lists=len({r.get("val_loaded_files_sha256") for r in rows}),
        yaml_hash_match={
            r["name"]: (
                any(s["sha256"] == r["completed"]["yaml_sha256"] for s in r["sources"])
                if r.get("completed", {}).get("yaml_sha256")
                else None
            )
            for r in rows
        },
        args_differences={
            k: [r["args"].get(k) for r in rows]
            for k in rows[0]["args"]
            if len({str(r["args"].get(k)) for r in rows}) > 1
        },
    )
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    for row in table:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
