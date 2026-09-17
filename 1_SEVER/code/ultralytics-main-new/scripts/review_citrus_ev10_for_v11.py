"""Re-read returned results and audit V10; never alter past results or AP definitions."""

import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path

import review_citrus_ev7_for_v8 as history

ROOT = history.ROOT
OUT = ROOT / "docs/E_V11_REVIEW_20260917"
AP = history.AP


def main():
    history.OUT = OUT
    with contextlib.redirect_stdout(io.StringIO()):
        history.main()
    data = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    rows = [r for r in data["history"] if "/E/E_V10/" in r["path"].replace("\\", "/")]
    data["v10"] = rows
    (OUT / "audit.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = dict(
        csv_count=len(data["history"]),
        unique_csv=data["unique_csv"],
        read_errors=data["errors"],
        yaml_mapped=sum(bool(r["sources"]) for r in data["history"]),
        epochs={r["name"]: r["epochs"] for r in rows},
        yaml_hash_match={
            r["name"]: any(s["sha256"] == r["completed"]["yaml_sha256"] for s in r["sources"]) for r in rows
        },
        train_lists=len({r.get("train_loaded_files_sha256") for r in rows}),
        val_lists=len({r.get("val_loaded_files_sha256") for r in rows}),
        args_differences={
            k: [r["args"].get(k) for r in rows]
            for k in rows[0]["args"]
            if len({str(r["args"].get(k)) for r in rows}) > 1
        },
        limitation="Single seed validation screening; current source is not proof of identical historical Python.",
    )
    (OUT / "protocol_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    symbols = {layer[2]: [] for r in data["history"] for s in r["sources"] for layer in s["layers"]}
    for p in (ROOT / "ultralytics/nn").rglob("*.py"):
        source = p.read_text(encoding="utf-8-sig")
        for n in ast.walk(ast.parse(source)):
            if isinstance(n, ast.ClassDef) and n.name in symbols:
                symbols[n.name].append(dict(path=str(p.relative_to(ROOT)), line=n.lineno, end=n.end_lineno))
    (OUT / "historical_symbol_sources.json").write_text(json.dumps(symbols, indent=2), encoding="utf-8")
    table, pr = [], []
    for r in rows:
        g = r["paired"]["fine"]["summary"]["global"]
        f = r["paired"]["fine"]["summary"]["trustedmask"]
        row = dict(
            model=r["name"],
            epoch=int(r["best"]["epoch"]),
            train_recall=r["best"]["metrics/recall(M)"] * 100,
            train_ap50=r["best"]["metrics/mAP50(M)"] * 100,
            train_ap=r["best"][AP] * 100,
            last20_ap=r["last20"] * 100,
            median_epoch_s=r["median_epoch_s"],
            global_ap=g[AP] * 100,
            fine_ap=f[AP] * 100,
            global_tiny=g["tiny_matched"],
            fine_tiny=f["tiny_matched"],
            global_bg=g["errors25"]["background"],
            fine_bg=f["errors25"]["background"],
        )
        for tag in ("coarse", "fine"):
            report = Path(r["path"]).parent / f"paired_{tag}_eval/paired_metrics.json"
            raw = json.loads(report.read_text(encoding="utf-8"))
            for mode in ("global", "trustedmask"):
                c = raw["empirical_mask_pr"][mode]["0"]
                eligible = [
                    (rec, conf, prec)
                    for prec, rec, conf in zip(c["precision"], c["recall"], c["confidence"])
                    if prec >= 0.9
                ]
                r90 = max(eligible, default=(0, None, None))
                s = raw["summary"][mode]
                pr.append(
                    dict(
                        model=r["name"],
                        budget=tag,
                        mode=mode,
                        targets=c["targets"],
                        rmax=c["rmax"],
                        last_precision=c["precision"][-1],
                        r90_exact=r90[0],
                        confidence_r90=r90[1],
                        points=len(c["precision"]),
                        fn25=c["targets"] - s["errors25"]["tp"],
                        tiny_fn25=s["tiny_n"] - s["tiny_matched"],
                        errors25=s["errors25"],
                        size_bins25=s["size_bins25"],
                        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                    )
                )
                if tag == "fine":
                    row[f"{mode}_r90"] = r90[0] * 100
        table.append(row)
    with (OUT / "V10指标对照.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (OUT / "pr_bottlenecks.json").write_text(json.dumps(pr, indent=2), encoding="utf-8")
    lines = [
        "# V10 同协议结果对照",
        "",
        "单位%；训练指标取最佳 Mask AP50–95 的同一轮。单 seed 验证集筛选。",
        "R90 为完整经验 PR 中 P≥90% 的最大 Recall；不是置信度0.9。",
        "",
        "|模型|训练R|训练AP50|训练AP|末20轮AP|整图AP|细切片AP|细切片R90|整图tiny|细切片tiny|细切片背景FP|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in table:
        lines.append(
            f"|{t['model']}|{t['train_recall']:.3f}|{t['train_ap50']:.3f}|{t['train_ap']:.3f}|"
            f"{t['last20_ap']:.3f}|{t['global_ap']:.3f}|{t['fine_ap']:.3f}|{t['trustedmask_r90']:.3f}|"
            f"{t['global_tiny']}/161|{t['fine_tiny']}/161|{t['fine_bg']}|"
        )
    lines += [
        "",
        "tiny：统一长边640栅格可见面积<256；匹配IoU=.5/conf=.25，不是COCO APs。",
        "CSV训练评估与统一原图栅格评估不能混用。背景错误不能全部解释成叶片颜色混淆。",
        "历史结果未覆盖或改写；当前YAML与符号索引不等于完整历史服务器源码审计。",
    ]
    (OUT / "V10结果对照.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
