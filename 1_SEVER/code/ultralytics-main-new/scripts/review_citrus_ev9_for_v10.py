"""Re-read every returned result/YAML and audit V9 without editing past evidence."""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import review_citrus_ev7_for_v8 as history

ROOT = history.ROOT
OUT = ROOT / "docs/E_V10_REVIEW_20260916"
AP = history.AP


def main():
    history.OUT = OUT
    with contextlib.redirect_stdout(io.StringIO()):
        history.main()
    path = OUT / "audit.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["history"] if "/E/E_V9/" in r["path"].replace("\\", "/")]
    data["v9"] = rows
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    differences = {
        k: [r["args"].get(k) for r in rows] for k in rows[0]["args"] if len({str(r["args"].get(k)) for r in rows}) > 1
    }
    checks = dict(
        total_csv=len(data["history"]),
        unique_csv=data["unique_csv"],
        errors=data["errors"],
        mapped=sum(bool(r["sources"]) for r in data["history"]),
        epochs={r["name"]: r["epochs"] for r in rows},
        args_differences=differences,
        train_lists=len({r.get("train_loaded_files_sha256") for r in rows}),
        val_lists=len({r.get("val_loaded_files_sha256") for r in rows}),
        yaml_hash_match={
            r["name"]: any(s["sha256"] == r["completed"]["yaml_sha256"] for s in r["sources"]) for r in rows
        },
    )
    (OUT / "protocol_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    catalog = {layer[2]: [] for r in data["history"] for s in r["sources"] for layer in s["layers"]}
    for p in (ROOT / "ultralytics/nn").rglob("*.py"):
        source = p.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in catalog:
                catalog[node.name].append(
                    dict(
                        path=str(p.relative_to(ROOT)),
                        line=node.lineno,
                        end=node.end_lineno,
                        sha256=hashlib.sha256(source.encode()).hexdigest(),
                    )
                )
    (OUT / "historical_symbol_sources.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    lines = [
        "# V9 结果与 Recall 审计",
        "",
        "单位：%；训练行均为最佳 Mask AP50–95 同轮。单 seed 验证集筛选，不是测试集结论。",
        "训练CSV与统一原图栅格评估不可混用；R90为P≥90%时的Recall，不是固定阈值Recall。",
        "",
        "|模型|训练R|训练AP50|训练AP|末20轮AP|整图AP|细切片AP|细切片R90|tiny检出|背景FP|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pr_rows = []
    for r in rows:
        g = r["paired"]["fine"]["summary"]["global"]
        f = r["paired"]["fine"]["summary"]["trustedmask"]
        b = r["best"]
        lines.append(
            f"|{r['name']}|{b['metrics/recall(M)'] * 100:.3f}|{b['metrics/mAP50(M)'] * 100:.3f}|"
            f"{b[AP] * 100:.3f}|{r['last20'] * 100:.3f}|{g[AP] * 100:.3f}|{f[AP] * 100:.3f}|"
            f"{f['operating_p90']['recall'] * 100:.3f}|{f['tiny_matched']}/{f['tiny_n']}|{f['errors25']['background']}|"
        )
        report = Path(r["path"]).parent / "paired_fine_eval/paired_metrics.json"
        raw = json.loads(report.read_text(encoding="utf-8"))
        for mode in ("global", "trustedmask"):
            curve = raw["empirical_mask_pr"][mode]["0"]
            s = raw["summary"][mode]
            pr_rows.append(
                dict(
                    name=r["name"],
                    mode=mode,
                    targets=curve["targets"],
                    maximum_observed_recall=curve["rmax"],
                    last_precision=curve["precision"][-1],
                    fn25=curve["targets"] - s["errors25"]["tp"],
                    tiny_fn25=s["tiny_n"] - s["tiny_matched"],
                    errors25=s["errors25"],
                    operating_p90=s["operating_p90"],
                )
            )
    lines += [
        "",
        "tiny定义：统一长边640栅格可见掩膜面积<256，IoU=.5/conf=.25；不是COCO APs。",
        "历史YAML与源码符号已全量关联，但当前源码不保证等于历史服务器版本。",
        "背景FP只是自动错误归类，不能据此断言全部为叶片颜色混淆。",
    ]
    (OUT / "V9结果对照.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "pr_bottlenecks.json").write_text(json.dumps(pr_rows, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
