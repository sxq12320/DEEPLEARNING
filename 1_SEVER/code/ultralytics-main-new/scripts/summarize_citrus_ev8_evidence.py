"""Create protocol checks, V8 comparison and unpadded PR diagnostics from returned files."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/E_V9_REVIEW_20260915"
AP = "metrics/mAP50-95(M)"


def main():
    data = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    rows = data["v8"]
    differences = {
        k: [r["args"].get(k) for r in rows] for k in rows[0]["args"] if len({str(r["args"].get(k)) for r in rows}) > 1
    }
    checks = dict(
        epochs=[r["epochs"] for r in rows],
        args_differences=differences,
        unique_train_lists=len({r.get("train_loaded_files_sha256") for r in rows}),
        unique_val_lists=len({r.get("val_loaded_files_sha256") for r in rows}),
        yaml_hash_match={
            r["name"]: any(s["sha256"] == r["completed"]["yaml_sha256"] for s in r["sources"]) for r in rows
        },
    )
    (OUT / "protocol_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    lines = [
        "# V8 结果：统一口径、同轮指标",
        "",
        "单位：百分数；AP列均为Mask。只作单seed验证集筛选。",
        "训练CSV的GT/输入栅格与原图统一640评估不同，不把两张表的绝对AP直接相减。",
        "",
        "|模型|训练AP50|训练AP50–95|末20轮AP|整图AP|细切片AP|整图极小检出|细切片极小检出|细切片背景FP|P≥90%时R|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    detail = []
    for r in rows:
        g = r["paired"]["fine"]["summary"]["global"]
        f = r["paired"]["fine"]["summary"]["trustedmask"]
        lines.append(
            f"|{r['name']}|{r['best']['metrics/mAP50(M)'] * 100:.3f}|{r['best'][AP] * 100:.3f}|"
            f"{r['last20'] * 100:.3f}|{g[AP] * 100:.3f}|{f[AP] * 100:.3f}|{g['tiny_matched']}/161|"
            f"{f['tiny_matched']}/161|{f['errors25']['background']}|{f['operating_p90']['recall'] * 100:.3f}|"
        )
        detail.append(
            dict(
                name=r["name"],
                global_fn=1049 - g["errors25"]["tp"],
                global_tiny_fn=161 - g["tiny_matched"],
                global_rmax=g["mask_rmax_at_conf001"],
                fine_rmax=f["mask_rmax_at_conf001"],
                fine_errors=f["errors25"],
            )
        )
    lines += [
        "",
        "极小目标定义：原图统一长边640评估栅格中可见mask面积<256；IoU=.5/conf=.25。不是COCO APs。",
        "P90R来自阈值扫描，和conf=.25的TP/FP不能混用。时延受共享硬件影响，应同机独占复测。",
        "",
        "## 核查",
        "",
        f"结果CSV {len(data['history'])}份，按内容去重{data['unique_csv']}份；"
        f"全部有当前YAML候选，读取错误{len(data['errors'])}个。",
        "历史Python源码不能仅凭当前YAML反推；旧协议不与新协议直接排名。",
        f"V8训练文件清单哈希种类={checks['unique_train_lists']}，验证清单={checks['unique_val_lists']}。",
        f"V8已回传YAML哈希全部匹配当前配置：{all(checks['yaml_hash_match'].values())}。",
        f"V8实际参数差异键：{', '.join(differences)}。",
        "",
    ]
    (OUT / "V8结果对照.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "pr_bottlenecks.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    # Map every historical YAML symbol to current source definitions, without
    # asserting that current code is a byte-identical historical server snapshot.
    needed = {layer[2] for r in data["history"] for s in r["sources"] for layer in s["layers"]}
    catalog = {name: [] for name in sorted(needed)}
    for path in (ROOT / "ultralytics/nn").rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in catalog:
                catalog[node.name].append(
                    dict(
                        path=str(path.relative_to(ROOT)),
                        line=node.lineno,
                        end=node.end_lineno,
                        sha256=hashlib.sha256(text.encode()).hexdigest(),
                    )
                )
    (OUT / "historical_symbol_sources.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    selected = [rows[0], rows[3], rows[5]]
    for ax, mode in zip(axes, ("global", "trustedmask")):
        for r, color in zip(selected, ("#0072B2", "#D55E00", "#777777")):
            path = Path(r["path"]).parent / "paired_fine_eval/paired_metrics.json"
            raw = json.loads(path.read_text(encoding="utf-8"))["empirical_mask_pr"][mode]["0"]
            ax.plot(
                raw["recall"],
                raw["precision"],
                color=color,
                lw=1.1,
                label=r["name"].split("_seed")[0].replace("V8_", ""),
            )
            ax.plot(raw["recall"][-1], raw["precision"][-1], "o", ms=3, color=color)
        ax.set(
            xlim=(0.65, 1),
            ylim=(0, 1.02),
            xlabel="Recall (not confidence)",
            ylabel="Precision",
            title="Single image" if mode == "global" else "Fine tiles + trusted-mask merge",
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Observed PR points only: no envelope and no padded zero endpoint", fontsize=10)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"V8_empirical_PR.{ext}", dpi=300)
    plt.close(fig)
    print(json.dumps(checks, ensure_ascii=False))
    print(json.dumps(detail, ensure_ascii=False))


if __name__ == "__main__":
    main()
