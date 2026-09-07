"""Derive a V5 result table and diagnostic summary without modifying uploaded results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/sage_v6_20260905"


def main():
    raw = json.loads((OUT / "v5_audit/v4r_audit.json").read_text(encoding="utf-8"))
    rows = []
    for run in raw["runs"]:
        p = run["peak"]
        rows.append(dict(
            model=run["name"], epochs=run["epochs"], peak_epoch=p["epoch"],
            mask_ap=100 * p["metrics/mAP50-95(M)"], mask_ap50=100 * p["metrics/mAP50(M)"],
            box_ap=100 * p["metrics/mAP50-95(B)"], precision=100 * p["metrics/precision(M)"],
            recall=100 * p["metrics/recall(M)"], tail20_mask_ap=100 * run["tail20_ap"],
            median_epoch_seconds=run["epoch_median_seconds"], path=run["path"],
        ))
    with (OUT / "V5_RESULTS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_number = {int(r["model"][4:6]): r for r in rows}
    a = {n: r["mask_ap"] for n, r in by_number.items()}
    effects = dict(late_proto_without_relay=a[50] - a[42], late_proto_with_relay=a[52] - a[51],
                   relay_without_late_proto=a[51] - a[42], relay_with_late_proto=a[52] - a[50])
    diagnostics = []
    for path in sorted((OUT / "v5_audit").glob("*_diagnostics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        diagnostics.append(dict(name=path.stem, summary=data["summary"], protocol=data["protocol"]))
    (OUT / "v5_effects.json").write_text(
        json.dumps(dict(single_seed_peak_differences_pp=effects, diagnostics=diagnostics), indent=2), encoding="utf-8"
    )
    lines = ["# V5完整300轮结果与V6取舍", "", "所有AP单位为%，AP50和框AP取最佳严格Mask AP同一轮。单seed42，不能称统计显著。", "",
             "| 模型 | 峰值轮 | Mask AP50–95 | 同轮AP50 | 同轮Box AP50–95 | 尾20 Mask AP | 每轮秒中位数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['peak_epoch']:.0f} | {r['mask_ap']:.3f} | {r['mask_ap50']:.3f} | "
                     f"{r['box_ap']:.3f} | {r['tail20_mask_ap']:.3f} | {r['median_epoch_seconds']:.2f} |")
    lines += ["", "## 配方2×2消融", ""]
    lines += [f"- {key}: {value:+.3f}个百分点。" for key, value in effects.items()]
    lines += ["", "## 统一LOCAL诊断（不替代服务器CSV）", "",
              "| 权重 | tiny n | tiny R@.001 | tiny R@.25 | 全体分裂代理 | 全体合并代理 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in diagnostics:
        s = row["summary"]
        lines.append(f"| {row['name']} | {s['tiny']['n']} | {100*s['tiny']['mask_matched_0.001']:.2f}% | "
                     f"{100*s['tiny']['mask_matched_0.25']:.2f}% | {100*s['all']['split_proxy']:.2f}% | "
                     f"{100*s['all']['merge_proxy']:.2f}% |")
    lines += ["", "tiny为640输入stride4掩膜面积<256的分组；这里是Recall，不是AP_small。身份错误是需人工复核的代理。",
              "", "结论：撤回默认late_proto；relay只有很弱的严格AP收益，未证明救回极小果。保留原型高分辨率细化，",
              "把主干和颈部的预算重新分配作为下一轮结构假设。SAGE52的219秒级每轮耗时不应直接解释为该架构固有速度。",
              "", "同批保存训练参数一致，676训练图/193验证图/1049验证实例。源码记录仅RUN_SAGE_V5.py与本地不一致；",
              "本地入口还为screen/50，而实际args证实all/300。实现核心源文件在新增V6代码前已核对一致。",
              "", "本次未重新进行全历史评估；历史对照边界沿用09-05重审报告，不与不同数据/AMP记录混排。"]
    (OUT / "V5_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(effects, indent=2))
    print("\n".join(lines[-15:]))


if __name__ == "__main__":
    main()
