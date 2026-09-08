"""Derived evidence tables and conservative historical source mapping; no result edits."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/sage_v8_20260907"


def main():
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    mapping = []
    for run in audit["history"]:
        original = str(run["args"].get("model", ""))
        filename = original.replace("\\", "/").split("/")[-1]
        candidates = [p for p in audit["yaml_index"] if Path(p).name == filename]
        mapping.append(dict(
            result=run["path"], saved_model=original, local_yaml_candidates=candidates,
            status="filename candidate only; not historical source hash proof" if candidates else "unresolved",
        ))
    (OUT / "history_source_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# V7R 实测结果与诊断", "", "所有 AP 均为百分数；服务器列按最高 Mask AP50–95 选择同一行，不拼接各指标峰值。", "",
             "| 模型 | 轮数/最佳轮 | Mask AP50–95 | 同轮 AP50 | 尾20轮均值 | 每轮中位秒 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in audit["v7r"]:
        p = r["peak"]
        lines.append(f"| {r['name']} | {r['rows']}/{int(p['epoch'])} | {p['metrics/mAP50-95(M)']*100:.3f} | "
                     f"{p['metrics/mAP50(M)']*100:.3f} | {r['tail20']*100:.3f} | {r['seconds']:.2f} |")
    lines += ["", "## 同一验证集的本地诊断", "",
              "CPU FP32、batch1、best_mask.pt、193张/1049实例；输入640，tiny为stride4栅格掩膜面积<256。",
              "这是固定阈值的子集召回，不是COCO AP_small；服务器与本地数值有差异，不能拼接为一个排名。", "",
              "| 模型 | tiny R@.25 | tiny R@.001 | larger R@.25 | 低颜色对比 R@.25 |",
              "|---|---:|---:|---:|---:|"]
    colour = {}
    for path in sorted((OUT / "diagnostics").glob("*_diagnostics.json")):
        d = json.loads(path.read_text())
        s = d["summary"]
        lines.append(f"| {path.stem.split('_')[0]} | {100*s['tiny']['mask_matched_0.25']:.2f} | "
                     f"{100*s['tiny']['mask_matched_0.001']:.2f} | {100*s['larger']['mask_matched_0.25']:.2f} | "
                     f"{100*s['low_color_contrast']['mask_matched_0.25']:.2f} |")
        groups = {}
        for lo, hi, size in ((0, 256, "tiny"), (256, 1024, "small"), (1024, float("inf"), "larger")):
            for low in (True, False):
                rows = [r for r in d["instances"] if lo <= r["mask_area_at_input"] < hi
                        and r["lab_mean_delta"] is not None and (r["lab_mean_delta"] < 10) == low]
                groups[f"{size}_{'low' if low else 'higher'}_contrast"] = dict(
                    n=len(rows), recall25=sum(r["mask_matched_0.25"] for r in rows) / len(rows) if rows else None)
        colour[path.stem] = groups
    (OUT / "size_conditioned_colour.json").write_text(json.dumps(colour, indent=2), encoding="utf-8")
    lines += ["", "## PR 末端与候选失败分解", "",
              "| 模型 | 实测最大R | 末个实测P（非零） | tiny无合格原始框 | tiny合格框但分数<.25 | tiny成功 |",
              "|---|---:|---:|---:|---:|---:|"]
    for folder in ("pr70", "pr76"):
        d = json.loads((OUT / folder / "diagnostic.json").read_text())
        p, b = d["empirical_pr"], d["summary"]["tiny"]["buckets"]
        lines.append(f"| {folder} | {100*p['maximum_recall']:.2f} | {100*p['final_precision']:.2f} | "
                     f"{b.get('no_raw_box_iou50',0)} | {b.get('raw_box_present_score_low',0)} | {b.get('success',0)} |")
    resolved = sum(bool(r["local_yaml_candidates"]) for r in mapping)
    lines += ["", "## 审计覆盖范围与限制", "",
              f"扫描 {len(audit['history'])} 个历史CSV，内容去重后 {len(set(r['csv_sha256'] for r in audit['history']))} 个；"
              f"读取 {len(audit['yaml_index'])} 个历史YAML；CSV解析错误 {len(audit['errors'])}。",
              f"按保存的YAML文件名找到本地候选 {resolved}/{len(mapping)} 个。未恢复项见 history_source_mapping.json，"
              "不能把文件名匹配当作历史源码完全相同。",
              "本次改动前 V7R 记录的全部实现/协议/预训练文件 SHA 均与本地相同；该快照保存在 audit.json。"
              "此后新增V8注册会使当前tasks/__init__/foreground的SHA改变，这是新系列修改而非原实验错误。",
              "V7R同轮运行的训练参数差异为空；本地验证只核对文件名集合，不能证明服务器图像字节完全相同。",
              "重点源码复核包括G10的混合主干/CARAFE/BiFPN与P2原型头、V4R语义细节、V5双路原型、"
              "V6交换主干、V7压缩预测头、V7R上下文路由。没有宣称逐行阅读所有历史文件。",
              "按尺度/颜色的诊断利用验证GT，仅用于分析，未进入推理。训练未删除超小目标或修改标注。",
              ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote RESULTS.md; historical filename candidates {resolved}/{len(mapping)}")


if __name__ == "__main__":
    main()
