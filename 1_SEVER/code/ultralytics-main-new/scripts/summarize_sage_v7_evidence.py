"""Render measured V6 findings and V7 costs from stored audit/benchmark JSON."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/sage_v7_20260906"


def main():
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    lines = [
        "# V6结果与V7实验依据",
        "",
        "2026-09-06。所有服务器AP为百分数；AP50取严格Mask AP最佳同一轮。",
        "",
        "| 模型 | 已有轮数 | Mask AP50–95 | 同轮AP50 | 前118轮最佳严格AP | 尾20严格AP | 每轮秒中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in audit["v6"]:
        with Path(run["path"]).open(encoding="utf-8-sig") as f:
            rows = [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(f)]
        early = max(r["metrics/mAP50-95(M)"] for r in rows if r["epoch"] <= 118)
        lines.append(
            f"| {run['name']} | {run['rows']} | {100 * run['peak']['metrics/mAP50-95(M)']:.3f} | "
            f"{100 * run['peak']['metrics/mAP50(M)']:.3f} | {100 * early:.3f} | {100 * run['tail20']:.3f} | "
            f"{run['seconds']:.2f} |"
        )
    lines += [
        "",
        "SAGE62只有118轮且无completed标记，不是完成的300轮结果；其约219.7秒每轮不是受控测速。",
        "其余四组300轮并有完成标记；组内保存参数除模型和输出标识外一致。",
        "新增V7前，V6记录的实现源码及初始化权重hash全部与本地相同。",
        "",
        f"历史覆盖：{len(audit['history'])}份CSV，{len({r['csv_sha256'] for r in audit['history']})}份不同内容，"
        f"{len(audit['yaml_index'])}份改动前YAML配置。副本不算重复实验。详细对应见history_mapping。",
        "读取范围是全CSV/配置盘点与关键模块实现复核，不声称逐行读完全部依赖；历史对应缺源码快照时仅为候选。",
        "",
        "## 本地统一诊断（不替换服务器指标）",
        "",
        "| 权重 | 极小实例数 | Mask R@.001 | Mask R@.25 | 较大组R@.25 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for p in sorted((OUT / "diagnostics").glob("*_diagnostics.json")):
        d = json.loads(p.read_text())["summary"]
        lines.append(
            f"| {p.stem} | {d['tiny']['n']} | {100 * d['tiny']['mask_matched_0.001']:.2f}% | "
            f"{100 * d['tiny']['mask_matched_0.25']:.2f}% | {100 * d['larger']['mask_matched_0.25']:.2f}% |"
        )
    lines += [
        "",
        "极小按640输入stride4栅格面积<256分组，是Recall而非COCO AP_small。",
        "CPU FP32 batch1、best_mask、rectFalse；文件名核对不等于跨机器图像字节已核验。",
        "SAGE60本地严格Mask AP约69.36%，高于服务器CSV的67.503%；此差异尚未归因，禁止用本地值替换或混排。",
        "",
        "## SAGE60逐GT预测阶段诊断",
        "",
    ]
    stages = json.loads((OUT / "candidate_stages60.json").read_text())["summary"]
    for group, values in stages.items():
        lines.append(f"- {group}，n={values['n']}：{values['buckets']}")
    lines += [
        "",
        "固定conf=.25；先按框贪心一对一配对，再检查同一预测ID的mask。",
        "NMS桶同时包含max_det与GT竞争，不是纯NMS因果分析；未分解实际训练TAL正样本质量。",
        "极小实例：85个原始框存在但分数低、42个原始框IoU不足、3个匹配框的掩膜失败、23个成功。",
        "这支持优先测试高分辨率候选表示/分类定位，但不证明P2必然有效，也不证明掩膜问题不存在。",
    ]
    if (OUT / "cpu_benchmark640.json").exists():
        bench = json.loads((OUT / "cpu_benchmark640.json").read_text())
        lines += [
            "",
            "## V7本地实测成本",
            "",
            "| 模型 | 参数M | GFLOPs640 | 前向ms | 前向+损失+反向ms |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for r in sorted(bench["results"], key=lambda r: r["model"]):
            lines.append(
                f"| {r['model']} | {r['params'] / 1e6:.3f} | {r['gflops640']:.3f} | "
                f"{r['forward_median_ms']:.2f} | {r['forward_loss_backward_median_ms']:.2f} |"
            )
        lines += [
            "",
            "CPU、FP32、batch1、640，20次合成微测；不含数据加载/优化器/NMS/验证，不是服务器FPS。",
            "P2候选数从8400升至34000，TAL/分类与NMS成本仍须同GPU实测，低GFLOPs不保证低训练时间。",
        ]
    if (OUT / "cpu_interleaved640.json").exists():
        final = json.loads((OUT / "cpu_interleaved640.json").read_text())
        lines += ["", "### 控制顺序波动后的交错测速（优先采用）", "",
                  "| 模型 | 前向ms | 前向+损失+反向ms |", "| --- | ---: | ---: |"]
        for name, row in final["medians"].items():
            lines.append(f"| {name} | {row['forward']:.2f} | {row['train']:.2f} |")
        lines += ["", "每轮随机模型顺序，3次预热+20次测量；CPU batch1 FP32 640。",
                  "71比70训练步约少15.4%，72/74与70接近。首次按模型连续测量的差异较大，不能据此断言模块速度。",
                  "无GPU速度或正式精度保证；测试环境Python3.9.13/Torch2.8.0 CPU。"]
    (OUT / "RESULTS_AND_DECISIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
