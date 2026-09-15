"""Read-only source/result audit. Outputs are compact evidence, never training inputs."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from scripts.review_citrus_e_for_v2 import paired_recall

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parents[1] / "results"
SOURCE = RESULTS / "E/E_V2/CITRUS_EV2_ALL_300EP"
OUT = ROOT / "docs/E_V3_evidence"
AP = "metrics/mAP50-95(M)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history, runs, errors = [], [], []
    for path in sorted(RESULTS.rglob("results.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [{k.strip(): float(v) for k, v in r.items() if k and v.strip()} for r in csv.DictReader(stream)]
            peak = max(rows, key=lambda x: x[AP])
            args_path = path.parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
            name = path.parent.name.split("_seed")[0]
            yaml_name = Path(str(args.get("model", "")).replace("\\", "/")).name
            candidates = list((ROOT / "0_orange_yaml").rglob(yaml_name)) if yaml_name.endswith(".yaml") else []
            item = dict(name=name, path=str(path), epochs=len(rows), peak=peak,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        args=args, source_candidates=[str(p.relative_to(ROOT)) for p in candidates])
            history.append(item)
            if path.parent.parent != SOURCE:
                continue
            read = lambda f: json.loads((path.parent / f).read_text(encoding="utf-8"))
            metrics = read("paired_sliced_eval/paired_metrics.json")
            init = read("initialization_transfer.json")
            marker = read("completed.json")
            config_path = ROOT / "0_orange_yaml/E_V2_series" / (name + ".yaml")
            run = dict(item, summary=metrics["summary"], records=metrics["records"],
                       init_fraction=init["equal_fraction"], params=init["total_parameter_numel"],
                       loaded=read("loaded_data_summary.json"),
                       yaml=yaml.safe_load(config_path.read_text()),
                       yaml_matches=hashlib.sha256(config_path.read_bytes()).hexdigest() == marker["yaml_sha256"],
                       val_list_sha256=hashlib.sha256((path.parent / "val_loaded_files.txt").read_bytes()).hexdigest(),
                       last20_ap=statistics.mean(x[AP] for x in rows[-20:]),
                       median_epoch_s=statistics.median(b["time"]-a["time"] for a,b in zip(rows,rows[1:])))
            runs.append(run)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=repr(error)))
    assert len(runs) == 8, "Do not silently omit an E V2 result"
    ignore = {"name", "model", "data", "save_dir", "project"}
    for run in runs:
        run["args_diff"] = {k: [runs[0]["args"].get(k),run["args"].get(k)]
                            for k in runs[0]["args"].keys() | run["args"].keys()
                            if k not in ignore and runs[0]["args"].get(k) != run["args"].get(k)}
    comparisons = {}
    for before, after in ((0,1),(0,2),(0,3),(3,4),(4,5),(5,6),(5,7)):
        for mode in ("global", "trustedmask"):
            a, b = [[x for x in runs[i]["records"] if x["mode"] == mode] for i in (before,after)]
            comparisons[f'{runs[after]["name"]}-{runs[before]["name"]}/{mode}'] = paired_recall(a,b)
    for run in runs:
        a,b = [[x for x in run["records"] if x["mode"] == mode] for mode in ("global","trustedmask")]
        comparisons[run["name"]+"/trusted-global"] = paired_recall(a,b)
        del run["records"]  # Raw uploaded records remain intact; do not copy thousands of rows.
    payload = dict(history=history, runs=runs, comparisons=comparisons, errors=errors,
                   unique_csv=len({r["sha256"] for r in history}),
                   missing_protocol=not (SOURCE / "_protocol").exists())
    (OUT / "audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = ["# E V2 实证复核", "", "CSV 在最佳 Mask AP50–95 同一轮读取；单位为百分数。", "",
             "| 模型 | AP50 | AP50–95 | ΔAP50–95 | 后20轮均值 | 秒/轮中位数 | 预训练继承 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        p=r["peak"]
        lines.append(f'| {r["name"]} | {100*p["metrics/mAP50(M)"]:.3f} | {100*p[AP]:.3f} | '
                     f'{100*(p[AP]-runs[0]["peak"][AP]):+.3f} | {100*r["last20_ap"]:.3f} | '
                     f'{r["median_epoch_s"]:.2f} | {100*r["init_fraction"]:.2f}% |')
    lines += ["", "切片独立评估统一在原图长边640掩膜栅格；不能与上表官方 stride4 AP 混比。",
              "tiny=该栅格面积<256；召回统计conf≥0.25、mask IoU≥0.5。", "",
              "| 模型 | 全图tiny检出 | 切片tiny检出 | 切片AP50 | 切片AP50–95 | R@P≥90% | 背景误检 | 重复误检 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        g,s=r["summary"]["global"],r["summary"]["trustedmask"]
        lines.append(f'| {r["name"]} | {g["tiny_matched"]}/161 | {s["tiny_matched"]}/161 | '
                     f'{100*s["metrics/mAP50(M)"]:.3f} | {100*s[AP]:.3f} | '
                     f'{100*s["operating_p90"]["recall"]:.3f} | {s["errors25"]["background"]} | '
                     f'{s["errors25"]["duplicates"]} |')
    lines += ["", f'历史CSV={len(history)}，去除内容重复后={payload["unique_csv"]}，读取错误={len(errors)}。',
              f'V2全部300轮={all(r["epochs"]==300 for r in runs)}；YAML均匹配={all(r["yaml_matches"] for r in runs)}；',
              f'验证文件列表一致={len({r["val_list_sha256"] for r in runs})==1}；',
              f'除路径/名称外训练参数一致={all(not r["args_diff"] for r in runs)}。',
              "", "上传结果未包含 _protocol，无法证明服务器运行源文件与本地代码逐字节相同。",
              "历史路径/AMP/初始化/输入协议不同，不把跨系列最高值排序解释为架构净收益。",
              "每轮时间包含验证且可能受服务器负载影响，不能作为纯算子速度证明。",
              "bootstrap 是图像重采样的不确定性，不是多训练seed置信区间。详见 audit.json。"]
    (OUT / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
