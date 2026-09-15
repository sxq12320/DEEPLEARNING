"""Read all historical results and the received V5 arms; do not infer missing runs."""
# ruff: noqa: E402
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v5_suite import NAMES, FACTORS

RESULTS = ROOT.parents[1] / "results"
SOURCE = RESULTS / "E/E_V5/CITRUS_EV5_ALL_300EP"
OUT = ROOT / "docs/E_V6_REVIEW_20260912"
AP = "metrics/mAP50-95(M)"


def main():
    history, runs, errors = [], [], []
    yaml_index = {}
    for path in (ROOT / "0_orange_yaml").rglob("*.yaml"):
        yaml_index.setdefault(path.name, []).append(path)
    for path in sorted(RESULTS.rglob("results.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(f)]
            args_path = path.parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
            stem = Path(str(args.get("model", "")).replace("\\", "/")).name
            sources = []
            for cfg in yaml_index.get(stem, []):
                content = yaml.safe_load(cfg.read_text(encoding="utf-8"))
                if isinstance(content, dict):
                    sources.append(dict(path=str(cfg), sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
                                        modules=[r[2] for r in content.get("backbone", []) + content.get("head", [])]))
            r = dict(path=str(path), name=path.parent.name, epochs=len(rows),
                     best=max(rows, key=lambda r: r[AP]),
                     best50=max(rows, key=lambda r: r["metrics/mAP50(M)"]),
                     last20=statistics.mean(r[AP] for r in rows[-20:]), args=args, sources=sources,
                     median_epoch_s=statistics.median(b["time"]-a["time"] for a,b in zip(rows,rows[1:]))
                     if len(rows)>1 and "time" in rows[0] else None,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            history.append(r)
            if path.parent.parent != SOURCE:
                continue
            r = dict(r)
            r["model"] = path.parent.name.split("_seed")[0]
            r["factors"] = FACTORS[NAMES.index(r["model"])]
            r["paired"] = {}
            for budget in ("coarse", "fine"):
                pointer = path.parent / f"paired_{budget}_evaluation.json"
                candidates = ([path.parent / json.loads(pointer.read_text())["report"]] if pointer.exists() else [])
                candidates += sorted(path.parent.glob(f"paired_{budget}_eval*/paired_metrics.json"))
                report = next((p for p in candidates if p.is_file()), None)
                if report is None:
                    r["paired"][budget] = None
                    continue
                data = json.loads(report.read_text())
                r["paired"][budget] = dict(path=str(report), summary=data["summary"], protocol=data["protocol"],
                                          records=len(data["records"]),
                                          images=len({x["image"] for x in data["records"]}))
            r["loaded"] = json.loads((path.parent / "loaded_data_summary.json").read_text())
            r["val_list_sha256"] = hashlib.sha256((path.parent / "val_loaded_files.txt").read_bytes()).hexdigest()
            marker = json.loads((path.parent / "completed.json").read_text())
            r["yaml_matches"] = any(s["sha256"] == marker["yaml_sha256"] for s in sources)
            runs.append(r)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=repr(error)))
    pairs = []
    for axis, label in enumerate(("fine_input", "mask_route", "region")):
        for a in runs:
            if a["factors"][axis]:
                continue
            desired = list(a["factors"])
            desired[axis] = 1
            b = next((r for r in runs if list(r["factors"]) == desired), None)
            if b is not None:
                pairs.append(dict(factor=label, before=a["model"], after=b["model"],
                                  delta_ap=100*(b["best"][AP]-a["best"][AP]),
                                  delta_last20=100*(b["last20"]-a["last20"])))
    output = dict(history=history, runs=runs, pairs=pairs, errors=errors,
                  unique_csv=len({r["sha256"] for r in history}),
                  missing_models=sorted(set(NAMES)-{r["model"] for r in runs}),
                  limit="Current YAML/module mapping is not proof of the unprovided server Python source identity.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 全部历史结果索引", "", "AP50 和 AP50–95 为同轮指标；跨数据/AMP/初始化/评估协议不可直接排名。", "",
             "| 运行 | 轮数 | Mask AP50 | Mask AP50–95 | AMP |", "|---|---:|---:|---:|---|"]
    for r in history:
        lines.append(f"| {Path(r['path']).relative_to(RESULTS).as_posix()} | {r['epochs']} | "
                     f"{100*r['best']['metrics/mAP50(M)']:.3f} | {100*r['best'][AP]:.3f} | {r['args'].get('amp')} |")
    (OUT/"全部历史结果索引.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"History={len(history)}, unique={output['unique_csv']}, errors={errors}, missing={output['missing_models']}")
    for r in runs:
        print(r["model"], r["epochs"], round(100*r["best"][AP],3),
              round(100*r["best"]["metrics/mAP50(M)"],3), round(100*r["last20"],3), r["median_epoch_s"])
        for budget, d in r["paired"].items():
            if d:
                s=d["summary"]["trustedmask"]
                print(budget, "AP",round(100*s[AP],3), "tiny",s["tiny_matched"], "bg",s["errors25"]["background"],
                      "P90R",s["operating_p90"], "images",d["images"])
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
