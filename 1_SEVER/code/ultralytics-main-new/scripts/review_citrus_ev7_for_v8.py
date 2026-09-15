"""Read-only result/source audit; comparisons remain stratified by protocol."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parents[1] / "results"
OUT = ROOT / "docs/E_V8_REVIEW_20260914"
AP = "metrics/mAP50-95(M)"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index = {}
    for p in (ROOT / "0_orange_yaml").rglob("*.yaml"):
        index.setdefault(p.name, []).append(p)
    history, errors = [], []
    for p in sorted(RESULTS.rglob("results.csv")):
        try:
            with p.open(encoding="utf-8-sig", newline="") as f:
                rows = [{k.strip(): float(v) for k, v in r.items() if k and v.strip()} for r in csv.DictReader(f)]
            args_path = p.parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
            sources = []
            basename = Path(str(args.get("model", "")).replace("\\", "/")).name
            for source in index.get(basename, []):
                config = yaml.safe_load(source.read_text(encoding="utf-8"))
                if isinstance(config, dict):
                    sources.append(
                        dict(
                            path=str(source),
                            sha256=sha(source),
                            layers=config.get("backbone", []) + config.get("head", []),
                        )
                    )
            times = [b["time"] - a["time"] for a, b in zip(rows, rows[1:]) if "time" in a and b["time"] > a["time"]]
            r = dict(
                path=str(p),
                name=p.parent.name,
                epochs=len(rows),
                sha256=sha(p),
                args=args,
                best=max(rows, key=lambda x: x[AP]),
                last20=statistics.mean(x[AP] for x in rows[-20:]),
                median_epoch_s=statistics.median(times) if times else None,
                sources=sources,
                paired={},
            )
            for budget in ("coarse", "fine"):
                reports = sorted(p.parent.glob(f"paired_{budget}_eval*/paired_metrics.json"))
                if reports:
                    d = json.loads(reports[-1].read_text(encoding="utf-8"))
                    r["paired"][budget] = {k: d[k] for k in ("summary", "protocol", "weights")}
            for name in ("completed", "loaded_data_summary", "initialization_transfer"):
                q = p.parent / f"{name}.json"
                if q.exists():
                    r[name] = json.loads(q.read_text(encoding="utf-8"))
            for name in ("train_loaded_files", "val_loaded_files"):
                q = p.parent / f"{name}.txt"
                if q.exists():
                    r[name + "_sha256"] = sha(q)
            history.append(r)
        except (OSError, ValueError, TypeError, KeyError) as e:
            errors.append(dict(path=str(p), error=repr(e)))
    v7 = [r for r in history if "/E/E_V7/" in r["path"].replace("\\", "/")]
    OUT.mkdir(parents=True, exist_ok=True)
    output = dict(
        history=history,
        v7=v7,
        errors=errors,
        unique_csv=len({r["sha256"] for r in history}),
        limitation="Current YAML mapping is not historical server Python provenance. Single seed, validation only.",
    )
    (OUT / "audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 全历史结果与当前源码索引",
        "",
        "不同划分、AMP、初始化和评估协议不直接排名；同一行指标取同轮。",
        "",
        "|结果路径|轮数|Mask AP50|Mask AP50–95|末20轮AP|AMP|YAML候选数|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in history:
        lines.append(
            f"|{Path(r['path']).relative_to(RESULTS).as_posix()}|{r['epochs']}|"
            f"{r['best']['metrics/mAP50(M)'] * 100:.3f}|{r['best'][AP] * 100:.3f}|"
            f"{r['last20'] * 100:.3f}|{r['args'].get('amp')}|{len(r['sources'])}|"
        )
    (OUT / "全部历史结果索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("history", len(history), "unique", output["unique_csv"], "mapped", sum(bool(r["sources"]) for r in history))
    print("errors", errors)
    for r in v7:
        print(
            r["name"],
            r["epochs"],
            "AP50/AP/last20",
            *(round(x * 100, 3) for x in (r["best"]["metrics/mAP50(M)"], r["best"][AP], r["last20"])),
            "sec",
            r["median_epoch_s"],
        )
        for budget, d in r["paired"].items():
            for mode in ("global", "trustedmask"):
                s = d["summary"][mode]
                print(
                    budget,
                    mode,
                    "AP50/AP",
                    round(s["metrics/mAP50(M)"] * 100, 3),
                    round(s[AP] * 100, 3),
                    "tiny",
                    s["tiny_matched"],
                    "bg",
                    s["errors25"]["background"],
                    "P90R",
                    round(s["operating_p90"]["recall"] * 100, 3),
                    "ms",
                    round(s["median_ms"], 2),
                )


if __name__ == "__main__":
    main()
