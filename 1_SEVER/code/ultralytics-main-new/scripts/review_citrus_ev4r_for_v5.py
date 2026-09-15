"""Read uploaded E V4R and all historical CSVs; preserve original results and report caveats."""

import csv
import hashlib
import inspect
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v4r_suite import FACTORS, NAMES

RESULTS = ROOT.parents[1] / "results"
SOURCE = RESULTS / "E/E_V4R/CITRUS_EV4R_ALL_300EP"
OUT = ROOT / "docs/E_V5_RECONSTRUCTION_EVIDENCE"
AP = "metrics/mAP50-95(M)"


def main():
    history, runs, errors = [], [], []
    yaml_index = {}
    for path in (ROOT / "0_orange_yaml").rglob("*.yaml"):
        yaml_index.setdefault(path.name, []).append(path)
    for path in sorted(RESULTS.rglob("results.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [
                    {k.strip(): float(v) for k, v in r.items() if k and v and v.strip()} for r in csv.DictReader(stream)
                ]
            peak = max(rows, key=lambda row: row[AP])
            args_path = path.parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
            stem = Path(str(args.get("model", "")).replace("\\", "/")).name
            candidates = yaml_index.get(stem, [])
            source_configs = []
            for candidate in candidates:
                cfg = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                if isinstance(cfg, dict):
                    source_configs.append(
                        dict(
                            path=str(candidate.relative_to(ROOT)),
                            modules=[layer[2] for layer in cfg.get("backbone", []) + cfg.get("head", [])],
                            sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                        )
                    )
            item = dict(
                name=path.parent.name,
                path=str(path),
                epochs=len(rows),
                peak=peak,
                peak_ap50=max(rows, key=lambda row: row["metrics/mAP50(M)"]),
                args=args,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_configs=source_configs,
                last20_ap=statistics.mean(row[AP] for row in rows[-20:]),
            )
            history.append(item)
            if path.parent.parent != SOURCE:
                continue
            read = lambda name: json.loads((path.parent / name).read_text(encoding="utf-8"))
            init = read("initialization_transfer.json")
            marker = read("completed.json")
            name = path.parent.name.split("_seed")[0]
            cfg = ROOT / f"0_orange_yaml/E_V4_series/reconstruction_20260910/{name}.yaml"
            metrics = read("paired_sliced_eval/paired_metrics.json")
            r = dict(
                item,
                name=name,
                factors=list(FACTORS[NAMES.index(name)]),
                summary=metrics["summary"],
                metrics_keys=list(metrics),
                init_fraction=init["equal_fraction"],
                params=init["total_parameter_numel"],
                loaded=read("loaded_data_summary.json"),
                yaml_matches=hashlib.sha256(cfg.read_bytes()).hexdigest() == marker["yaml_sha256"],
                selected=read("best_mask_selection.json"),
                val_list_sha256=hashlib.sha256((path.parent / "val_loaded_files.txt").read_bytes()).hexdigest(),
                median_epoch_s=statistics.median(b["time"] - a["time"] for a, b in zip(rows, rows[1:])),
            )
            records = metrics.get("records", [])
            r["record_example"] = records[:1]
            r["record_count"] = len(records)
            runs.append(r)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=repr(error)))
    runs.sort(key=lambda r: r["name"])
    assert len(runs) == 8, f"Missing E V4R results: {len(runs)}; errors={errors}"
    ignored = {"name", "model", "data", "save_dir", "project"}
    for r in runs:
        r["args_diff"] = {
            k: [runs[0]["args"].get(k), r["args"].get(k)]
            for k in runs[0]["args"].keys() | r["args"].keys()
            if k not in ignored and runs[0]["args"].get(k) != r["args"].get(k)
        }
    effects = {}
    for axis, name in enumerate(("deep8", "detail", "quality")):
        pairs = []
        for a in runs:
            if a["factors"][axis]:
                continue
            desired = a["factors"].copy()
            desired[axis] = 1
            b = next(r for r in runs if r["factors"] == desired)
            pairs.append(
                dict(
                    before=a["name"],
                    after=b["name"],
                    peak_delta_pp=100 * (b["peak"][AP] - a["peak"][AP]),
                    last20_delta_pp=100 * (b["last20_ap"] - a["last20_ap"]),
                    trusted_ap_delta_pp=100 * (b["summary"]["trustedmask"][AP] - a["summary"]["trustedmask"][AP]),
                    trusted_tiny_delta=b["summary"]["trustedmask"]["tiny_matched"]
                    - a["summary"]["trustedmask"]["tiny_matched"],
                )
            )
        effects[name] = pairs
    payload = dict(
        history=history,
        runs=runs,
        effects=effects,
        errors=errors,
        unique_csv=len({r["sha256"] for r in history}),
        missing_protocol=not (SOURCE / "_protocol").exists(),
    )
    # This maps CURRENT implementations, not unprovided server source snapshots.
    import ultralytics.nn.tasks as tasks

    symbols = sorted({symbol for item in history for cfg in item["source_configs"] for symbol in cfg["modules"]})
    implementations = {}
    for symbol in symbols:
        obj = getattr(tasks, symbol, None)
        if obj is None:
            implementations[symbol] = {"status": "framework symbol or unresolved", "symbol": symbol}
            continue
        try:
            source_file = Path(inspect.getsourcefile(obj))
            source_lines, line = inspect.getsourcelines(obj)
            implementations[symbol] = dict(
                path=str(source_file),
                line=line,
                class_source_sha256=hashlib.sha256("".join(source_lines).encode()).hexdigest(),
                source_lines=len(source_lines),
                status="current local implementation",
            )
        except (TypeError, OSError) as error:
            implementations[symbol] = {"status": repr(error)}
    payload["implementations"] = implementations
    payload["source_limit"] = (
        "All CSV/YAML references enumerated; current class symbols mapped. This is not proof of server source identity or an exhaustive semantic audit of every historical line."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 全部历史运行索引",
        "",
        "_2026-09-11；同一 epoch 的 Mask AP50 / AP50–95，非跨实验公平排行榜。_",
        "",
        "---",
        "",
        "同内容 CSV 详见 audit.json；展示副本仍列出，以路径区分。参数、数据划分、初始化和评估器未完全相同的运行不能据此判定架构胜负。",
        "",
        "| 运行 | Epochs | Mask AP50 | Mask AP50–95 | AMP / 优化器 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in history:
        relative = str(Path(item["path"]).relative_to(RESULTS)).replace("\\", "/")
        lines.append(
            f"| {relative} | {item['epochs']} | {100 * item['peak']['metrics/mAP50(M)']:.3f} | "
            f"{100 * item['peak'][AP]:.3f} | {item['args'].get('amp', '缺失')} / {item['args'].get('optimizer', '缺失')} |"
        )
    (OUT / "全部历史运行索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"History={len(history)} unique={payload['unique_csv']} errors={errors}")
    print("model ep AP50 AP5095 last20 sec init% | globalTiny sliceTiny sliceAP5095 R@P90 bg dup")
    for r in runs:
        g, s = r["summary"]["global"], r["summary"]["trustedmask"]
        print(
            r["name"],
            r["epochs"],
            *[
                round(v, 3)
                for v in [
                    100 * r["peak"]["metrics/mAP50(M)"],
                    100 * r["peak"][AP],
                    100 * r["last20_ap"],
                    r["median_epoch_s"],
                    100 * r["init_fraction"],
                ]
            ],
            "|",
            g["tiny_matched"],
            s["tiny_matched"],
            round(100 * s[AP], 3),
            round(100 * s["operating_p90"]["recall"], 3),
            s["errors25"],
        )
    print(json.dumps(effects, indent=2))
    print("Protocol deltas", [r["args_diff"] for r in runs])
    print("YAML checks", [r["yaml_matches"] for r in runs], "val lists", len({r["val_list_sha256"] for r in runs}))


if __name__ == "__main__":
    main()
