"""VS Code entry: run all E12 candidates sequentially on original and min8 labels."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

import RUN_LABEL8_COMPARE as paired
from RUN_ES03_LABEL8_COMPARE import audit_labels
from citrus_control78_batch import ROOT, digest, project_lock, save_json, training_done
from citrus_e_v12_suite import NAMES as V12_NAMES
from citrus_paper_suite import NAMES as PAPER_NAMES

OLD_DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
NEW_DATA = "/data/sxq/datasets/orange_yolo_grouped_min8px640_20261107/data.yaml"
PROJECT = "/data/sxq/results/E12_ALL_LABEL8_20261108"
DEVICE = "1"
SEEDS = [42]
EPOCHS = 300
BATCH = 16
WORKERS = 4
CACHE = True
EVALUATE_TEST = False
DRY_RUN = False
SMOKE = False
ONLY = []  # Empty = all E12 models. Otherwise supply YAML stems, e.g. ["ES01_detail", "V12_04_shared_route_control"].

SIMPLE_NAMES = ("ES00_baseline", "ES01_detail", "ES02_semantic", "ES03_detail_semantic")


def candidates():
    """Group named controls by the graph actually trained with the scratch nano recipe."""
    groups = {}
    for folder, names in (("E12_SIMPLE", SIMPLE_NAMES), ("E_V12_series", V12_NAMES),
                          ("PAPER_ABLATION", PAPER_NAMES)):
        for name in names:
            path = ROOT / "0_orange_yaml" / folder / (name + ".yaml")
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            config.update(nc=1, scale="n", scales={"n": config["scales"]["n"]})
            # Only transfer metadata differs between certain historical aliases. No pretrained weights are loaded.
            for key in ("pretrained_map_family", "pretrained_layer_map"):
                config.pop(key, None)
            signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if signature not in groups:
                groups[signature] = dict(name=name, aliases=[], config=config)
            groups[signature]["aliases"].append(name)
    requested = set(ONLY)
    unknown = requested - {name for group in groups.values() for name in group["aliases"]}
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return [g for g in groups.values() if not requested or requested.intersection(g["aliases"])]


def worker(plan_path, model_name):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    yaml_path = Path(plan["root"]) / "configs" / (model_name + ".yaml")
    if digest(yaml_path) != plan["yaml_hashes"][model_name]:
        raise RuntimeError(f"Effective YAML changed for {model_name}")
    paired.BATCH, paired.WORKERS, paired.CACHE = plan["batch"], plan["workers"], plan["cache"]
    paired.main(model_yaml=yaml_path, project_dir=Path(plan["root"]) / model_name, old_data=plan["old"],
                new_data=plan["new"], device=plan["device"], seeds=plan["seeds"], epochs=plan["budget"],
                evaluate_test=plan["test"], check_labels=lambda specs: plan["audit"])


def completed(folder, plan):
    expected = len(plan["seeds"]) * (2 if plan["test"] else 1)
    for filename, count in (("LABEL8_RESULTS.csv", 4*expected), ("LABEL8_DELTA.csv", 2*expected)):
        file = folder / filename
        if not file.exists():
            return False
        with file.open(encoding="utf-8-sig", newline="") as handle:
            if len(list(csv.DictReader(handle))) != count:
                return False
    return all(training_done(folder / f"train_{policy}_seed{seed}", plan["budget"])
               for seed in plan["seeds"] for policy in ("all", "min8"))


def summarize(root, queue, plan):
    for file in ("LABEL8_RESULTS.csv", "LABEL8_DELTA.csv"):
        rows = []
        for item in queue:
            folder = root / item["name"]
            if (folder / "suite_completed.json").is_file() and completed(folder, plan):
                with (folder / file).open(encoding="utf-8-sig", newline="") as handle:
                    rows.extend(dict(model=item["name"], aliases="|".join(item["aliases"]), **row)
                                for row in csv.DictReader(handle))
        paired.write_table(root / ("ALL_" + file), rows)


def main():
    queue = candidates()
    if not SEEDS or len(set(SEEDS)) != len(SEEDS):
        raise ValueError("Use distinct seeds")
    fits = 2 * len(SEEDS) * len(queue)
    print(f"{len(queue)} models, {fits} fits, {fits * (4 if EVALUATE_TEST else 2)} final evaluations", flush=True)
    for item in queue:
        print("  " + " = ".join(item["aliases"]), flush=True)
    if DRY_RUN:
        return
    specs = paired.dataset_configs(OLD_DATA, NEW_DATA, EVALUATE_TEST)
    audit = audit_labels(specs)
    root = Path(str(PROJECT) + ("_SMOKE" if SMOKE else "")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with project_lock(root):
        configs = {item["name"]: yaml.safe_dump(item["config"], sort_keys=False).encode() for item in queue}
        plan = dict(root=str(root), old=str(Path(OLD_DATA).resolve()), new=str(Path(NEW_DATA).resolve()),
                    device=str(DEVICE), budget=1 if SMOKE else EPOCHS, batch=BATCH, workers=WORKERS, cache=CACHE,
                    seeds=SEEDS, test=EVALUATE_TEST, audit=audit,
                    yaml_hashes={name: hashlib.sha256(data).hexdigest() for name, data in configs.items()})
        plan_path = root / "suite_protocol.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise FileExistsError(f"Existing project has a different queue or settings: {root}")
        save_json(plan_path, plan)
        (root / "configs").mkdir(exist_ok=True)
        for name, data in configs.items():
            (root / "configs" / (name + ".yaml")).write_bytes(data)
        paired.write_table(root / "MODELS.csv", [dict(model=g["name"], aliases="|".join(g["aliases"])) for g in queue])
        for i, item in enumerate(queue, 1):
            name = item["name"]
            folder = root / name
            marker = folder / "suite_completed.json"
            if not marker.is_file() or not completed(folder, plan):
                print(f"[{i}/{len(queue)}] {name}", flush=True)
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                                str(plan_path), name], check=True)
                if not completed(folder, plan):
                    raise RuntimeError(f"Incomplete training/evaluation for {name}")
                save_json(marker, dict(model=name, aliases=item["aliases"]))
            else:
                print(f"[{i}/{len(queue)}] Skip finished {name}", flush=True)
            summarize(root, queue, plan)
    print(f"Finished: {root / 'ALL_LABEL8_DELTA.csv'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
