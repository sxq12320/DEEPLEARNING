"""Deterministically render J4 YAMLs in conventional one-layer-per-line format."""

import copy
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_j4_suite import DESCRIPTIONS, NAMES, YAML_DIR  # noqa: E402
from scripts.generate_citrus_j2_yaml import BODY  # noqa: E402


def configs():
    anchor = yaml.safe_load((ROOT / "0_orange_yaml/E_V12_series/V12_04_shared_route_control.yaml").read_text("utf-8"))
    anchor["nc"] = 1
    anchor.pop("pretrained_layer_map", None)
    anchor.pop("pretrained_map_family", None)
    stock = yaml.safe_load(BODY + "  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]\n")
    result = {NAMES[0]: stock, NAMES[1]: anchor}
    direct = copy.deepcopy(anchor)
    direct["head"] = direct["head"][:6] + [
        [[20, 17, 14, 2, 0, 10, 9], 1, "SegmentCitrusEV12", copy.deepcopy(anchor["head"][-1][3])]
    ]
    result[NAMES[2]] = direct
    projected = copy.deepcopy(direct)
    projected["head"] = [
        [[10, 14], 1, "J2ResidualFuse", [512, 1]],
        [[5, 15], 1, "J2ResidualFuse", [256, 1]],
        [[16, 15, 14, 2, 0, 10, 9], 1, "SegmentCitrusEV12", copy.deepcopy(anchor["head"][-1][3])],
    ]
    result[NAMES[3]] = projected
    narrow = copy.deepcopy(projected)
    for index in (11, 12, 13, 14):
        narrow["backbone"][index][3][0] = 768
    result[NAMES[4]] = narrow
    shared = copy.deepcopy(narrow)
    shared["head"][-1][2] = "SegmentCitrusJ4"
    shared["head"][-1][3] += [48, True, True]
    result[NAMES[5]] = shared
    base_loss = copy.deepcopy(shared)
    for index in (9, 10, 11):
        base_loss["head"][-1][3][index] = 0.0
    base_loss["head"][-1][3][-1] = False
    result[NAMES[6]] = base_loss
    thin = copy.deepcopy(shared)
    thin["backbone"][3][3][0] = 12
    thin["head"][-1][3][3] = 12
    result[NAMES[7]] = thin
    return result


def render(name, config):
    lines = [f"# {name}: {DESCRIPTIONS[name]}", "# Untrained candidate; common hyperparameters in RUN_CITRUS_J4.py.",
             "nc: 1", "scale: n", "scales:", "  n: [0.50, 0.25, 1024]"]
    for section in ("backbone", "head"):
        lines.append(section + ":")
        lines.extend("  - " + json.dumps(row) for row in config[section])
    return "\n".join(lines) + "\n"


def main(check=False):
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, config in configs().items():
        path, text = YAML_DIR / (name + ".yaml"), render(name, config)
        if check:
            if not path.is_file() or path.read_text("utf-8") != text:
                raise RuntimeError(f"Stale J4 YAML: {path}")
        else:
            path.write_text(text, encoding="utf-8")
    if not check:
        index = ROOT / "0_orange_yaml/MODEL_INDEX.csv"
        with index.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [r for r in reader if r["series"] != "J_V4_series"]
        for name, config in configs().items():
            rows.append(dict(series="J_V4_series", model=name, yaml=f"J_V4_series/{name}.yaml",
                             head=config["head"][-1][2], nc="1", series_status="untrained/controlled hypothesis",
                             parse_error=""))
        with index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{'Checked' if check else 'Generated'} eight J4 YAMLs")


if __name__ == "__main__":
    main("--check" in sys.argv)
