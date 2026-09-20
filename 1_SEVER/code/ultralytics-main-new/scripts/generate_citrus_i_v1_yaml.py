"""Generate I V1 YAMLs from explicit V12 source anchors, without overwriting history."""
# ruff: noqa: E402

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_i_v1_suite import FACTORS, NAMES, V12_PARENTS, V12_YAML_DIR, YAML_DIR


def configs():
    results = []
    for name, (parent, route, dual, context, feedback, lead) in zip(NAMES, FACTORS):
        path = V12_YAML_DIR / f"{V12_PARENTS[parent]}.yaml"
        cfg = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
        cfg["head"][-1][2] = "SegmentCitrusIV1"
        args = cfg["head"][-1][3]
        args[12], args[13] = route, True
        args.extend([dual, context, feedback, lead])
        cfg["pretrained_map_family"] = name
        results.append(cfg)
    return results


if __name__ == "__main__":
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        path = YAML_DIR / f"{name}.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(
            "# I V1: controlled recipe in citrus_i_v1_suite.py\n" + yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        print(path.name)
