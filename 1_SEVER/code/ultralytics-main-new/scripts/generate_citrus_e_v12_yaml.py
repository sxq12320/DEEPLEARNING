"""Generate V12 YAMLs from explicit V11 source anchors, without overwriting history."""
# ruff: noqa: E402

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v11_suite import NAMES as OLD_NAMES
from citrus_e_v12_suite import FACTORS, NAMES, YAML_DIR


def configs():
    results = []
    for name, (parent, route, difference, box, stem, tiny) in zip(NAMES, FACTORS):
        path = ROOT / "0_orange_yaml/E_V11_series" / f"{OLD_NAMES[parent]}.yaml"
        cfg = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")))
        cfg["backbone"][0][2] = "EV10ContrastStem" if stem else "Conv"
        cfg["head"][-1][2] = "SegmentCitrusEV12"
        args = cfg["head"][-1][3]
        args[6], args[7], args[8], args[9] = box, 0.0, 0.0, tiny
        args.extend([route, difference])
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
            "# E V12: controlled recipe in citrus_e_v12_suite.py\n" + yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        print(path.name)
