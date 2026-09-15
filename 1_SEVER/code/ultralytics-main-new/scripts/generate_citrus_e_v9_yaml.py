"""Reproducible E V9 graph generation; refuse to overwrite any model config."""
# ruff: noqa: E402

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v9_suite import FACTORS, NAMES, YAML_DIR


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V8_series/V8_03_geometry_cp.yaml").read_text(encoding="utf-8"))
    result = []
    for i, (neck, compact, tiny, negative) in enumerate(FACTORS):
        cfg = deepcopy(base)
        cfg["head"][-1] = [
            [16, 19, 10, 2, 0, 6],
            1,
            "SegmentCitrusEV9",
            [
                "nc",
                32,
                256,
                16,
                bool(neck),
                bool(compact),
                0.25 * tiny,
                0.1 * negative,
                0.5 if i else 0.0,
                0.25 if i else 0.0,
            ],
        ]
        cfg["pretrained_map_family"] = f"citrus_ev9_n{neck}_c{compact}"
        result.append(cfg)
    return result


if __name__ == "__main__":
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        target = YAML_DIR / f"{name}.yaml"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(
            "# E V9: training recipe in citrus_e_v9_suite.py; see docs/E_V9_RECONSTRUCTION.md\n"
            + yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        print(target.name)
