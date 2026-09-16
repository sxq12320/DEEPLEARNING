"""Deterministic V10 graphs, with independent legacy controls and no overwritten YAML."""
# ruff: noqa: E402

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v10_suite import FACTORS, NAMES, YAML_DIR


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V9_series/V9_05_tiny_overlap.yaml").read_text(encoding="utf-8"))
    result = []
    for stem, transport, decoder, tiny, visibility in FACTORS:
        cfg = deepcopy(base)
        if stem:
            cfg["backbone"][0][2] = "EV10ContrastStem"
        cfg["head"][-1] = [
            [16, 19, 10, 2, 0, 6],
            1,
            "SegmentCitrusEV10",
            ["nc", 32, 256, 16, bool(transport), bool(decoder), 0.25 * tiny, 0.1 * visibility, 0.5, 0.25],
        ]
        cfg["pretrained_map_family"] = f"citrus_ev10_s{stem}_t{transport}_d{decoder}"
        result.append(cfg)
    return result


if __name__ == "__main__":
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        target = YAML_DIR / f"{name}.yaml"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(
            "# E V10: fixed training recipe in citrus_e_v10_suite.py\n" + yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        print(target.name)
