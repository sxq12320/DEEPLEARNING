"""Deterministically generate ten V8 configs; never overwrite prior experiments."""
# ruff: noqa: E402 -- establish local project import root before importing the suite.

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v8_suite import FACTORS, NAMES, YAML_DIR


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V7_series/V7_00_control.yaml").read_text(encoding="utf-8"))
    result = []
    for backbone, neck, geometry, _cp, dense in FACTORS:
        cfg = deepcopy(base)
        mapping = {}
        if backbone:
            cfg["backbone"][6] = [-1, 2, "EV8ContextStage", [512]]
            mapping[6] = -1  # New whole stage: do not accidentally transfer same-shaped tensors.
        if neck:
            cfg["head"][6] = [-1, 1, "nn.Identity", []]  # remove P3 downsample
            cfg["head"][7] = [-1, 1, "nn.Identity", []]  # remove concat
            cfg["head"][8] = [[13, 6, 4], 1, "EV8P4Reconcile", [512]]
            mapping[19] = -1
        args = cfg["head"][-1][3]
        args[6] = True if dense else "phase"
        args[10:12] = [0.5, 0.25] if geometry else [0.0, 0.0]
        cfg["pretrained_map_family"] = f"citrus_ev8_b{backbone}_n{neck}_d{dense}"
        if mapping:
            cfg["pretrained_layer_map"] = mapping
        result.append(cfg)
    return result


if __name__ == "__main__":
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        target = YAML_DIR / f"{name}.yaml"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(
            "# E V8: see docs/E_V8_RECONSTRUCTION.md; losses/CP are explicit ablations.\n"
            + yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        print(target.name)
