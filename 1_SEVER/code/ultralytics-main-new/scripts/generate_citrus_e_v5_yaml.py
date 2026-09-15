"""Generate checked-in V5 YAMLs; identical architectures intentionally repeat for input ablations."""

from copy import deepcopy

import yaml

from citrus_e_v5_suite import FACTORS, ROOT


def configs():
    base = yaml.safe_load(
        (ROOT / "0_orange_yaml/E_V4_series/reconstruction_20260910/V4R05_deep_quality.yaml").read_text(encoding="utf-8")
    )
    output = []
    for fine, route, region in FACTORS:
        cfg = deepcopy(base)
        cfg["head"][-1][2] = "SegmentCitrusEV5"
        cfg["head"][-1][3] = ["nc", 32, 256, 16, bool(route), 0.05 if region else 0.0]
        # Dataset policy is owned by the runner, not silently activated by parse_model().
        output.append(cfg)
    return output
