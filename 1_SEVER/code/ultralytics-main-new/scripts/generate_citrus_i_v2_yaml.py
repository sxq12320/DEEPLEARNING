"""Generate the controlled I V2 YAML family from the frozen I01 architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citrus_i_v2_suite import FACTORS, NAMES, YAML_DIR  # noqa: E402

SOURCE = ROOT / "0_orange_yaml/I_V1_series/I01_replay_shared.yaml"


def _segment_args(factor):
    return [
        "nc",
        32,
        256,
        16,
        True,
        False,
        False,
        factor["assignment_mix"],
        0.0,
        0.25,
        0.5,
        0.25,
        factor["route"],
        True,
        factor["p2_mode"] >= 0,
    ]


def build_yaml(name):
    """Return one factorized architecture without mutating the source recipe."""
    factor = FACTORS[name]
    model = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    model["pretrained_map_family"] = "I01_replay_shared"

    # Only backbone C3/C4 interior mixers change; downsampling and CSP projections remain explicit.
    if factor["ls_depth"] >= 1:
        model["backbone"][5][2] = "IV2LargeSmallStage"
        model["backbone"][5][3].append(7)
    if factor["ls_depth"] >= 2:
        model["backbone"][8][2] = "IV2LargeSmallStage"
        model["backbone"][8][3].extend((0.5, 7))

    model["head"].pop()
    if factor["p2_mode"] >= 0:
        model["head"].append([[9, 20], 1, "IV2P2Candidate", [256, factor["p2_mode"], 7]])
        sources = [20, 23, 14, 27, 2, 0, 10, 9]
        model["pretrained_layer_map"][27] = -1
        model["pretrained_layer_map"][28] = 23
    else:
        sources = [20, 23, 14, 2, 0, 10, 9]
        model["pretrained_layer_map"][27] = 23
    model["head"].append([sources, 1, "SegmentCitrusIV2", _segment_args(factor)])
    return model


def main():
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        target = YAML_DIR / f"{name}.yaml"
        target.write_text(yaml.safe_dump(build_yaml(name), sort_keys=False), encoding="utf-8")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
