"""Return canonical E V3 configurations; generation is verified against checked-in YAMLs."""

from copy import deepcopy

import yaml

from citrus_e_v3_suite import FACTORS, ROOT


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V2_series/E20_fixed_control.yaml").read_text())
    output = []
    for deep, neck, quality in FACTORS:
        cfg = deepcopy(base)
        if deep:
            for i in (6, 8):
                cfg["backbone"][i][2] = "EV3DeepStage"
        if neck:
            # Indices 0--16 and 19--23 retain their original keys and widths.
            cfg["head"][6] = [-1, 1, "EV3DetailDown", [256]]  # layer 17, P3 -> P4
            cfg["head"][7] = [[17, 13, 6], 1, "EV3NativeFusion", []]
        if quality:
            cfg["head"][-1][2] = "SegmentCitrusEV3Quality"
            cfg["head"][-1][3].append(1.0)
        output.append(cfg)
    return output
