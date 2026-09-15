"""Canonical configurations; checked-in files are verified against these definitions."""

from copy import deepcopy

import yaml

from citrus_e_v4r_suite import FACTORS, ROOT


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V3_series/E30_control.yaml").read_text(encoding="utf-8"))
    output = []
    for deep, detail, quality in FACTORS:
        cfg = deepcopy(base)
        if deep:
            cfg["backbone"][8][2] = "EV3DeepStage"
        if quality:
            cfg["head"][-1][2] = "SegmentCitrusEV4Quality"
            cfg["head"][-1][3].extend([1.0, bool(detail), 0.5])
        elif detail:
            cfg["head"][-1][2] = "SegmentCitrusEV4Detail"
        output.append(cfg)
    return output
