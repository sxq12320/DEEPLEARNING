"""V11 source-registered standard YOLO graphs; no modification of historical YAML."""
# ruff: noqa: E402

import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_e_v11_suite import CONTEXT, FACTORS, NAMES, YAML_DIR


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V10_series/V10_05_structure_pair.yaml").read_text(encoding="utf-8"))
    results = []
    for name, (stem, persistent, rep, native, assignment, ring, tiny, box) in zip(NAMES, FACTORS):
        cfg = deepcopy(base)
        cfg["backbone"][0][2] = "EV10ContrastStem" if stem else "Conv"
        if persistent:
            old = cfg["backbone"]
            cfg["backbone"] = [
                old[0],
                old[1],
                old[2],
                [[2, 0], 1, "EV11DetailSeed", [16]],
                [2, *old[3][1:]],
                old[4],
                [[3, 5], 1, "EV11DetailExchange", []],
                [5, *old[5][1:]],
                old[6],
                [[6, 8], 1, "EV11DetailExchange", []],
                [[8, 9], 1, "EV11DetailInject", []],
                [10, *old[7][1:]],
                old[8],
                old[9],
                old[10],
            ]
            if rep:
                cfg["backbone"][5][2] = cfg["backbone"][8][2] = cfg["backbone"][12][2] = "EV11RepStage"
            cfg["head"] = [
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 10], 1, "Concat", [1]],
                [-1, 2, "C3k2", [512, False]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 5], 1, "Concat", [1]],
                [-1, 2, "C3k2", [256, False]],
                [-1, 1, "Conv", [256, 3, 2]],
                [[-1, 17, 10], 1, "EV11NativeFusion", []] if native else [[-1, 17], 1, "Concat", [1]],
                [-1, 2, "C3k2", [512, False]],
                [-1, 1, "nn.Identity", []],
                [-1, 1, "nn.Identity", []],
                [-1, 1, "nn.Identity", []],
                [[20, 23, 14, 2, 0, 10, 9], 1, "SegmentCitrusEV11", []],
            ]
            # Official pretrained stages/towers must follow their semantic role, not raw YAML index.
            cfg["pretrained_layer_map"] = {
                3: -1,
                4: 3,
                5: 4,
                6: -1,
                7: 5,
                8: 6,
                9: -1,
                10: -1,
                11: 7,
                12: 8,
                13: 9,
                14: 10,
                15: 11,
                16: 12,
                17: 13,
                18: 14,
                19: 15,
                20: 16,
                21: 17,
                22: -1,
                23: 19,
                24: -1,
                25: -1,
                26: -1,
                27: 23,
            }
        else:
            cfg["head"][-1][2] = "SegmentCitrusEV11"
        cfg["head"][-1][3] = [
            "nc",
            32,
            256,
            16,
            bool(persistent),
            not bool(persistent),
            bool(box),
            float(assignment),
            float(ring),
            float(tiny),
            0.5,
            0.25,
        ]
        cfg["pretrained_map_family"] = f"citrus_ev11_s{stem}_h{persistent}_r{rep}_n{native}_q{box}"
        if name in CONTEXT:
            cfg["backbone"][14][2:] = ["EV11ContextStage", [1024, CONTEXT[name]]]
            cfg["pretrained_map_family"] += f"_selective{int(CONTEXT[name])}"
        results.append(cfg)
    return results


if __name__ == "__main__":
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        p = YAML_DIR / f"{name}.yaml"
        if p.exists():
            raise FileExistsError(p)
        p.write_text(
            "# E V11: fixed recipe in citrus_e_v11_suite.py\n" + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        print(p.name)
