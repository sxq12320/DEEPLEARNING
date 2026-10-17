"""Predeclared paper controls, not another model-search series."""

import random
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/PAPER_ABLATION"
BASE = ROOT / "0_orange_yaml/E_V12_series/V12_04_shared_route_control.yaml"
NAMES = (
    "P00_yolo11n_control", "P01_v12_full", "P02_no_persistent_detail",
    "P03_no_recognition", "P04_classification_only", "P05_full_pan_neck",
    "P06_bilinear_only", "P07_no_boundary_loss", "P08_no_neighbor_loss",
    "P09_no_tiny_dice",
    "P10_no_extra_supervision",
)


def configurations():
    """Return YAML dictionaries and explicit original-layer transfer maps."""
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    official = yaml.safe_load(
        (ROOT / "ultralytics/cfg/models/11/yolo11-seg.yaml").read_text(encoding="utf-8")
    )
    official.update(scale="n", nc=1)
    result = {NAMES[0]: official}
    for name in NAMES[1:]:
        result[name] = deepcopy(base)
        result[name]["nc"] = 1
    cfg = result[NAMES[2]]
    layers = cfg["backbone"] + cfg["head"]
    kept = [i for i in range(len(layers)) if i not in {3, 6, 9, 10}]
    mapping = {old: new for new, old in enumerate(kept)}

    def remap(index):
        return index if index < 0 else mapping[8 if index == 10 else index]

    rebuilt = []
    for old in kept:
        item = deepcopy(layers[old])
        if old == 27:
            item[0] = item[0][:-1]
            item[3][4] = False
        item[0] = [remap(i) for i in item[0]] if isinstance(item[0], list) else remap(item[0])
        rebuilt.append(item)
    cfg["backbone"], cfg["head"] = rebuilt[:11], rebuilt[11:]
    cfg["pretrained_layer_map"] = {
        mapping[i]: int(base["pretrained_layer_map"].get(i, i)) for i in kept
    }
    result[NAMES[3]]["head"][-1][3][12] = 0
    result[NAMES[4]]["head"][-1][3][12] = 1
    cfg = result[NAMES[5]]
    cfg["head"][-4:-1] = [
        [23, 1, "Conv", [512, 3, 2]],
        [[24, 14], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],
    ]
    cfg["head"][-1][0][2] = 26
    cfg = result[NAMES[6]]
    cfg["head"][-1][2] = "SegmentCitrusPaperAblation"
    cfg["head"][-1][3].extend([False, True])
    for name, index in ((NAMES[7], 10), (NAMES[8], 11), (NAMES[9], 9)):
        result[name]["head"][-1][3][index] = 0.0
    cfg = result[NAMES[10]]
    cfg["head"][-1][2] = "SegmentCitrusPaperAblation"
    cfg["head"][-1][3].extend([True, False])
    for index in (9, 10, 11):
        cfg["head"][-1][3][index] = 0.0
    for name, cfg in result.items():
        if name != NAMES[0]:
            cfg["pretrained_map_family"] = name
    return result


def jobs(suite="paper", seeds=(42, 43, 44), amp_audit=False):
    """Block by seed, randomize arm order and separate AMP sensitivity."""
    if suite not in {"core", "paper"}:
        raise ValueError("suite must be core or paper")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    queue = []
    for seed in seeds:
        names = list(NAMES[:2] if suite == "core" or seed != seeds[0] else NAMES)
        random.Random(20261014 + seed).shuffle(names)
        queue.extend(
            dict(model=n, seed=int(seed), amp=False, role="main" if n in NAMES[:2] else "ablation")
            for n in names
        )
    if amp_audit:
        queue.extend(dict(model=n, seed=int(seeds[0]), amp=True, role="amp_sensitivity") for n in NAMES[:2])
    return queue
