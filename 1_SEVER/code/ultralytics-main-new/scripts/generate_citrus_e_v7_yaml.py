"""Generate checked-in V7 YAMLs from the V6 graph, rebased on the phase-fine path.

Every arm carries the V6-supported candidate stride-2 prototype pathway
(``fine_mask="phase"``), so head inputs include the stride-2 stem feature
(layer 0) after the detail input: ``[P3, P4, P5, (P2), detail, stem]``. The
added fine-path modules live inside the head, so weighted layer indices are
unchanged and the pretrained transfer still lands on inherited tensors. P2
arms insert a stride-4 neck branch after the P3 C3k2 (layers 17-19 in the new
graph) and append the P2 feature as the FOURTH head input;
``pretrained_layer_map`` realigns the three shifted weighted layers and
``pretrained_map_family`` prevents double-remapping of already-adapted V7
checkpoints.
"""

# ruff: noqa: E402 -- the generator establishes the project import root itself.
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from citrus_e_v7_suite import BOUNDARY_GAIN, FACTORS, NAMES, NEIGHBOR_GAIN, ROOT

P2_NECK_INSERT = [
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 17: P3 80x80 -> 160x160
    [[-1, 2], 1, "Concat", [1]],  # 18: fuse backbone C2 (stride-4) detail
    [-1, 1, "C3k2", [128, False, 0.25]],  # 19: P2 feature, n-scale 32ch (was 64ch)
]
P2_LAYER_MAP = {19: -1, 20: 17, 22: 19, 26: 23}  # new P2 must not inherit accidental same-index tensors


def head_args(p2, deform, pmce, boundary):
    """Serialize the SegmentCitrusEV7 argument list for the official YAML.

    ``fine_mask`` is the adopted phase mode for every arm; it is part of the
    rebased control, not a V7 factor.
    """
    return [
        "nc",
        32,
        256,
        16,
        False,
        0.0,
        "phase",
        bool(p2),
        bool(deform),
        bool(pmce),
        BOUNDARY_GAIN if boundary else 0.0,
        NEIGHBOR_GAIN if boundary else 0.0,
    ]


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_V6_series/V6_00_control.yaml").read_text(encoding="utf-8"))
    output = []
    for name, (p2, deform, pmce, boundary) in zip(NAMES, FACTORS):
        cfg = deepcopy(base)
        cfg["pretrained_map_family"] = "citrus_ev7_p2" if p2 else "citrus_ev7_control"
        if p2:
            head = cfg["head"]
            # Splice the stride-4 branch between the P3 C3k2 (index 5 of the
            # head list, layer 16) and the P3->P4 downsample. The downsample's
            # ``-1`` reference would drift to the new P2 C3k2, so it must be
            # pinned to layer 16 explicitly; the later concat's [-1, 13] still
            # resolves correctly because -1 follows the shifted conv.
            head[6:6] = deepcopy(P2_NECK_INSERT)
            head[9][0] = 16  # P3 -> P4 downsample must read layer 16, not the new P2 branch
            head[-1][0] = [16, 22, 10, 19, 2, 0]
            cfg["pretrained_layer_map"] = P2_LAYER_MAP
        else:
            cfg["head"][-1][0] = [16, 19, 10, 2, 0]
            cfg.pop("pretrained_layer_map", None)
        cfg["head"][-1][2] = "SegmentCitrusEV7"
        cfg["head"][-1][3] = head_args(p2, deform, pmce, boundary)
        # mask_ratio/nwd/copy_paste stay on the locked protocol via RUN_OVERRIDES.
        output.append(cfg)
    return output


if __name__ == "__main__":
    out = ROOT / "0_orange_yaml/E_V7_series"
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        path = out / f"{name}.yaml"
        if path.exists():
            raise FileExistsError(path)
        header = (
            "# E V7 official YOLO YAML. Four architecture factors (p2/deform/pmce/boundary); "
            "see docs/E_V7_RECONSTRUCTION.md.\n"
        )
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(path)
