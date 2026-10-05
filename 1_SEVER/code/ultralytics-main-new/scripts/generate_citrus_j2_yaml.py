"""Generate J V2 YAMLs (one layer per line): retained gaze controls + structural ablations.

Run from the repository root: python scripts/generate_citrus_j2_yaml.py [--check]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citrus_j2_suite import DESCRIPTIONS, FACTORS, NAMES, YAML_DIR  # noqa: E402
from citrus_j2_structure_suite import STRUCTURE_NAMES, structural_graph  # noqa: E402

BODY = """nc: 1
scale: n
scales:
  n: [0.50, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32)
"""


def render(name):
    if name in STRUCTURE_NAMES:
        if name == "J29_i20_structure_anchor":
            original = ROOT / "0_orange_yaml/J_series/J01_i20_structure_anchor.yaml"
            return "# J29: preserved current I20 structural graph; scratch, no historical checkpoint replay.\n" + original.read_text(encoding="utf-8")
        backbone, neck = structural_graph(name)
        lines = [f"# {name}: {DESCRIPTIONS[name]}", "# Experimental candidate, no accuracy claim.",
                 "nc: 1", "scale: n", "scales:", "  n: [1.0, 1.0, 1024]"]
        for key, layers in (("backbone", backbone), ("head", neck)):
            lines.append(key + ":")
            for source, repeats, module, args in layers:
                source = json.dumps(source)
                payload = json.dumps(args)
                lines.append(f"  - [{source}, {repeats}, {module}, {payload}]")
        return "\n".join(lines) + "\n"
    f = FACTORS[name]
    if f["density"]:
        head = "  - [[16, 19, 22], 1, SegmentCitrusJ2, [nc, 32, 256, True, {}, 0.5]] # Segment + P3 small-fruit density\n"
        head = head.format("True" if f["detach"] else "False")
    else:
        head = "  - [[16, 19, 22], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5)\n"
    runner = ", ".join(f"{k}={v}" for k, v in f.items() if k not in ("density", "detach"))
    return (
        f"# {name} | {DESCRIPTIONS[name]}\n"
        f"# Runner factors (citrus_j2_suite.FACTORS): {runner}. Scratch initialization; no accuracy claim.\n" + BODY + head
    )


def main(check=False):
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for name in NAMES:
        path, text = YAML_DIR / (name + ".yaml"), render(name)
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        raise SystemExit(f"Stale J V2 YAMLs: {stale}")
    print(("Checked " if check else "Wrote ") + f"{len(NAMES)} YAMLs in {YAML_DIR}")


if __name__ == "__main__":
    main("--check" in sys.argv)
