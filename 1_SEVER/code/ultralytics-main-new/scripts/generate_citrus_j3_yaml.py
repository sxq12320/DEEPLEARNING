"""Generate J V3 YAMLs (one layer per line). Every arm deploys the YOLO11n-seg graph identical to J20/J00.

Run from the repository root: python scripts/generate_citrus_j3_yaml.py [--check]
Existing files with different content are never overwritten silently: regenerate only after deleting them yourself.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citrus_j3_suite import DESCRIPTIONS, FACTORS, NAMES, YAML_DIR  # noqa: E402
from scripts.generate_citrus_j2_yaml import BODY  # noqa: E402

HEAD = "  - [[16, 19, 22], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5): unchanged deployable head\n"


def render(name):
    runner = ", ".join(f"{k}={v}" for k, v in FACTORS[name].items())
    return (
        f"# {name} | {DESCRIPTIONS[name]}\n"
        f"# Runner factors (citrus_j3_suite.FACTORS): {runner}.\n"
        "# Deployed graph = YOLO11n-seg (identical to J20/J00); J V3 changes TRAINING supervision only. No accuracy claim.\n"
        + BODY
        + HEAD
    )


def main(check=False):
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    stale, conflicts = [], []
    for name in NAMES:
        path, text = YAML_DIR / (name + ".yaml"), render(name)
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            continue
        if check:
            stale.append(path.name)
        elif current is not None:
            conflicts.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        raise SystemExit(f"Stale J V3 YAMLs: {stale}")
    if conflicts:
        raise SystemExit(f"Refusing to overwrite edited J V3 YAMLs: {conflicts}")
    print(("Checked " if check else "Wrote/verified ") + f"{len(NAMES)} YAMLs in {YAML_DIR}")


if __name__ == "__main__":
    main("--check" in sys.argv)
