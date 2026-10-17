"""Print canonical compact YAML payloads; the editor installs them with apply_patch."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from citrus_paper_suite import configurations


def compact(cfg):
    lines = ["# Paper control only; same training protocol for every arm."]
    for key, value in cfg.items():
        if key in {"backbone", "head"}:
            lines.append(key + ":")
            for layer in value:
                lines.append("  - " + yaml.safe_dump(layer, default_flow_style=True, width=10000).strip())
        else:
            lines.append(yaml.safe_dump({key: value}, sort_keys=False, width=10000).strip())
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(json.dumps({name: compact(cfg) for name, cfg in configurations().items()}))
