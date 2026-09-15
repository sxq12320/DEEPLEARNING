"""Deterministic E V2 architecture manifest generator; never overwrites differing files."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def configs():
    base = yaml.safe_load((ROOT / "0_orange_yaml/E_series/E01_sliced_control.yaml").read_text())
    contrast = deepcopy(base)
    contrast["head"][-1][2] = "SegmentCitrusEV2"
    hub = deepcopy(base)
    hub.update(pretrained_layer_map={15: 23}, pretrained_map_family="citrus_e_v2_hub")
    hub["head"] = [
        [[4, 6, 10], 1, "EV2ContextHub", [256]],
        [[4, 11], 1, "EV2ContextInject", [256]],
        [[6, 11], 1, "EV2ContextInject", [512]],
        [[10, 11], 1, "EV2ContextInject", [1024]],
        [[12, 13, 14, 2], 1, "SegmentCitrusSAGEV5", ["nc", 32, 256, 16, True, False, 0, 0]],
    ]
    rep = deepcopy(base)
    for i in (2, 4, 6, 8):
        # One mixer in high-resolution stages, two deeper down at nano scale.
        # Every backbone CSP stage is replaced; reserve compute for C2 details.
        c2 = base["backbone"][i][3][0]
        rep["backbone"][i] = [-1, 2 if i in (2, 4) else 4, "EV2RepStage", [c2]]
    for i in (3, 5, 7):
        rep["backbone"][i] = [-1, 1, "EV2Down", [base["backbone"][i][3][0]]]
    rep["backbone"][10] = [-1, 2, "EV2RepStage", [1024]]
    rep_hub = deepcopy(hub)
    rep_hub["backbone"] = deepcopy(rep["backbone"])
    full = deepcopy(rep_hub)
    full["head"][-1][2] = "SegmentCitrusEV2"
    return [base, contrast, hub, rep, rep_hub, full, deepcopy(full), deepcopy(full)]


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    from citrus_e_v2_suite import NAMES, YAML_DIR

    YAML_DIR.mkdir(parents=True, exist_ok=True)
    for name, config in zip(NAMES, configs()):
        path = YAML_DIR / (name + ".yaml")
        text = "# E V2: see docs/CITRUS_E_V2.md for architecture, initialization and controlled input factors.\n"
        for key, value in config.items():
            if key in {"backbone", "head"}:
                text += key + ":\n"
                for layer in value:
                    text += "  - " + yaml.safe_dump(layer, default_flow_style=True, width=150).strip() + "\n"
            else:
                text += yaml.safe_dump({key: value}, sort_keys=False, default_flow_style=False)
        if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(path)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
