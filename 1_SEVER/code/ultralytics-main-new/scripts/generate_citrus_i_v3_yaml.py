"""Generate ten index-safe I V3 YAMLs from the exact I20 control graph."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_i_v3_suite import FACTORS, NAMES, YAML_DIR  # noqa: E402

SOURCE = ROOT / "0_orange_yaml/I_V2_series/I20_corrected_control.yaml"


def build_yaml(name):
    """Insert a narrow stride-4/8 stream, remapping every graph edge explicitly."""
    factor = FACTORS[name]
    model = copy.deepcopy(yaml.safe_load(SOURCE.read_text(encoding="utf-8")))
    if factor["view"] < 0:
        model["head"][-1][3][7] = factor["assign"]
        model["head"][-1][3][9] = factor["tiny"]
        model["head"][-1][3][12] = factor["route"]
        return model

    view_channels = 1 if factor["view"] == 0 else 2
    old_layers = model["backbone"] + model["head"]
    old_backbone_size = len(model["backbone"])
    new_layers = [
        [-1, 1, "IV3InputTwin", [factor["view"]]],
        [0, 1, "Index", [3, 0]],
        [0, 1, "Index", [view_channels, 1]],
    ]
    source_map = {0: -1, 1: -1, 2: -1}
    index_map = {}
    gray_p2 = None
    backbone_end = None
    original_pretrained = model.get("pretrained_layer_map", {})

    def add(layer, pretrained=-1):
        index = len(new_layers)
        new_layers.append(layer)
        source_map[index] = pretrained
        return index

    def map_from(source, previous):
        if isinstance(source, list):
            return [map_from(item, previous) for item in source]
        return 1 if source == -1 and previous < 0 else index_map[previous if source == -1 else source]

    for old_index, old_layer in enumerate(old_layers):
        layer = copy.deepcopy(old_layer)
        layer[0] = map_from(layer[0], old_index - 1)
        new_index = add(layer, int(original_pretrained.get(old_index, old_index)))
        index_map[old_index] = new_index
        if old_index == 2:
            gray_s2 = add([2, 1, "Conv", [32, 3, 2]])
            gray_p2 = add([gray_s2, 1, "Conv", [64, 3, 2]])
            index_map[old_index] = add([[new_index, gray_p2], 1, "IV3AsymGrayFuse", [factor["gate"]]])
        if old_index == 5 and factor["p3"]:
            gray_p3 = add([gray_p2, 1, "Conv", [96, 3, 2]])
            index_map[old_index] = add([[new_index, gray_p3], 1, "IV3AsymGrayFuse", [factor["gate"]]])
        if old_index == old_backbone_size - 1:
            backbone_end = len(new_layers)

    model["backbone"] = new_layers[:backbone_end]
    model["head"] = new_layers[backbone_end:]
    model["head"][-1][3][7] = factor["assign"]
    model["head"][-1][3][9] = factor["tiny"]
    model["head"][-1][3][12] = factor["route"]
    model["pretrained_layer_map"] = source_map
    return model


def format_yaml(model):
    """Keep each YOLO layer on one line while preserving YAML semantics."""
    lines = []
    for key, value in model.items():
        if key in {"backbone", "head"}:
            lines.extend((f"{key}:", "  # [from, repeats, module, args]"))
            for layer in value:
                compact = yaml.safe_dump(layer, default_flow_style=True, sort_keys=False, width=10000).strip()
                if "\n" in compact:
                    raise ValueError(f"Could not format {key} layer on one line: {layer}")
                lines.append(f"  - {compact}")
        elif key == "scales":
            lines.append("scales:")
            for scale, settings in value.items():
                compact = yaml.safe_dump(settings, default_flow_style=True, width=10000).strip()
                lines.append(f"  {scale}: {compact}")
        else:
            lines.append(yaml.safe_dump({key: value}, sort_keys=False, width=10000).rstrip())
    result = "\n".join(lines) + "\n"
    if yaml.safe_load(result) != model:
        raise ValueError("Compact YAML changed the model configuration")
    return result


def main():
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    for name in NAMES:
        target = YAML_DIR / f"{name}.yaml"
        formatted = format_yaml(build_yaml(name))
        if target.exists() and yaml.safe_load(target.read_text(encoding="utf-8")) != yaml.safe_load(formatted):
            raise ValueError(f"Refusing to change model semantics: {target}")
        pending.append((target, formatted))
    for target, formatted in pending:
        target.write_text(formatted, encoding="utf-8")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
