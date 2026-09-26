"""Build compact I V4 graphs. Emit patches; never silently overwrite old models."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from citrus_i_v4_suite import FACTORS, NAMES, YAML_DIR
from scripts.generate_citrus_i_v3_yaml import build_yaml as previous_yaml, format_yaml


def build_yaml(name):
    factor = FACTORS[name]
    model = copy.deepcopy(previous_yaml("I33_contrast_p23" if factor["gray"] else "I30_rgb_control"))
    model["nc"] = 1
    if factor["mode"] < 0:
        return model
    # Explicit raw encoder indices, no serial top-down/bottom-up neck.
    inputs = [13, 18, 22, 8, 3, 18, 17] if factor["gray"] else [5, 10, 14, 2, 0, 10, 9]
    head_index = len(model["backbone"])
    model["head"] = [[inputs, 1, "SegmentCitrusIV4",
                      ["nc", 32, 256, 16, factor["mode"], factor["steps"], factor["aux"]]]]
    model["pretrained_layer_map"] = {i: original for i, original in model["pretrained_layer_map"].items()
                                     if i < head_index}
    model["pretrained_layer_map"][head_index] = 23
    model["pretrained_map_family"] = "IV4_layered_scene_gray" if factor["gray"] else "IV4_layered_scene"
    return model


def main():
    print("*** Begin Patch")
    for name in NAMES:
        target = YAML_DIR / f"{name}.yaml"
        if target.exists():
            raise FileExistsError(target)
        print(f"*** Add File: {target.as_posix()}")
        for line in format_yaml(build_yaml(name)).splitlines():
            print("+" + line)
    print("*** End Patch")


if __name__ == "__main__":
    main()
