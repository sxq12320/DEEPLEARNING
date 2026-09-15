"""Generate checked-in V6 YAMLs; identical architectures intentionally repeat for training-factor ablations."""

from copy import deepcopy

import yaml

from citrus_e_v6_suite import FACTORS, NAMES, PHASE_MODELS, ROOT


def configs():
    base = yaml.safe_load(
        (ROOT / "0_orange_yaml/E_V5_series/V5_01_fine.yaml").read_text(encoding="utf-8")
    )
    output = []
    for name, (fine, _nwd, _cp) in zip(NAMES, FACTORS):
        cfg = deepcopy(base)
        cfg["head"][-1][0] = [16, 19, 10, 2, 0] if fine else [16, 19, 10, 2]
        cfg["head"][-1][2] = "SegmentCitrusEV6"
        cfg["head"][-1][3] = ["nc", 32, 256, 16, False, 0.0, "phase" if name in PHASE_MODELS else bool(fine)]
        # mask_ratio/nwd/copy_paste live in RUN_OVERRIDES, owned by the runner.
        output.append(cfg)
    return output


if __name__ == "__main__":
    out = ROOT / "0_orange_yaml/E_V6_series"
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in zip(NAMES, configs()):
        path = out / f"{name}.yaml"
        if path.exists():
            raise FileExistsError(path)
        header = (
            "# E V6 official YOLO YAML. Training factors live in RUN_CITRUS_E_V6.py; "
            "see docs/E_V6_RECONSTRUCTION.md.\n"
        )
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(path)
