"""I V3 foreground batch entry; reuses the verified V2 sequential training/evaluation engine."""

from __future__ import annotations

import importlib.util

from citrus_i_v3_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, SUITES, YAML_DIR, select_names


def main():
    """Run V3 in the current process, preserving V2 training and paired-eval semantics."""
    spec = importlib.util.spec_from_file_location("_citrus_i_v3_engine", ROOT / "20260922_citrus_i_v2_batch.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.FACTORS = FACTORS
    engine.NAMES = NAMES
    engine.RUN_OVERRIDES = RUN_OVERRIDES
    engine.SUITES = SUITES
    engine.YAML_DIR = YAML_DIR
    engine.select_names = select_names
    engine.SERIES_LABEL = "CITRUS-I-V3"
    engine.IMPLEMENTATION_REVISION = "IV3_20260923_asymmetric_achromatic_controlled"
    engine.ASSIGNMENT_DESC = "standard TAL, isolated I38 tiny assignment mix=0.2"
    engine.INPUT_DESC = "same V2 source-balanced multiscale views; RGB and achromatic branch share augmentation"
    engine.EXTRA_SOURCE_FILES = (
        ROOT / "20260923_citrus_i_v3_batch.py",
        ROOT / "RUN_CITRUS_I_V3.py",
        ROOT / "citrus_i_v3_suite.py",
        ROOT / "scripts/generate_citrus_i_v3_yaml.py",
        ROOT / "ultralytics/nn/modules/citrus_i_v3.py",
        ROOT / "protocols/citrus_i_v3.yaml",
    )
    engine.main()


if __name__ == "__main__":
    main()
