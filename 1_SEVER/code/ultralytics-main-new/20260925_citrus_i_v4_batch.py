"""I V4 foreground sequential training using the common fixed-protocol engine."""
import importlib.util

from citrus_i_v4_suite import FACTORS, NAMES, ROOT, RUN_OVERRIDES, SUITES, YAML_DIR, select_names


def main():
    spec = importlib.util.spec_from_file_location("_citrus_i_v4_engine", ROOT / "20260922_citrus_i_v2_batch.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    for key, value in dict(FACTORS=FACTORS, NAMES=NAMES, RUN_OVERRIDES=RUN_OVERRIDES,
                           SUITES=SUITES, YAML_DIR=YAML_DIR, select_names=select_names).items():
        setattr(engine, key, value)
    engine.SERIES_LABEL = "CITRUS-I-V4"
    engine.IMPLEMENTATION_REVISION = "IV4_20260925_supervised_scene_layers_parallel_neck"
    engine.ASSIGNMENT_DESC = "unchanged standard TAL; no P2 detection or NWD assignment"
    engine.INPUT_DESC = "unchanged V2 source-balanced multiscale input, identical validation and crop budgets"
    engine.EXTRA_SOURCE_FILES = tuple(ROOT / p for p in (
        "20260925_citrus_i_v4_batch.py", "RUN_CITRUS_I_V4.py", "citrus_i_v4_suite.py",
        "scripts/generate_citrus_i_v4_yaml.py", "ultralytics/nn/modules/citrus_i_v4.py",
        "ultralytics/utils/citrus_i_v4_loss.py", "ultralytics/nn/modules/citrus_i_v3.py",
        "protocols/citrus_i_v4.yaml",
    ))
    # The inherited fallback output path is I_V2: override it explicitly when no project is supplied.
    original_parse = engine.parse_args

    def parse_args():
        args = original_parse()
        if args.project is None:
            args.project = ROOT / "1_results/I_V4" / f"CITRUS_IV4_{args.suite.upper()}_{args.epochs}EP"
        return args

    engine.parse_args = parse_args
    engine.main()


if __name__ == "__main__":
    main()
