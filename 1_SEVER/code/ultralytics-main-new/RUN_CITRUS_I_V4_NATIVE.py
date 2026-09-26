"""Edit these constants; click VS Code's Run Python File button to train sequentially."""

from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path(__file__).resolve().parent
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
DEVICE = "1"
EPOCHS = 50  # Screen first; use 300 only after convergence/latency checks.
SEED = 42
PROJECT = "/data/sxq/results/I/I_V4/CITRUS_IV4_NATIVE_{}EP".format(EPOCHS)
ONLY = ""  # Empty = I48 through I53 in order; or e.g. "I48_native_set,I53_native_edge".
DRY_RUN = False
SOURCE_BALANCED = True  # Same global/coarse/fine training views as the earlier I_V4 protocol.


def main():
    path = ROOT / "20260925_citrus_i_v4_native_batch.py"
    spec = spec_from_file_location("citrus_i_v4_native_batch", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(DATA, DEVICE, PROJECT, EPOCHS, SEED, ONLY, DRY_RUN, SOURCE_BALANCED)


if __name__ == "__main__":
    main()
