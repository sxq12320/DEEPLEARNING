"""Same frozen-weight score-calibration ablation; reuse unchanged E V2 mask metrics."""

import argparse

from eval_citrus_e_v2 import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("weights", "data", "output"):
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw-quality", action="store_true", help="Disable mask-quality score multiplication")
    args = vars(parser.parse_args())
    args["quality_calibration"] = False if args.pop("raw_quality") else None
    evaluate(**args)
