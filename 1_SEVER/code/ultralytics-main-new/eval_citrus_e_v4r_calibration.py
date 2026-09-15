"""Evaluate the SAME quality checkpoint at predeclared score floors; no retraining.

Applicable to existing E33/E35/E36/E37 and new E V4R quality arms. Uses no GT
for selection. Three full validation passes cost time but do not alter training.
"""

import argparse
import json
from pathlib import Path

from eval_citrus_e_v2 import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("weights", "data", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for floor in (0.0, 0.5, 1.0):
        summaries[str(floor)] = evaluate(
            args.weights,
            args.data,
            output / f"floor_{floor}",
            device=args.device,
            limit=args.limit,
            quality_calibration=True,
            quality_floor=floor,
        )
    (output / "calibration_summary.json").write_text(
        json.dumps(
            dict(
                weights=args.weights,
                limit=args.limit,
                summary=summaries,
                warning="Exploratory validation comparison, not an unbiased test-set optimum",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
