"""Validate the converted COCO dataset used by Torchvision Mask R-CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision_maskrcnn_common import validate_prepared_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--class-name", action="append", default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    report = validate_prepared_dataset(
        dataset_root=args.dataset,
        splits=args.splits,
        class_names=args.class_name or ["orange_immature"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
