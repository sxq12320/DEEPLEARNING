"""Export deterministic parameter/FLOP and pretrained-transfer profiles for CitrusB YAMLs."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "B_series"
OUTPUT = ROOT / "1_results" / "_compatibility" / "citrus_b_profiles.csv"


def main() -> None:
    """Profile all YAMLs at nc=1 and a 640-square input."""
    checkpoint = torch.load(ROOT / "yolo11n-seg.pt", map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    rows = []
    for yaml_path in sorted(YAML_DIR.glob("*.yaml")):
        model = SegmentationModel(yaml_path, ch=3, nc=1, verbose=False)
        target = model.state_dict()
        matched = intersect_dicts(source, target, exclude=())
        rows.append(
            {
                "model": yaml_path.stem,
                "yaml": str(yaml_path.relative_to(ROOT)),
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "gflops_640": get_flops(model, imgsz=640),
                "direct_pretrained_items": len(matched),
                "target_items": len(target),
                "direct_pretrained_element_ratio": sum(target[key].numel() for key in matched)
                / sum(value.numel() for value in target.values()),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['model']}: {row['params'] / 1e6:.3f}M, {row['gflops_640']:.2f} GFLOPs, "
            f"direct transfer {row['direct_pretrained_element_ratio']:.1%}"
        )


if __name__ == "__main__":
    main()
