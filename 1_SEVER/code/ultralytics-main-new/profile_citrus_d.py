"""Export parameter, FLOP, and explicit pretrained-transfer profiles for CitrusD."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "D_series"
OUTPUT = ROOT / "1_results" / "_compatibility" / "citrus_d_profiles.csv"


def transferred_keys(model: SegmentationModel, source: dict[str, torch.Tensor]) -> set[str]:
    """Return direct and YAML-remapped target keys whose shapes match the checkpoint."""
    target = model.state_dict()
    loaded = set(intersect_dicts(source, target, exclude=()))
    for target_index, source_index in model.yaml.get("pretrained_layer_map", {}).items():
        source_prefix = f"model.{int(source_index)}."
        target_prefix = f"model.{int(target_index)}."
        for source_key, value in source.items():
            if not source_key.startswith(source_prefix):
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            if target_key in target and target[target_key].shape == value.shape:
                loaded.add(target_key)
    return loaded


def main() -> None:
    """Profile every D YAML at nc=1 and a 640-square input."""
    checkpoint = torch.load(ROOT / "yolo11n-seg.pt", map_location="cpu", weights_only=False)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    rows = []
    for yaml_path in sorted(YAML_DIR.glob("*.yaml")):
        model = SegmentationModel(yaml_path, ch=3, nc=1, verbose=False)
        target = model.state_dict()
        direct = set(intersect_dicts(source, target, exclude=()))
        loaded = transferred_keys(model, source)
        target_elements = sum(value.numel() for value in target.values())
        rows.append(
            {
                "model": yaml_path.stem,
                "yaml": str(yaml_path.relative_to(ROOT)),
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "gflops_640": get_flops(model, imgsz=640),
                "direct_pretrained_items": len(direct),
                "mapped_pretrained_items": len(loaded),
                "target_items": len(target),
                "direct_pretrained_element_ratio": sum(target[key].numel() for key in direct) / target_elements,
                "mapped_pretrained_element_ratio": sum(target[key].numel() for key in loaded) / target_elements,
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
            f"mapped transfer {row['mapped_pretrained_element_ratio']:.1%}"
        )
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
