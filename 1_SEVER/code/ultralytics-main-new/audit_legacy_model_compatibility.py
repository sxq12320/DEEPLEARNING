"""Audit whether every archived citrus model YAML can use the standard Ultralytics model entry point."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_YAML_ROOT = ROOT / "0_orange_yaml"
DEFAULT_OUTPUT = ROOT / "1_results" / "_compatibility" / "all_series_yaml_compatibility"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml-root", type=Path, default=DEFAULT_YAML_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--forward-size", type=int, default=64, help="Square input used for an explicit eval forward.")
    parser.add_argument("--no-forward", action="store_true", help="Build only; model construction still initializes stride.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional checkpoint loaded through YOLO(yaml).load().")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def flatten_tensors(value) -> list[torch.Tensor]:
    """Return every tensor contained in a nested model output."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in flatten_tensors(item)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in flatten_tensors(item)]
    return []


def main() -> None:
    """Build and optionally forward every model YAML, then save a machine-readable compatibility matrix."""
    args = parse_args()
    yaml_root = args.yaml_root.resolve()
    weights = args.weights.resolve() if args.weights else None
    yaml_files = sorted(yaml_root.rglob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found under {yaml_root}")
    if weights is not None and not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    from ultralytics import YOLO

    records: list[dict[str, object]] = []
    for index, yaml_path in enumerate(yaml_files, start=1):
        started = time.perf_counter()
        record: dict[str, object] = {
            "yaml": yaml_path.relative_to(yaml_root).as_posix(),
            "status": "failed",
            "head": "",
            "params": 0,
            "weights_loaded": False,
            "output_tensors": 0,
            "seconds": 0.0,
            "error": "",
        }
        try:
            wrapper = YOLO(str(yaml_path), task="segment", verbose=False)
            if weights is not None:
                wrapper.load(str(weights))
                record["weights_loaded"] = True
            model = wrapper.model.eval()
            record["head"] = type(model.model[-1]).__name__
            record["params"] = sum(parameter.numel() for parameter in model.parameters())
            if not args.no_forward:
                with torch.no_grad():
                    outputs = model(torch.zeros(1, 3, args.forward_size, args.forward_size))
                tensors = flatten_tensors(outputs)
                if not tensors or not all(torch.isfinite(tensor).all() for tensor in tensors):
                    raise RuntimeError("Forward output is empty or contains non-finite values")
                record["output_tensors"] = len(tensors)
            record["status"] = "passed"
        except Exception as error:  # noqa: BLE001 - every YAML must be recorded, including integration errors.
            record["error"] = f"{type(error).__name__}: {error}"
            if args.fail_fast:
                raise
        finally:
            record["seconds"] = round(time.perf_counter() - started, 4)
            records.append(record)
            print(f"[{index:03d}/{len(yaml_files):03d}] {record['status']:<6} {record['yaml']} {record['error']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    csv_path = args.output.with_suffix(".csv")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    passed = sum(record["status"] == "passed" for record in records)
    print(f"Compatibility: {passed}/{len(records)} passed")
    print(f"JSON: {json_path}\nCSV:  {csv_path}")
    if passed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
