"""One-epoch public-API pipeline smoke test on an explicitly selected tiny fixture.

This intentionally uses CPU/batch2/256/workers0/plotsFalse and is NOT a formal
accuracy experiment. It never changes the fixed protocol or original dataset.
"""

# ruff: noqa: E402 -- local checkout takes precedence for direct script execution
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from citrus_protocol import fixed_train_args
from citrus_sage_v5_suite import NAMES, YAML_DIR
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--only", default=",".join(NAMES[2:]))
    args = parser.parse_args()
    if args.project.exists():
        raise FileExistsError(args.project)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    names = args.only.split(",")
    if set(names) - set(NAMES):
        raise ValueError("Unknown smoke model")
    args.project.mkdir(parents=True)
    torch.set_num_threads(2)
    training = fixed_train_args()
    training.update(
        data=str(args.data.resolve()),
        project=str(args.project.resolve()),
        device="cpu",
        epochs=1,
        batch=2,
        imgsz=256,
        workers=0,
        plots=False,
        amp=False,
        seed=42,
        mosaic=0.0,
        close_mosaic=0,
        save=True,
        val=True,
    )
    (args.project / "SMOKE_NOT_FORMAL.json").write_text(
        json.dumps({"deviating_smoke_parameters": training, "models": names}, indent=2), encoding="utf-8"
    )
    for name in names:
        model = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment").load(str(ROOT / "yolo11n-seg.pt"))
        model.train(name=name, **training)
        assert (args.project / name / "weights/best.pt").is_file()
        print(f"SMOKE PASSED {name}", flush=True)


if __name__ == "__main__":
    main()
