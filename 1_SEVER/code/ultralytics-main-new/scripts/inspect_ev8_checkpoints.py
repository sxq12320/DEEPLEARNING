"""Verify returned checkpoints against the local V8 architecture, not only run names."""

import json
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/E_V9_REVIEW_20260915"


def main():
    torch.set_num_threads(2)
    rows = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))["v8"]
    results = []
    for row in rows:
        path = Path(row["path"]).parent / "weights/best_mask.pt"
        if not path.is_file():
            results.append(dict(name=row["name"], available=False))
            continue
        api = YOLO(str(path), verbose=False)
        head = api.model.model[-1]
        results.append(
            dict(
                name=row["name"],
                available=True,
                head=type(head).__name__,
                stage6=type(api.model.model[6]).__name__,
                neck19=type(api.model.model[19]).__name__,
                fine_mode=head.fine_mode,
                strides=head.stride.tolist(),
                boundary_gain=head.boundary_gain,
                neighbor_gain=head.neighbor_gain,
                params=sum(p.numel() for p in api.model.parameters()),
                checkpoint_epoch=api.ckpt.get("epoch"),
                task="Current classes used to deserialize historical checkpoint; not historical Python provenance",
            )
        )
    (OUT / "returned_checkpoint_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
