"""Read-only validation assignment probe; measures coverage, not new accuracy."""
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    hyp = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    dataset = YOLODataset(
        img_path=str(Path(cfg["path"]) / cfg["val"]),
        data=cfg,
        task="segment",
        imgsz=640,
        augment=False,
        hyp=hyp,
        batch_size=2,
        cache=False,
        rect=False,
    )
    model = YOLO(str(args.weights), verbose=False).model.float().eval()
    model.args = hyp
    criterion = model.init_criterion()
    rows = []
    with torch.inference_mode():
        # Evenly spread across sorted file list; no selection by observed model errors.
        indices = torch.linspace(0, len(dataset) - 1, min(args.limit, len(dataset))).long().tolist()
        for a in range(0, len(indices), 2):
            batch = dataset.collate_fn([dataset[i] for i in indices[a : a + 2]])
            batch["img"] = batch["img"].float() / 255
            pred = model(batch["img"])[1]
            assigned, _, _ = criterion.get_assigned_targets_and_loss(pred, batch)
            fg, ids, _, _, _ = assigned
            for i, path in enumerate(batch["im_file"]):
                boxes = batch["bboxes"][batch["batch_idx"] == i]
                h, w = batch["img"].shape[-2:]
                gt = torch.arange(1, len(boxes) + 1)[:, None, None]
                visible = ((batch["masks"][i] == gt).sum((1, 2)) * 4).tolist()
                counts = torch.bincount(ids[i, fg[i]], minlength=len(boxes)).tolist()
                for j, (b, area, n) in enumerate(zip(boxes, visible, counts)):
                    rows.append(
                        dict(
                            image=Path(path).name,
                            gt_index=j,
                            mask_area_input=area,
                            box_area_input=float(b[2] * b[3] * h * w),
                            assigned_anchors=n,
                        )
                    )

    def summarize(selected):
        return dict(
            gt=len(selected),
            zero_positive=sum(r["assigned_anchors"] == 0 for r in selected),
            mean_anchors=sum(r["assigned_anchors"] for r in selected) / max(1, len(selected)),
        )

    payload = dict(
        weights=str(args.weights),
        data=str(args.data.resolve()),
        images=len(indices),
        total=summarize(rows),
        tiny=summarize([r for r in rows if 0 < r["mask_area_input"] < 256]),
        rows=rows,
        limitation=(
            "Validation frozen predictions, not training dynamics or proof of causality. "
            "Dataset membership must match returned experiment before interpreting this as its validation probe."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}))


if __name__ == "__main__":
    main()
