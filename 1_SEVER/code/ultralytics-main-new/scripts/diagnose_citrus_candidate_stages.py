"""Read-only pre-NMS -> score -> post-NMS box -> mask failure decomposition.

GT is diagnostic only, never injected into the inference path. Post-NMS greedy
one-to-one box matching couples mask evaluation to that same prediction ID.
"""

# ruff: noqa: E402
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy
from scripts.review_sage_v4r import greedy_pairs


class StageValidator(SegmentationValidator):
    def postprocess(self, preds):
        raw = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        assert raw.shape[0] == 1, "Diagnostic is explicitly batch1"
        self.raw_boxes = xywh2xyxy(raw[0, :4].T).clone()
        self.raw_scores = raw[0, 4].clone()  # this dataset is single class
        return super().postprocess(preds)

    def _process_batch(self, preds, batch):
        result = super()._process_batch(preds, batch)
        gt = batch["masks"].float()
        if not len(gt):
            return result
        raw_iou = box_iou(batch["bboxes"], self.raw_boxes)
        post_iou = box_iou(batch["bboxes"], preds["bboxes"])
        kept = preds["conf"] >= 0.25
        kept_indices = torch.where(kept)[0]
        pairs = greedy_pairs(post_iou[:, kept].cpu().numpy(), 0.5)
        matches = {int(g): int(kept_indices[p]) for g, p in pairs}
        for i in range(len(gt)):
            geom = raw_iou[i] >= 0.5
            max_score = float(self.raw_scores[geom].max()) if geom.any() else 0.0
            pred_id = matches.get(i)
            miou = None
            if not geom.any():
                reason = "no_raw_box_iou50"
            elif max_score < 0.25:
                reason = "raw_box_present_score_low"
            elif pred_id is None:
                reason = "nms_limit_or_gt_competition"
            else:
                mask = preds["masks"][pred_id].float()
                intersection = (mask * gt[i]).sum()
                miou = float(intersection / (mask.sum() + gt[i].sum() - intersection).clamp_min(1))
                reason = "matched_box_bad_mask" if miou < 0.5 else "success"
            self.stage_records.append(
                dict(
                    image=Path(batch["im_file"]).name,
                    gt_index=i,
                    area=float(gt[i].sum() * 16),
                    bucket=reason,
                    max_raw_score=max_score,
                    matched_mask_iou=miou,
                )
            )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    validator = StageValidator(
        args=dict(
            data=args.data,
            device="cpu",
            imgsz=640,
            batch=1,
            workers=0,
            plots=False,
            half=False,
            conf=0.001,
            iou=0.7,
            max_det=300,
            rect=False,
            overlap_mask=True,
            mask_ratio=4,
        ),
        save_dir=args.output.parent / (args.output.stem + "_eval"),
    )
    validator.stage_records = []
    validator(model=YOLO(args.weights).model)
    records = validator.stage_records
    groups = {"all": records, "tiny": [r for r in records if r["area"] < 256]}
    summary = {key: dict(n=len(rows), buckets=dict(Counter(r["bucket"] for r in rows))) for key, rows in groups.items()}
    payload = dict(
        summary=summary,
        records=records,
        weights=args.weights,
        limits="LOCAL CPU FP32; fixed conf .25; box-first one-to-one matching, not official mask AP. "
        "NMS bucket includes GT competition. Raster stride4 area. No causal proof about training assignment.",
    )
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
