"""Forward-pass vs ground-truth visual audit — built to surface 漏标 (missing labels).

For the densest images in a split it renders a 3-panel comparison:
    [ Original | GT masks (green) | Predictions (red; YELLOW box = pred with NO matching GT = 漏标 candidate) ]
and prints per-image counts (GT / pred / 漏标-candidate / missed-GT).

A confident prediction sitting on a real fruit that has NO green GT next to it is the
signature of a missing annotation. A green GT with no red prediction is a model miss (hard case).

Run:
    python vis_pred_vs_gt.py --weights 1_results/001_yolo11n_seg_AdamW/weights/best.pt --split val --num 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA_ROOT = Path(r"E:/mastercode/data/test")
GREEN, RED, YELLOW, WHITE = (60, 200, 60), (40, 40, 220), (0, 215, 255), (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--num", type=int, default=12, help="How many of the DENSEST images to render.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.2, help="pred-vs-GT box IoU below this = 漏标 candidate.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--panel-w", type=int, default=900, help="width each panel is scaled to.")
    return p.parse_args()


def load_gt_boxes_polys(label_path: Path, w: int, h: int):
    """Return (boxes[x1,y1,x2,y2], polygons[list of Nx2 px]) from a YOLO-seg label file."""
    boxes, polys = [], []
    if not label_path.exists():
        return boxes, polys
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        xy = np.array(v[1:], dtype=float).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        polys.append(xy.astype(np.int32))
        boxes.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
    return boxes, polys


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw_polys(img, polys, color, alpha=0.35):
    ov = img.copy()
    for p in polys:
        cv2.fillPoly(ov, [p], color)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    for p in polys:
        cv2.polylines(img, [p], True, color, 3)
    return img


def scale(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h))


def main():
    a = parse_args()
    model = YOLO(a.weights)
    img_dir = DATA_ROOT / "images" / a.split
    lbl_dir = DATA_ROOT / "labels" / a.split
    out_dir = Path(a.weights).parents[1] / "pred_vs_gt" / a.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # rank images by GT instance count -> densest first (best chance to expose 漏标)
    ranked = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        n = len((lbl_dir / f"{img_path.stem}.txt").read_text().splitlines()) \
            if (lbl_dir / f"{img_path.stem}.txt").exists() else 0
        ranked.append((n, img_path))
    ranked.sort(reverse=True)
    chosen = [p for _, p in ranked[: a.num]]

    tot_gt = tot_pred = tot_leak = tot_miss = 0
    print(f"{'image':42} {'GT':>4} {'pred':>5} {'漏标?':>6} {'miss':>5}")
    for img_path in chosen:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_boxes, gt_polys = load_gt_boxes_polys(lbl_dir / f"{img_path.stem}.txt", w, h)

        r = model.predict(str(img_path), conf=a.conf, imgsz=a.imgsz, device=a.device,
                          retina_masks=True, verbose=False)[0]
        pred_polys = [p.astype(np.int32) for p in r.masks.xy] if r.masks is not None else []
        pred_boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []

        # panels
        p_orig = img.copy()
        p_gt = draw_polys(img.copy(), gt_polys, GREEN)
        p_pred = img.copy()
        p_pred = draw_polys(p_pred, pred_polys, RED)

        leak = 0
        for bb, cf in zip(pred_boxes, confs):
            best = max((iou(bb, g) for g in gt_boxes), default=0.0)
            is_leak = best < a.iou_match
            leak += int(is_leak)
            x1, y1, x2, y2 = map(int, bb)
            if is_leak:  # confident pred with no GT -> 漏标 candidate
                cv2.rectangle(p_pred, (x1, y1), (x2, y2), YELLOW, 6)
                cv2.putText(p_pred, f"leak? {cf:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, YELLOW, 3)
        miss = sum(1 for g in gt_boxes if max((iou(g, b) for b in pred_boxes), default=0.0) < a.iou_match)

        for p, tag in [(p_orig, "Original"), (p_gt, f"GT (green)  n={len(gt_boxes)}"),
                       (p_pred, f"Pred (red) leak?(yellow)={leak}")]:
            cv2.rectangle(p, (0, 0), (p.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(p, tag, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3)
        canvas = cv2.hconcat([scale(p_orig, a.panel_w), scale(p_gt, a.panel_w), scale(p_pred, a.panel_w)])
        cv2.imwrite(str(out_dir / f"{img_path.stem}_cmp.jpg"), canvas)

        tot_gt += len(gt_boxes); tot_pred += len(pred_boxes); tot_leak += leak; tot_miss += miss
        print(f"{img_path.name[:42]:42} {len(gt_boxes):>4} {len(pred_boxes):>5} {leak:>6} {miss:>5}")

    print(f"\nTOTAL  GT={tot_gt}  pred={tot_pred}  漏标候选={tot_leak}  漏检(GT无预测)={tot_miss}")
    print(f"-> comparisons saved to {out_dir}")


if __name__ == "__main__":
    main()
