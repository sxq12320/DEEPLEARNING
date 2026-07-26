"""predict_citrus_sliced.py — 切片推理（SAHI 思想）：把 3072 原图切块推理再合并，找回远处小果.

定位（theme8 调研结论）：**精度上界基线 + 远景小果检出评估工具**，不是部署主方案
（3x3 切片 ≈ 10x 推理代价）。数据依据：<32px@640 小果在原图有 93px 中位信息，
640 全图推理把 79% 线性分辨率扔掉了——切片推理直接量化"分辨率损失值多少 mAP"。
Reference: Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object
Detection" (ICIP 2022, doi:10.1109/ICIP46576.2022.9897990).

用法:
    python predict_citrus_sliced.py --weights 1_results/ORANGE_WUXI_SEG/<run>/weights/best.pt \
        --source E:/mastercode/data/orange_yolo/test/images --tiles 3 --overlap 0.2 --limit 10
输出: <run 同级>/sliced_pred/ 下每图对比图（左: 640 全图推理 | 右: 切片合并推理）+ counts.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO  # noqa: E402


def tile_grid(w: int, h: int, n: int, overlap: float):
    """N x N 重叠切片坐标 (x0, y0, x1, y1)."""
    step_x, step_y = w / n, h / n
    pad_x, pad_y = step_x * overlap / 2, step_y * overlap / 2
    for i in range(n):
        for j in range(n):
            x0 = max(0, int(j * step_x - pad_x))
            y0 = max(0, int(i * step_y - pad_y))
            x1 = min(w, int((j + 1) * step_x + pad_x))
            y1 = min(h, int((i + 1) * step_y + pad_y))
            yield x0, y0, x1, y1


def run_one(model: YOLO, img: np.ndarray, tiles: int, overlap: float, imgsz: int, conf: float):
    """全图 + 切片推理，返回合并后的 (boxes_xyxy, confs) 与全图基线数量."""
    h, w = img.shape[:2]
    r_full = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    n_full = 0 if r_full.boxes is None else len(r_full.boxes)

    all_boxes, all_confs = [], []
    if r_full.boxes is not None and len(r_full.boxes):
        all_boxes.append(r_full.boxes.xyxy.cpu())
        all_confs.append(r_full.boxes.conf.cpu())
    for x0, y0, x1, y1 in tile_grid(w, h, tiles, overlap):
        r = model.predict(img[y0:y1, x0:x1], imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.cpu() + torch.tensor([x0, y0, x0, y0], dtype=torch.float32)
        all_boxes.append(b)
        all_confs.append(r.boxes.conf.cpu())

    if not all_boxes:
        return torch.zeros(0, 4), torch.zeros(0), n_full
    boxes = torch.cat(all_boxes)
    confs = torch.cat(all_confs)
    from torchvision.ops import nms

    keep = nms(boxes, confs, iou_threshold=0.55)
    return boxes[keep], confs[keep], n_full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="图像目录或单图")
    ap.add_argument("--tiles", type=int, default=3, help="N x N 切片数（3072 原图建议 3）")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="最多处理图像数，0=全部")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = args.source
    files = (
        [src]
        if os.path.isfile(src)
        else [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if args.limit:
        files = files[: args.limit]
    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(args.weights)), "sliced_pred")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        boxes, confs, n_full = run_one(model, img, args.tiles, args.overlap, args.imgsz, args.conf)
        vis = img.copy()
        for (x0, y0, x1, y1), c in zip(boxes.numpy().astype(int), confs.numpy()):
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), max(2, img.shape[0] // 800))
        gain = len(boxes) - n_full
        name = os.path.basename(p)
        cv2.putText(vis, f"full:{n_full}  sliced:{len(boxes)}  (+{gain})", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, img.shape[0] / 1200, (0, 200, 255), max(2, img.shape[0] // 700))
        cv2.imwrite(os.path.join(out_dir, name), vis)
        rows.append((name, n_full, len(boxes), gain))
        print(f"{name}: full={n_full} sliced={len(boxes)} (+{gain})")

    with open(os.path.join(out_dir, "counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_full_640", "n_sliced", "gain"])
        w.writerows(rows)
    total_full = sum(r[1] for r in rows)
    total_sliced = sum(r[2] for r in rows)
    print(f"\nTOTAL: full={total_full} sliced={total_sliced} (+{total_sliced - total_full}, "
          f"{(total_sliced / max(total_full, 1) - 1) * 100:.1f}% more detections) -> {out_dir}")


if __name__ == "__main__":
    main()
