"""refine_labels_sam.py — 用 SAM2 精修"估计标注"的远处小果多边形（基础模型数据引擎）.

动机：远处模糊小果为估计式标注（用户自述"凭感觉标"），审查已发现 483 张数量不一致；
SAM2 以现有粗框做 prompt 可产出像素级精确掩码，把低质量多边形升级为高质量标注。
流程：
  1) 读原 YOLO-seg 多边形 → 取 bbox 作为 SAM2 的 box prompt（每图批量）；
  2) 只精修短边 < --max-side（默认 96px 原生 ≈ 640 下 20px）的小实例——大果人工标注已够准；
  3) SAM 掩码 → 最大连通域 → cv2 多边形化 → 与原多边形算 IoU；
  4) IoU ∈ [--min-iou, 1) 时采用 SAM 版（太低=SAM 可能分错对象，保留原标注防翻车）；
  5) 写入 labels_samrefined/（原目录不动），输出 refine_report.csv 逐实例记录。

依赖 Ultralytics 内置 SAM（权重自动下载；服务器 GPU 上跑，965 图约 30-60 分钟）：
    python refine_labels_sam.py --data E:/mastercode/data/orange_yolo --split train --model sam2.1_b.pt
可换 --model mobile_sam.pt 快速试跑；--prompt box_point 启用"框+质心双 prompt"（更稳但更慢）。
产出的新标签目录经人工抽查后，复制/软链为正式 labels 即可训练对照
（消融行：原标注 vs SAM 精修标注，量化"标注质量"这一变量——论文中少见的数据侧消融）。

文献预警与对策（theme11 调研，档案见 3_研究生/文献调研_远距离小目标_20260726/theme11_foundation.md）：
- SAM 在同色伪装场景（绿果贴绿叶=COD 情形）性能大跌（arXiv:2304.04709; MIR 2024,
  doi:10.1007/s11633-023-1385-0 含农业实证）→ 绝不能用 segment-everything 自动模式，
  必须用已有粗标注做 box(+质心) prompt——本脚本即此设计；
- SAM 类别无关且会过分割（按高光/阴影把一个果切两块，S⁴M arXiv:2504.05301 点名）
  → 取最大连通域 + IoU 闸门回退（SAMST arXiv:2507.11994 的 Threshold Filter 思想）；
- 论文创新点方向：固定 IoU 阈值会整批丢弃远处低置信样本——**分尺度精修可信度判别器**
  （借 Consistent-Teacher arXiv:2209.01589 的 GMM 动态阈值）为文献缺口，可作数据引擎创新。
先例定位：SAM 精修噪声伪标签已有顶会背书（SAM_WSSS arXiv:2305.05803; SemiRES ICML 2024
arXiv:2406.01451），农业"基础模型标注→轻量 YOLO"已跑通（SDM-D arXiv:2411.16196），
但**精修已有低质量人工估计标注 + 绿色幼果场景**无先例——本工具的切口。

Reference: SAM (Kirillov et al., ICCV 2023, arXiv:2304.02643); SAM 2 (Ravi et al., 2024, arXiv:2408.00714).
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np


def poly_to_mask(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    pts = np.round(poly_norm.reshape(-1, 2) * [w, h]).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m


def mask_to_poly(mask: np.ndarray, eps_frac: float = 0.004) -> np.ndarray | None:
    """二值掩码 → 最大连通域外轮廓 → 归一化多边形 (N,2)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = eps_frac * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
    if len(c) < 3:
        return None
    h, w = mask.shape
    return c / [w, h]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/mastercode/data/orange_yolo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="sam2.1_b.pt", help="sam2.1_b.pt / sam2.1_s.pt / mobile_sam.pt")
    ap.add_argument("--max-side", type=float, default=96.0, help="只精修原生短边 < 此值(px) 的实例")
    ap.add_argument("--min-iou", type=float, default=0.35, help="SAM 掩码与原标注 IoU 低于此值时保留原标注")
    ap.add_argument("--prompt", default="box", choices=["box", "box_point"],
                    help="box=批量框 prompt（快）；box_point=每实例框+质心双 prompt（同色伪装场景更稳，慢）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import SAM  # 延迟导入（权重自动下载）

    sam = SAM(args.model)
    img_dir = os.path.join(args.data, args.split, "images")
    lbl_dir = os.path.join(args.data, args.split, "labels")
    out_dir = os.path.join(args.data, args.split, "labels_samrefined")
    os.makedirs(out_dir, exist_ok=True)

    rows, n_ref, n_keep = [], 0, 0
    files = sorted(os.listdir(img_dir))
    if args.limit:
        files = files[: args.limit]
    for fi, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(os.path.join(img_dir, fn))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path, encoding="utf-8") as f:
            lines = [ln.split() for ln in f.read().strip().splitlines() if ln.strip()]

        polys = [np.array(ln[1:], dtype=np.float64) for ln in lines]
        classes = [ln[0] for ln in lines]
        boxes, todo = [], []
        for i, p in enumerate(polys):
            pts = p.reshape(-1, 2) * [w, h]
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            if min(x1 - x0, y1 - y0) < args.max_side:
                boxes.append([x0, y0, x1, y1])
                todo.append(i)
        new_polys = list(polys)
        if boxes:
            if args.prompt == "box_point":
                # 逐实例 框+质心 双 prompt：同色伪装场景下质心正点击可显著降低 SAM 跑偏概率
                masks = []
                for (x0, y0, x1, y1), i in zip(boxes, todo):
                    pts_i = polys[i].reshape(-1, 2) * [w, h]
                    cx, cy = float(pts_i[:, 0].mean()), float(pts_i[:, 1].mean())
                    r = sam(img, bboxes=[[x0, y0, x1, y1]], points=[[cx, cy]], labels=[1],
                            device=args.device, verbose=False)[0]
                    m = r.masks.data[0].cpu().numpy() if r.masks is not None and len(r.masks.data) else np.zeros((h, w))
                    masks.append(m)
                masks = np.stack(masks) if masks else np.zeros((0, h, w))
            else:
                res = sam(img, bboxes=boxes, device=args.device, verbose=False)[0]
                masks = res.masks.data.cpu().numpy() if res.masks is not None else np.zeros((0, h, w))
            for j, i in enumerate(todo):
                if j >= len(masks):
                    break
                sam_mask = cv2.resize(masks[j].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                old_mask = poly_to_mask(polys[i], w, h)
                iou = mask_iou(sam_mask > 0, old_mask > 0)
                new_p = mask_to_poly(sam_mask) if iou >= args.min_iou else None
                if new_p is not None:
                    new_polys[i] = new_p.reshape(-1)
                    n_ref += 1
                else:
                    n_keep += 1
                rows.append((f"{args.split}/{fn}", i, round(iou, 3), "refined" if new_p is not None else "kept"))
        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            for cls, p in zip(classes, new_polys):
                coords = " ".join(f"{v:.6f}" for v in np.asarray(p).reshape(-1))
                f.write(f"{cls} {coords}\n")
        if fi % 20 == 0:
            print(f"[{fi}/{len(files)}] refined={n_ref} kept={n_keep}", flush=True)

    with open(os.path.join(out_dir, "refine_report.csv"), "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["image", "instance_idx", "iou_old_vs_sam", "action"])
        wcsv.writerows(rows)
    print(f"\ndone: {n_ref} refined, {n_keep} kept (low IoU) -> {out_dir}")
    print("下一步：人工抽查 20 张对比图无误后，将 labels_samrefined 软链/复制为 labels 跑'标注质量'消融行。")


if __name__ == "__main__":
    main()
