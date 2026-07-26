"""analyze_citrus_dataset.py — 柑橘数据集量化体检：为"远处小果暗、糊、低对比"提供统计证据.

对每个实例（YOLO-seg 多边形标签）计算：
- 尺寸：640 输入下的等效 bbox 边长（min/max 边）、原生分辨率下的边长
- 亮度：HSV V 通道在实例掩码内的均值（0-255）
- 模糊度：实例 bbox 裁剪（放大到统一 64px）的 Laplacian 方差（越低越糊）
- 绿度对比：LAB a* 通道实例内 vs 周边背景环的差值绝对值（越低越"绿绿伪装"）
- 背景亮度差：实例 V 均值 - 背景环 V 均值（负值 = 果比背景暗）

按 640 等效尺寸分箱（<16 / 16-32 / 32-64 / 64-128 / >=128）聚合，输出：
- CSV 逐实例明细：0_orange_yaml/1_far_small/_dataset_stats.csv
- Markdown 报告：0_orange_yaml/1_far_small/_dataset_analysis.md

用法：python analyze_citrus_dataset.py [--data E:/mastercode/data/orange_yolo] [--splits train val test] [--limit 0]
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

BINS = [(0, 16, "<16px"), (16, 32, "16-32px"), (32, 64, "32-64px"), (64, 128, "64-128px"), (128, 10**9, ">=128px")]
PROC_LONG = 1536  # 分析用工作分辨率（长边），兼顾速度与小实例可测性


def bin_name(size640: float) -> str:
    for lo, hi, name in BINS:
        if lo <= size640 < hi:
            return name
    return BINS[-1][2]


def analyze_image(img_path: str, lbl_path: str, rows: list, img_id: str) -> None:
    img = cv2.imread(img_path)
    if img is None:
        return
    H0, W0 = img.shape[:2]
    scale = PROC_LONG / max(H0, W0)
    if scale < 1:
        img = cv2.resize(img, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    hsv_v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    lab_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(lbl_path, encoding="utf-8") as f:
        lines = [ln.split() for ln in f.read().strip().splitlines() if ln.strip()]

    for ln in lines:
        pts = np.array(ln[1:], dtype=np.float32).reshape(-1, 2)
        poly = np.round(pts * [W, H]).astype(np.int32)
        poly[:, 0] = poly[:, 0].clip(0, W - 1)  # 归一化坐标 ×W/H 四舍五入可能越界 1px
        poly[:, 1] = poly[:, 1].clip(0, H - 1)
        x, y, w, h = cv2.boundingRect(poly)
        w, h = min(w, W - x), min(h, H - y)
        if w < 2 or h < 2:
            continue
        # 尺寸换算（letterbox 按长边缩放到 640）
        min_side_640 = min(w, h) / max(W, H) * 640
        max_side_640 = max(w, h) / max(W, H) * 640
        min_side_native = min(w, h) / scale if scale < 1 else min(w, h)
        # 实例掩码与背景环
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 1)
        pad = max(3, int(0.4 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(ring, [poly - [x0, y0]], 1)
        ring_dil = cv2.dilate(ring, np.ones((2 * pad // 3 | 1, 2 * pad // 3 | 1), np.uint8))
        ring_only = (ring_dil > 0) & (ring == 0)
        m = mask.astype(bool)
        v_in = float(hsv_v[y : y + h, x : x + w][m].mean())
        a_in = float(lab_a[y : y + h, x : x + w][m].mean())
        v_bg = float(hsv_v[y0:y1, x0:x1][ring_only].mean()) if ring_only.any() else float("nan")
        a_bg = float(lab_a[y0:y1, x0:x1][ring_only].mean()) if ring_only.any() else float("nan")
        # 模糊度：裁剪统一放大到 64px 再算 Laplacian 方差（消除尺寸对方差的直接影响）
        crop = gray[y : y + h, x : x + w]
        crop64 = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_CUBIC)
        blur_var = float(cv2.Laplacian(crop64, cv2.CV_64F).var())
        rows.append(
            dict(
                image=img_id,
                min_side_640=round(min_side_640, 2),
                max_side_640=round(max_side_640, 2),
                min_side_native=round(min_side_native, 1),
                size_bin=bin_name(min_side_640),
                v_inst=round(v_in, 1),
                v_bg=round(v_bg, 1),
                dv=round(v_in - v_bg, 1),
                a_contrast=round(abs(a_in - a_bg), 2),
                blur_var=round(blur_var, 1),
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/mastercode/data/orange_yolo")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="每 split 最多处理图像数，0=全部")
    args = ap.parse_args()

    rows: list[dict] = []
    n_img = 0
    for split in args.splits:
        img_dir = os.path.join(args.data, split, "images")
        lbl_dir = os.path.join(args.data, split, "labels")
        files = sorted(os.listdir(img_dir))
        if args.limit:
            files = files[: args.limit]
        for i, fn in enumerate(files):
            lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            if not os.path.exists(lbl):
                continue
            analyze_image(os.path.join(img_dir, fn), lbl, rows, f"{split}/{fn}")
            n_img += 1
            if i % 50 == 0:
                print(f"[{split}] {i}/{len(files)} images, {len(rows)} instances", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "0_orange_yaml", "1_far_small")
    csv_path = os.path.join(out_dir, "_dataset_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 分箱聚合
    agg = defaultdict(list)
    for r in rows:
        agg[r["size_bin"]].append(r)
    lines = [
        "# 柑橘数据集量化体检报告（自动生成）",
        "",
        f"图像 {n_img} 张（原生多为 3072x3072），实例 {len(rows)} 个；工作分辨率长边 {PROC_LONG}。",
        "",
        "## 按 640 等效短边分箱",
        "",
        "| 尺寸箱 | 实例数 | 占比 | 亮度V(果) | 亮度差(果-背景) | 绿度对比\\|Δa*\\| | 模糊度(Lap.var@64) | 原生短边中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = [b[2] for b in BINS]
    for name in order:
        rs = agg.get(name, [])
        if not rs:
            continue
        med = lambda k: float(np.median([r[k] for r in rs if not np.isnan(r[k])]))  # noqa: E731
        lines.append(
            f"| {name} | {len(rs)} | {len(rs)/len(rows)*100:.1f}% | {med('v_inst'):.0f} | {med('dv'):+.1f} "
            f"| {med('a_contrast'):.1f} | {med('blur_var'):.0f} | {med('min_side_native'):.0f}px |"
        )
    # 关键结论自动判定
    small = [r for r in rows if r["min_side_640"] < 32]
    large = [r for r in rows if r["min_side_640"] >= 64]
    if small and large:
        sv = np.median([r["v_inst"] for r in small])
        lv = np.median([r["v_inst"] for r in large])
        sb = np.median([r["blur_var"] for r in small])
        lb = np.median([r["blur_var"] for r in large])
        sa = np.median([r["a_contrast"] for r in small])
        la = np.median([r["a_contrast"] for r in large])
        sn = np.median([r["min_side_native"] for r in small])
        lines += [
            "",
            "## 自动判定（<32px 小果 vs >=64px 大果，中位数）",
            "",
            f"- 亮度：小果 V={sv:.0f} vs 大果 V={lv:.0f}（{'小果显著更暗' if sv < lv - 5 else '亮度差异不显著'}）",
            f"- 模糊：小果 Lap.var={sb:.0f} vs 大果={lb:.0f}（{'小果显著更糊' if sb < 0.7 * lb else '模糊差异不显著'}）",
            f"- 绿度对比：小果 |Δa*|={sa:.1f} vs 大果={la:.1f}（{'小果更接近背景色' if sa < 0.8 * la else '对比差异不显著'}）",
            f"- **分辨率账**：<32px@640 的小果原生短边中位数 {sn:.0f}px —— 3072 原图压缩到 640 损失 {(1-640/3072)*100:.0f}% 线性分辨率，"
            f"这些果在原图上其实有 {sn:.0f}px 可用信息（支持 P2 层 / 更大 imgsz / 切片推理路线）",
        ]
    md_path = os.path.join(out_dir, "_dataset_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {csv_path}\nwrote {md_path}")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
