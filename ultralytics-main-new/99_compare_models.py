"""
公平对比 013 vs 015：同一验证集、同一OKS参数、无类别过滤。

用法:
    python 99_compare_models.py
    python 99_compare_models.py --sigma 0.25 --thresholds 0.5,0.6,0.7,0.8,0.9
"""

import argparse
import importlib.util
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO

from keypoint_map_utils import (
    compute_batch_oks,
    compute_mask_area,
    summarize_single_keypoint_map,
)


RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
VAL_IMG_DIR = r"E:\mastercode\data\shr_watermelon\segmentation\images\val"
VAL_LABEL_DIR = r"E:\mastercode\data\shr_watermelon\pose\labels\val"

MODEL_013_PATH = os.path.join(RESULTS_DIR, "13_roi_heatmap_v2", "best.pth")
MODEL_015_PATH = os.path.join(RESULTS_DIR, "15_roi_heatmap_distill", "best.pth")

MAX_GT_MATCH_DISTANCE_PX = 160
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── 工具函数 ───────────────────────────────────────────────

def load_module(name, filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_gt_points(label_path, w, h):
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_points = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label = shape.get("label", "")
        if label not in ("fully_visible", "partially_visible"):
            continue
        points = shape.get("points") or []
        if not points:
            continue
        pt = points[0]
        norm_pt = np.array([pt[0] / w, pt[1] / h], dtype=np.float32)
        gt_points.append({"label": label, "norm_pt": norm_pt})
    return gt_points


def compute_flower_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = np.array([0.5, 0.5], dtype=np.float32)
    if not contours:
        return center
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return center
    h, w = mask.shape
    return np.array([M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h], dtype=np.float32)


def match_masks_to_gt(masks, gt_points, img_w, img_h):
    if not masks or not gt_points:
        return []
    img_wh = np.array([img_w, img_h], dtype=np.float32)
    mask_centers = [compute_flower_center(m) for m in masks]
    candidates = []
    for mi, fc in enumerate(mask_centers):
        for gi, gp in enumerate(gt_points):
            dist = float(np.linalg.norm((gp["norm_pt"] - fc) * img_wh))
            candidates.append((dist, mi, gi))
    candidates.sort()
    used_m, used_g, matches = set(), set(), []
    for dist, mi, gi in candidates:
        if dist > MAX_GT_MATCH_DISTANCE_PX:
            break
        if mi in used_m or gi in used_g:
            continue
        used_m.add(mi)
        used_g.add(gi)
        matches.append((mi, gi))
    return matches


def mask_to_padded_bbox(mask, margin_ratio=0.25):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    px, py = int(round(bw * margin_ratio)), int(round(bh * margin_ratio))
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def gt_to_roi_xy(gt_norm, roi_box, img_w, img_h):
    gt_px = np.array([gt_norm[0] * img_w, gt_norm[1] * img_h], dtype=np.float32)
    x1, y1, x2, y2 = roi_box.astype(np.float32)
    roi_wh = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    if np.any(roi_wh <= 1):
        return None
    roi_xy = (gt_px - np.array([x1, y1])) / roi_wh
    if np.any(roi_xy < 0.0) or np.any(roi_xy > 1.0):
        return None
    return roi_xy.astype(np.float32)


def preprocess_roi(image, roi_box, roi_mask_resized, roi_size):
    x1, y1, x2, y2 = roi_box.astype(np.int32).tolist()
    crop = image[y1:y2, x1:x2]
    resized = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGE_MEAN) / IMAGE_STD
    mask_ch = (roi_mask_resized > 0).astype(np.float32)[..., None]
    roi = np.concatenate([rgb, mask_ch], axis=2)
    return np.transpose(roi, (2, 0, 1)).astype(np.float32)


def build_resized_roi_mask(mask, roi_box, roi_size):
    x1, y1, x2, y2 = roi_box.astype(np.int32).tolist()
    crop = mask[y1:y2, x1:x2]
    return cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)


def soft_argmax_2d(logits, beta=20.0):
    b, _, h, w = logits.shape
    flat = logits.reshape(b, -1)
    prob = F.softmax(flat * beta, dim=1)
    y_c = torch.linspace(0.0, 1.0, h, device=logits.device)
    x_c = torch.linspace(0.0, 1.0, w, device=logits.device)
    yy, xx = torch.meshgrid(y_c, x_c, indexing="ij")
    pred_x = torch.sum(prob * xx.reshape(-1)[None, :], dim=1)
    pred_y = torch.sum(prob * yy.reshape(-1)[None, :], dim=1)
    return torch.stack([pred_x, pred_y], dim=1)


def roi_xy_to_image_norm(roi_xy, roi_boxes, img_wh):
    x1y1 = roi_boxes[:, :2]
    roi_wh = torch.clamp(roi_boxes[:, 2:] - roi_boxes[:, :2], min=1.0)
    pred_px = x1y1 + roi_xy * roi_wh
    return torch.clamp(pred_px / torch.clamp(img_wh, min=1.0), 0.0, 1.0)


# ─── 构建统一验证集（无类别过滤）──────────────────────────

def build_full_val_set(seg_model, roi_size=128):
    """对所有验证图片，用YOLO提取全部mask，匹配GT，返回统一的ROI列表。"""
    img_files = sorted(
        f for f in os.listdir(VAL_IMG_DIR)
        if f.lower().endswith(".jpg") and not f.startswith("annotations")
    )
    samples = []
    skipped_no_mask = 0
    skipped_no_gt = 0
    skipped_no_roi = 0

    for img_file in tqdm(img_files, desc="Building unified val set", ncols=90):
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        label_path = os.path.join(VAL_LABEL_DIR, img_file.replace(".jpg", ".json"))
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        results = seg_model.predict(img_path, conf=0.25, verbose=False)
        if not results or results[0].masks is None:
            skipped_no_mask += 1
            continue

        masks = []
        for rm in results[0].masks:
            md = rm.data.cpu().numpy()[0]
            resized = cv2.resize(md, (w, h), interpolation=cv2.INTER_LINEAR)
            m = np.zeros((h, w), dtype=np.uint8)
            m[resized > 0.5] = 255
            if np.count_nonzero(m) > 0:
                masks.append(m)

        gt_points = load_gt_points(label_path, w, h)
        if not gt_points:
            skipped_no_gt += 1
            continue

        matches = match_masks_to_gt(masks, gt_points, w, h)
        for mask_idx, gt_idx in matches:
            mask = masks[mask_idx]
            gt_center = gt_points[gt_idx]["norm_pt"]
            mask_area = compute_mask_area(mask)
            roi_box = mask_to_padded_bbox(mask)
            if roi_box is None:
                skipped_no_roi += 1
                continue
            gt_roi_xy = gt_to_roi_xy(gt_center, roi_box, w, h)
            if gt_roi_xy is None:
                skipped_no_roi += 1
                continue
            roi_mask = build_resized_roi_mask(mask, roi_box, roi_size)
            roi = preprocess_roi(image, roi_box, roi_mask, roi_size)
            samples.append({
                "roi": torch.from_numpy(roi),
                "gt_roi_xy": torch.from_numpy(gt_roi_xy),
                "gt_center": torch.from_numpy(gt_center),
                "roi_box": torch.from_numpy(roi_box),
                "img_wh": torch.tensor([float(w), float(h)]),
                "mask_area": float(mask_area),
                "file": img_file,
            })

    print(f"Unified val set: {len(samples)} samples "
          f"(skipped: no_mask={skipped_no_mask}, no_gt={skipped_no_gt}, no_roi={skipped_no_roi})")
    return samples


# ─── 评估单个模型 ─────────────────────────────────────────

def evaluate_model(model, samples, device, sigma):
    model.eval()
    errors = []
    oks_scores = []
    for s in samples:
        roi = s["roi"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(roi)
        pred_roi_xy = soft_argmax_2d(logits).cpu()
        pred_center = roi_xy_to_image_norm(
            pred_roi_xy,
            s["roi_box"].unsqueeze(0),
            s["img_wh"].unsqueeze(0),
        )[0].numpy()
        gt_center = s["gt_center"].numpy()
        img_wh = s["img_wh"].numpy()
        error_px = float(np.linalg.norm((pred_center - gt_center) * img_wh))
        errors.append(error_px)
        oks_scores.append(
            compute_batch_oks(
                pred_center.reshape(1, -1),
                gt_center.reshape(1, -1),
                img_wh.reshape(1, -1),
                np.array([s["mask_area"]]),
                sigma=sigma,
            )[0]
        )
    return errors, oks_scores


# ─── 主函数 ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare 013 vs 015 on unified val set")
    parser.add_argument("--sigma", type=float, default=0.2, help="OKS sigma (default 0.2)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated OKS thresholds, e.g. 0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()

    thresholds = None
    if args.thresholds:
        thresholds = np.array([float(x) for x in args.thresholds.split(",")], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, OKS sigma: {args.sigma}")

    # 加载分割模型
    print("Loading YOLO segmentation model...")
    seg_model = YOLO(SEG_MODEL_PATH)

    # 构建统一验证集
    samples = build_full_val_set(seg_model, roi_size=128)
    if not samples:
        print("ERROR: No samples found.")
        return

    # 加载两个网络
    mod013 = load_module("net013", "013_improved_net_v2.py")
    mod015 = load_module("net015", "015_improved_net_v2.py")

    net013 = mod013.ROIHeatmapNet(in_channels=4, base_channels=32).to(device)
    ckpt013 = torch.load(MODEL_013_PATH, map_location=device)
    net013.load_state_dict(ckpt013["model"] if isinstance(ckpt013, dict) and "model" in ckpt013 else ckpt013)
    net013.eval()

    net015 = mod015.ROIHeatmapNet(in_channels=4, base_channels=8).to(device)
    ckpt015 = torch.load(MODEL_015_PATH, map_location=device)
    net015.load_state_dict(ckpt015["model"] if isinstance(ckpt015, dict) and "model" in ckpt015 else ckpt015)
    net015.eval()

    params013 = sum(p.numel() for p in net013.parameters())
    params015 = sum(p.numel() for p in net015.parameters() if p.requires_grad)

    # 评估
    print("\nEvaluating 013...")
    err013, oks013 = evaluate_model(net013, samples, device, args.sigma)
    print("Evaluating 015...")
    err015, oks015 = evaluate_model(net015, samples, device, args.sigma)

    # 计算指标
    err013_arr = np.asarray(err013, dtype=np.float32)
    err015_arr = np.asarray(err015, dtype=np.float32)
    map013 = summarize_single_keypoint_map(oks013, thresholds)
    map015 = summarize_single_keypoint_map(oks015, thresholds)

    # 输出对比表
    print("\n" + "=" * 70)
    print(f"  Fair Comparison: 013 vs 015  (val samples={len(samples)}, OKS sigma={args.sigma})")
    print("=" * 70)
    header = f"{'Metric':<25} {'013 (1.79M)':>18} {'015 (6.3K)':>18} {'Delta':>12}"
    print(header)
    print("-" * 70)

    rows = [
        ("Mean error (px)", f"{np.mean(err013_arr):.2f}", f"{np.mean(err015_arr):.2f}",
         f"{np.mean(err015_arr) - np.mean(err013_arr):+.2f}"),
        ("Median error (px)", f"{np.median(err013_arr):.2f}", f"{np.median(err015_arr):.2f}",
         f"{np.median(err015_arr) - np.median(err013_arr):+.2f}"),
        ("Std error (px)", f"{np.std(err013_arr):.2f}", f"{np.std(err015_arr):.2f}",
         f"{np.std(err015_arr) - np.std(err013_arr):+.2f}"),
        ("<10px ratio", f"{np.mean(err013_arr < 10) * 100:.1f}%", f"{np.mean(err015_arr < 10) * 100:.1f}%",
         f"{(np.mean(err015_arr < 10) - np.mean(err013_arr < 10)) * 100:+.1f}%"),
        ("<20px ratio", f"{np.mean(err013_arr < 20) * 100:.1f}%", f"{np.mean(err015_arr < 20) * 100:.1f}%",
         f"{(np.mean(err015_arr < 20) - np.mean(err013_arr < 20)) * 100:+.1f}%"),
        ("OKS mean", f"{map013['oks_mean']:.4f}", f"{map015['oks_mean']:.4f}",
         f"{map015['oks_mean'] - map013['oks_mean']:+.4f}"),
        ("OKS median", f"{map013['oks_median']:.4f}", f"{map015['oks_median']:.4f}",
         f"{map015['oks_median'] - map013['oks_median']:+.4f}"),
        ("mAP50", f"{map013['mAP50']:.4f}", f"{map015['mAP50']:.4f}",
         f"{map015['mAP50'] - map013['mAP50']:+.4f}"),
        ("mAP50-95", f"{map013['mAP50-95']:.4f}", f"{map015['mAP50-95']:.4f}",
         f"{map015['mAP50-95'] - map013['mAP50-95']:+.4f}"),
        ("Params", f"{params013:,}", f"{params015:,}", f"{params015/params013:.4f}x"),
    ]
    for name, v013, v015, delta in rows:
        print(f"{name:<25} {v013:>18} {v015:>18} {delta:>12}")

    print("-" * 70)

    # 各阈值AP
    print(f"\n  AP by OKS threshold (sigma={args.sigma}):")
    print(f"  {'Threshold':<12} {'013':>10} {'015':>10} {'Delta':>10}")
    for t_key in sorted(map013["ap_by_threshold"].keys()):
        a013 = map013["ap_by_threshold"][t_key]
        a015 = map015["ap_by_threshold"][t_key]
        print(f"  {t_key:<12} {a013:>10.4f} {a015:>10.4f} {a015 - a013:>+10.4f}")

    # 保存结果
    out = {
        "sigma": args.sigma,
        "num_samples": len(samples),
        "model_013": {
            "params": params013,
            "mean_error_px": float(np.mean(err013_arr)),
            "median_error_px": float(np.median(err013_arr)),
            "std_error_px": float(np.std(err013_arr)),
            "mAP50": map013["mAP50"],
            "mAP50-95": map013["mAP50-95"],
            "oks_mean": map013["oks_mean"],
            "oks_median": map013["oks_median"],
            "ap_by_threshold": map013["ap_by_threshold"],
        },
        "model_015": {
            "params": params015,
            "mean_error_px": float(np.mean(err015_arr)),
            "median_error_px": float(np.median(err015_arr)),
            "std_error_px": float(np.std(err015_arr)),
            "mAP50": map015["mAP50"],
            "mAP50-95": map015["mAP50-95"],
            "oks_mean": map015["oks_mean"],
            "oks_median": map015["oks_median"],
            "ap_by_threshold": map015["ap_by_threshold"],
        },
    }
    out_path = os.path.join(RESULTS_DIR, "99_fair_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
