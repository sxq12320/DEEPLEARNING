import math

import numpy as np


DEFAULT_OKS_SIGMA = 0.2
DEFAULT_OKS_THRESHOLDS = np.arange(0.50, 1.00, 0.05, dtype=np.float32)


def compute_mask_area(mask):
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return 0.0
    return float(np.count_nonzero(mask_array > 0))


def compute_oks(pred_xy_norm, gt_xy_norm, img_wh, area_px, sigma=DEFAULT_OKS_SIGMA):
    pred = np.asarray(pred_xy_norm, dtype=np.float32).reshape(2)
    gt = np.asarray(gt_xy_norm, dtype=np.float32).reshape(2)
    wh = np.asarray(img_wh, dtype=np.float32).reshape(2)

    area = max(float(area_px), 1.0)
    sigma = max(float(sigma), 1e-6)

    diff_px = (pred - gt) * wh
    squared_distance = float(np.dot(diff_px, diff_px))
    denominator = 8.0 * (sigma ** 2) * area + 1e-7
    return float(math.exp(-squared_distance / denominator))


def compute_batch_oks(pred_xy_norm_batch, gt_xy_norm_batch, img_wh_batch, area_px_batch, sigma=DEFAULT_OKS_SIGMA):
    pred_batch = np.asarray(pred_xy_norm_batch, dtype=np.float32)
    gt_batch = np.asarray(gt_xy_norm_batch, dtype=np.float32)
    wh_batch = np.asarray(img_wh_batch, dtype=np.float32)
    area_batch = np.asarray(area_px_batch, dtype=np.float32).reshape(-1)

    oks_scores = [
        compute_oks(pred_xy, gt_xy, img_wh, area_px, sigma=sigma)
        for pred_xy, gt_xy, img_wh, area_px in zip(pred_batch, gt_batch, wh_batch, area_batch)
    ]
    return np.asarray(oks_scores, dtype=np.float32)


def summarize_single_keypoint_map(oks_scores, thresholds=None):
    thresholds = DEFAULT_OKS_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    oks = np.asarray(list(oks_scores), dtype=np.float32)

    ap_by_threshold = {f"{threshold:.2f}": 0.0 for threshold in thresholds}
    if oks.size == 0:
        return {
            "num_samples": 0,
            "oks_mean": 0.0,
            "oks_median": 0.0,
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "ap_by_threshold": ap_by_threshold,
        }

    ap_values = np.asarray([(oks >= threshold).mean() for threshold in thresholds], dtype=np.float32)
    ap_by_threshold = {
        f"{threshold:.2f}": float(ap_value)
        for threshold, ap_value in zip(thresholds, ap_values)
    }
    return {
        "num_samples": int(oks.size),
        "oks_mean": float(np.mean(oks)),
        "oks_median": float(np.median(oks)),
        "mAP50": float(ap_values[0]),
        "mAP50-95": float(np.mean(ap_values)),
        "ap_by_threshold": ap_by_threshold,
    }
