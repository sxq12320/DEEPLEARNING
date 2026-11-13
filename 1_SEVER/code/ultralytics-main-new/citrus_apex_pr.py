"""Publication-grade PR Curve Analysis & Zero-Tail Mitigation Tool for Citrus APEX.

Explains and eliminates visual/analytical confusion from Ultralytics' PR curve cliff:
1. `ultralytics/utils/metrics.py` (`compute_ap` & `ap_per_class`) appends sentinel points
   `(recall[-1], 0.0)` and `(1.0, 0.0)` and linearly interpolates `p_curve` onto 1000
   uniform recall bins `np.linspace(0, 1, 1000)`. Consequently, whenever empirical max
   recall `R_max < 1.0`, the plotted curve drops vertically to 0 at `R_max`.
2. This tool computes exact COCO AP metrics untouched while generating publication-ready
   figures (`MaskPR_publication.png` and `.pdf`) separating:
   - True supported PR trajectory on `[0, R_max]`;
   - Explicit terminal operating marker `(R_max, P_end)`;
   - Shaded Unreached Recall Domain `(R_max, 1.0]` (`Missing Recall = 1 - R_max`);
   - Confidence vs. Recall/Precision dual curve diagnosing low-score tail FPs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def compute_empirical_pr_trajectory(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iou_col: int = 0,
    eps: float = 1e-16,
) -> dict[str, Any]:
    """Compute exact sorted empirical PR curve without sentinel zero-drop artifacts.

    Args:
        tp: Boolean array of shape `(N, 10)` from Ultralytics validator (`stat['tp']`).
        conf: Float array `(N,)` of predicted confidences.
        pred_cls: Int array `(N,)` of predicted class ids.
        target_cls: Int array `(M,)` of GT class ids.
        iou_col: IoU threshold column index (`0` corresponds to `IoU=0.50`).
    """
    if len(conf) == 0 or len(target_cls) == 0:
        return {
            "num_gt": int(len(target_cls)),
            "num_pred": int(len(conf)),
            "r_max": 0.0,
            "p_end": 0.0,
            "ap50_coco": 0.0,
            "recall": np.zeros(1),
            "precision": np.zeros(1),
            "conf_sorted": np.zeros(1),
        }

    order = np.argsort(-conf)
    tp_s = tp[order, iou_col].astype(np.float64)
    conf_s = conf[order].astype(np.float64)
    fp_s = 1.0 - tp_s

    tp_cum = np.cumsum(tp_s)
    fp_cum = np.cumsum(fp_s)
    num_gt = float(len(target_cls))

    recall = tp_cum / (num_gt + eps)
    precision = tp_cum / (tp_cum + fp_cum + eps)

    # Exact COCO 101-point or continuous monotone-envelope AP for fair verification:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x_grid = np.linspace(0.0, 1.0, 101)
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    ap50 = float(trapz_fn(np.interp(x_grid, mrec, mpre), x_grid))

    # Subsample smoothly to <= 1000 points for plotting while preserving terminal point.
    if len(recall) > 1000:
        idx = np.unique(np.append(np.linspace(0, len(recall) - 1, 999, dtype=int), len(recall) - 1))
        r_plot, p_plot, c_plot = recall[idx], precision[idx], conf_s[idx]
    else:
        r_plot, p_plot, c_plot = recall, precision, conf_s

    return {
        "num_gt": int(num_gt),
        "num_pred": int(len(conf)),
        "matched_gt": int(tp_cum[-1]),
        "unmatched_gt": int(num_gt - tp_cum[-1]),
        "total_fp": int(fp_cum[-1]),
        "low_conf_fp_below_005": int(((conf_s < 0.05) & (fp_s > 0.5)).sum()),
        "r_max": float(recall[-1]),
        "p_end": float(precision[-1]),
        "ap50_coco": ap50,
        "recall": r_plot,
        "precision": p_plot,
        "conf_sorted": c_plot,
    }


def plot_publication_pr_suite(
    curves: dict[str, dict[str, Any]],
    save_dir: Path,
    title: str = "Whole-Image (640x640) Mask Precision-Recall Analysis",
) -> Path:
    """Plot clean dual-panel PR analysis separating empirical support from sentinel drop."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.0})

    fig, (ax_pr, ax_conf) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=220)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 2)))

    max_rmax = 0.0
    for idx, (label, data) in enumerate(curves.items()):
        c = colors[idx % len(colors)]
        r = np.asarray(data["recall"])
        p = np.asarray(data["precision"])
        conf = np.asarray(data["conf_sorted"])
        rmax = float(data["r_max"])
        pend = float(data["p_end"])
        ap50 = float(data["ap50_coco"])
        max_rmax = max(max_rmax, rmax)

        # Panel A: Supported empirical curve on [0, R_max] (NO misleading drop to 0)
        ax_pr.plot(
            np.concatenate(([0.0], r)),
            np.concatenate(([1.0], p)),
            color=c,
            linewidth=2.0,
            label=f"{label} (AP50={ap50*100:.2f}%, R_max={rmax:.3f})",
        )
        ax_pr.scatter([rmax], [pend], color=c, s=36, zorder=5, edgecolors="white", linewidths=0.8)
        # Faint dashed drop to illustrate Ultralytics' default sentinel closure.
        ax_pr.plot([rmax, rmax, 1.0], [pend, 0.0, 0.0], color=c, linestyle=":", alpha=0.35, linewidth=1.1)

        # Panel B: Confidence threshold vs Recall and Precision
        ax_conf.plot(conf, r, color=c, linewidth=1.8, linestyle="-", label=f"{label} Recall")
        ax_conf.plot(conf, p, color=c, linewidth=1.4, linestyle="--", alpha=0.85, label=f"{label} Prec")

    if 0.0 < max_rmax < 1.0:
        ax_pr.axvspan(max_rmax, 1.0, color="#f2dede", alpha=0.35, label="Unreached Recall Zone (1 - R_max)")
        ax_pr.axvline(max_rmax, color="#b94a48", linestyle="--", linewidth=1.0, alpha=0.7)

    ax_pr.set_xlim(0.0, 1.02)
    ax_pr.set_ylim(0.0, 1.03)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{title}\n(Empirical Support [0, R_max] vs. Sentinel Tail)")
    ax_pr.grid(True, linestyle="--", alpha=0.3)
    ax_pr.legend(loc="lower left", fontsize=8, frameon=True)

    ax_conf.set_xlim(0.0, 1.0)
    ax_conf.set_ylim(0.0, 1.03)
    ax_conf.set_xlabel("Confidence Threshold")
    ax_conf.set_ylabel("Precision / Recall")
    ax_conf.set_title("Operating Point Diagnosis (Low-Conf FP Surge Region)")
    ax_conf.axvspan(0.0, 0.05, color="#fcf8e3", alpha=0.55, label="conf < 0.05 tail FP zone")
    ax_conf.grid(True, linestyle="--", alpha=0.3)
    ax_conf.legend(loc="center right", fontsize=7.5, frameon=True, ncol=2)

    fig.tight_layout()
    out_png = save_dir / "MaskPR_publication.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(save_dir / "MaskPR_publication.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        k: {
            "num_gt": v.get("num_gt", 0),
            "matched_gt": v.get("matched_gt", 0),
            "unmatched_gt": v.get("unmatched_gt", 0),
            "total_fp": v.get("total_fp", 0),
            "low_conf_fp_below_005": v.get("low_conf_fp_below_005", 0),
            "r_max": round(float(v["r_max"]), 6),
            "p_end": round(float(v["p_end"]), 6),
            "ap50_coco": round(float(v["ap50_coco"]), 6),
        }
        for k, v in curves.items()
    }
    (save_dir / "pr_tail_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_png


if __name__ == "__main__":
    # Quick synthetic self-test demonstrating clean PR plot generation without sentinel drop artifacts.
    rng = np.random.default_rng(0)
    n_gt = 1049
    curves_demo = {}
    for name, r_cap, noise_n in [("Baseline-Style", 0.872, 14500), ("APEX-Target-Style", 0.935, 3800)]:
        n_tp = int(round(n_gt * r_cap))
        tp = np.zeros((n_tp + noise_n, 1), dtype=bool)
        tp[:n_tp, 0] = True
        conf = np.concatenate([rng.uniform(0.25, 0.98, n_tp), rng.beta(0.4, 8.0, noise_n)])
        curves_demo[name] = compute_empirical_pr_trajectory(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))
    demo_dir = Path(__file__).resolve().parent / "docs" / "APEX_PR_DEMO"
    out = plot_publication_pr_suite(curves_demo, demo_dir)
    print(f"Saved demonstration PR plot to {out}")
