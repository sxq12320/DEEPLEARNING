"""Difficulty-attribute benchmark for immature-citrus instance segmentation.

Computes four per-instance difficulty dimensions + per-image density, then builds
the evaluation subsets defined in the research plan: small, dense, adhesion, high-camouflage.

Output: data/test/difficulty_benchmark.json

Usage:  python compute_difficulty_attrs.py  [--ring-width 20]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path(r"E:/mastercode/data/orange_yolo")
IMGSZ = 640

# ---------- geometry helpers ----------

def polygon_area_original(vertices_norm, w, h):
    """Shoelace area of a normalized polygon in original image pixels."""
    pts = vertices_norm * np.array([w, h])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_area_at_imgsz(vertices_norm, w, h):
    """Area of a normalized polygon scaled to imgsz training resolution (square)."""
    scale = IMGSZ / max(w, h)
    pts = vertices_norm * np.array([w * scale, h * scale])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def point_segment_dist(p, a, b):
    ab = b - a; ap = p - a
    t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(ap - t * ab))

def polygon_min_distance(poly1, poly2):
    """Exact minimum distance between two polygon boundaries (point-to-segment)."""
    best = float("inf")
    for i in range(len(poly1)):
        p = poly1[i]
        for j in range(len(poly2)):
            a, b = poly2[j], poly2[(j+1) % len(poly2)]
            best = min(best, point_segment_dist(p, a, b))
    for i in range(len(poly2)):
        p = poly2[i]
        for j in range(len(poly1)):
            a, b = poly1[j], poly1[(j+1) % len(poly1)]
            best = min(best, point_segment_dist(p, a, b))
    return best

def polygon_mask(poly_norm, w, h):
    """Binary mask from normalized polygon."""
    pts = (poly_norm * np.array([w, h])).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask

# ---------- colour helpers ----------

def lab_mean(img_bgr, mask):
    """Mean CIELAB inside mask region."""
    if mask.sum() < 10:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    pixels = lab[mask > 0]
    return pixels.mean(axis=0)  # (L, a, b)

def delta_e(lab1, lab2):
    """CIE76 ΔE between two mean LAB colours."""
    if lab1 is None or lab2 is None:
        return None
    return float(np.linalg.norm(lab1 - lab2))

def rgb_hist_chi2(img_bgr, mask_fg, mask_ring):
    """Chi-squared distance between RGB histograms of foreground and ring."""
    if mask_fg.sum() < 10 or mask_ring.sum() < 10:
        return None
    chi2 = 0.0
    for ch in range(3):
        h_fg = np.histogram(img_bgr[mask_fg > 0, ch], bins=64, range=(0, 256))[0].astype(float)
        h_ring = np.histogram(img_bgr[mask_ring > 0, ch], bins=64, range=(0, 256))[0].astype(float)
        chi2 += np.sum((h_fg - h_ring) ** 2 / (h_fg + h_ring + 1e-10))
    return float(chi2 / 3.0)

# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ring-width", type=int, default=20, help="Background ring dilation in original pixels")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    return p.parse_args()

def load_labels(label_path, w, h):
    """Return list of (poly_norm) arrays from a YOLO-seg label file."""
    polys = []
    if not label_path.exists():
        return polys
    for line in label_path.read_text().splitlines():
        vals = line.split()
        if len(vals) < 7:
            continue
        xy = np.array(vals[1:], dtype=float).reshape(-1, 2)
        xy[:, 0] *= w / IMGSZ if False else 1.0  # we keep norm coords
        polys.append(xy)
    return polys

def compute(args):
    out = {
        "config": {
            "imgsz": args.imgsz,
            "ring_width_px": args.ring_width,
            "small_thresholds": {"extreme": 256, "small": 1024, "medium": 4096},
            "dense_thresholds": {"extreme": 20, "dense": 10, "moderate": 5},
            "adhesion_threshold_px": 5,
            "camouflage_delta_e_threshold": 8.0,
            "camouflage_chi2_threshold": 0.3,
        },
        "splits": {},
    }

    for split in ["train", "val", "test"]:
    for split in ["train", "val", "test"]:
        img_dir = DATA_ROOT / split / "images"
        lbl_dir = DATA_ROOT / split / "labels"
        if not img_dir.exists():
            continue

        instances = []
        images = {}
        n_processed = 0

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            polys_norm = load_labels(lbl_dir / f"{stem}.txt", w, h)
            polys_orig = [p * np.array([w, h]) for p in polys_norm]
            n_inst = len(polys_orig)

            # per-image
            img_info = {
                "stem": stem,
                "width": w,
                "height": h,
                "n_instances": n_inst,
                "dense_level": "extreme" if n_inst >= 20 else "dense" if n_inst >= 10 else "moderate" if n_inst >= 5 else "sparse",
            }
            images[stem] = img_info

            # ---------- pairwise distances for adhesion ----------
            min_dists = [float("inf")] * n_inst
            if n_inst >= 2:
                for i in range(n_inst):
                    for j in range(i + 1, n_inst):
                        d = polygon_min_distance(polys_orig[i], polys_orig[j])
                        min_dists[i] = min(min_dists[i], d)
                        min_dists[j] = min(min_dists[j], d)

            for idx, (poly_norm, poly_orig) in enumerate(zip(polys_norm, polys_orig)):
                # --- small ---
                area_640 = polygon_area_at_imgsz(poly_norm, w, h)
                small_level = "extreme" if area_640 < 256 else "small" if area_640 < 1024 else "medium" if area_640 < 4096 else "large"

                # --- adhesion ---
                min_d = min_dists[idx]
                adhesion_level = "touching" if min_d < 3 else "close" if min_d < args.ring_width else "separated"

                # --- camouflage (CIELAB + RGB hist) ---
                mask_fg = polygon_mask(poly_norm, w, h)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.ring_width * 2 + 1, args.ring_width * 2 + 1))
                mask_dilated = cv2.dilate(mask_fg, kernel, iterations=1)
                mask_ring = mask_dilated.astype(np.uint8) - mask_fg.astype(np.uint8)

                lab_fg = lab_mean(img, mask_fg)
                lab_ring = lab_mean(img, mask_ring)
                de = delta_e(lab_fg, lab_ring)
                chi2 = rgb_hist_chi2(img, mask_fg, mask_ring)

                inst = {
                    "image": stem,
                    "instance_idx": idx,
                    "area_640": round(area_640, 1),
                    "small_level": small_level,
                    "min_neighbor_dist_px": round(min_d, 2) if min_d < float("inf") else None,
                    "adhesion_level": adhesion_level if n_inst >= 2 else "isolated",
                    "delta_e_cielab": round(de, 2) if de is not None else None,
                    "rgb_chi2": round(chi2, 4) if chi2 is not None else None,
                }
                # camouflage = both colour metric AND colour histogram similar
                if de is not None and chi2 is not None:
                    inst["camouflage_level"] = "high" if (de < 8.0 and chi2 < 0.3) else "moderate" if (de < 14.0 or chi2 < 0.6) else "low"
                else:
                    inst["camouflage_level"] = "unknown"
                instances.append(inst)

            n_processed += 1
            if n_processed % 100 == 0:
                print(f"  [{split}] {n_processed} images processed...", file=sys.stderr)

        out["splits"][split] = {"images": images, "instances": instances}
        print(f"[{split}] done: {n_processed} images, {len(instances)} instances", file=sys.stderr)

    # ---------- build subsets ----------
    subsets = {}
    for tag, attr in [("small", "small_level"), ("dense", "dense_level"), ("adhesion", "adhesion_level"), ("camouflage", "camouflage_level")]:
        key_suffix = {"small": ("extreme", "small"), "dense": ("extreme", "dense"), "adhesion": ("touching", "close"), "camouflage": ("high", "moderate")}[tag]
        subset_stems = set()
        for split_data in out["splits"].values():
            if tag == "dense":
                for stem, info in split_data["images"].items():
                    if info.get("dense_level") in key_suffix:
                        subset_stems.add(stem)
            else:
                for inst in split_data["instances"]:
                    lv = inst.get(attr, "")
                    if lv in key_suffix:
                        subset_stems.add(inst["image"])
        subsets[tag] = sorted(subset_stems)

    out["subsets"] = subsets

    # ---------- summary stats ----------
    all_inst = []
    for sd in out["splits"].values():
        all_inst.extend(sd["instances"])
    out["summary"] = {
        "total_images": sum(len(sd["images"]) for sd in out["splits"].values()),
        "total_instances": len(all_inst),
        "small_pct": sum(1 for i in all_inst if i["small_level"] in ("extreme", "small")) / max(len(all_inst), 1) * 100,
        "dense_images_pct": sum(1 for sd in out["splits"].values() for v in sd["images"].values() if v.get("dense_level") in ("extreme", "dense")) / max(sum(len(sd["images"]) for sd in out["splits"].values()), 1) * 100,
        "adhesion_pct": sum(1 for i in all_inst if i["adhesion_level"] in ("touching", "close")) / max(len(all_inst), 1) * 100,
        "camouflage_high_pct": sum(1 for i in all_inst if i["camouflage_level"] == "high") / max(len(all_inst), 1) * 100,
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
    }

    dest = DATA_ROOT / "difficulty_benchmark.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten -> {dest}", file=sys.stderr)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(json.dumps({k: len(v) for k, v in subsets.items()}, ensure_ascii=False))


if __name__ == "__main__":
    compute(parse_args())
