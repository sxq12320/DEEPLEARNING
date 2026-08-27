"""Quantify task-specific difficulty in the grouped citrus instance dataset.

The audit is deliberately label-preserving: it reads images and YOLO polygons but
does not rewrite either. Metrics are computed after the same aspect-preserving
letterbox scaling used for a 640-pixel training input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit citrus small-object, topology, and camouflage difficulty.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"E:\mastercode\data\orange_yolo_grouped_dedup_20260820"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\mastercode\1_SEVER\results\_analysis_20260824_network_redesign\dataset_difficulty"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def describe(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "n": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "p10": percentile(finite, 10),
        "p25": percentile(finite, 25),
        "median": percentile(finite, 50),
        "p75": percentile(finite, 75),
        "p90": percentile(finite, 90),
    }


def read_polygons(label_path: Path, width: int, height: int, imgsz: int) -> list[np.ndarray]:
    scale = min(imgsz / width, imgsz / height)
    pad_x = (imgsz - width * scale) / 2
    pad_y = (imgsz - height * scale) / 2
    polygons: list[np.ndarray] = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            continue
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = coords[:, 0] * width * scale + pad_x
        coords[:, 1] = coords[:, 1] * height * scale + pad_y
        coords = np.clip(np.rint(coords), 0, imgsz - 1).astype(np.int32)
        if len(np.unique(coords, axis=0)) >= 3:
            polygons.append(coords)
    return polygons


def letterbox(image: np.ndarray, imgsz: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(imgsz / width, imgsz / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    y0 = (imgsz - resized.shape[0]) // 2
    x0 = (imgsz - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def polygon_mask(contour: np.ndarray, imgsz: int) -> np.ndarray:
    mask = np.zeros((imgsz, imgsz), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    return mask


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def raster_gap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    x0 = max(min(box_a[0], box_b[0]) - 3, 0)
    y0 = max(min(box_a[1], box_b[1]) - 3, 0)
    x1 = min(max(box_a[2], box_b[2]) + 4, mask_a.shape[1])
    y1 = min(max(box_a[3], box_b[3]) + 4, mask_a.shape[0])
    crop_a = mask_a[y0:y1, x0:x1]
    crop_b = mask_b[y0:y1, x0:x1]
    if np.any(crop_a & crop_b):
        return 0.0
    distance = cv2.distanceTransform((1 - crop_a).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[crop_b.astype(bool)].min()) if crop_b.any() else float("inf")


def color_metrics(
    lab: np.ndarray,
    gradient: np.ndarray,
    mask: np.ndarray,
    other_masks: np.ndarray,
) -> tuple[float, float, float]:
    kernel_inner = np.ones((3, 3), np.uint8)
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    inner = cv2.erode(mask, kernel_inner, iterations=1).astype(bool)
    if inner.sum() < 8:
        inner = mask.astype(bool)
    dilated = cv2.dilate(mask, kernel_outer, iterations=1).astype(bool)
    ring = dilated & ~mask.astype(bool) & ~other_masks.astype(bool)
    valid = np.any(lab != np.array([122, 128, 128], dtype=np.uint8), axis=2)
    ring &= valid
    if inner.sum() < 4 or ring.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    delta_e = float(np.linalg.norm(lab[inner].mean(axis=0) - lab[ring].mean(axis=0)))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel_inner).astype(bool)
    boundary_gradient = float(gradient[boundary].mean()) if boundary.any() else float("nan")
    ring_gradient = float(gradient[ring].mean()) + 1e-6
    edge_ratio = boundary_gradient / ring_gradient
    green_fraction = float(((lab[inner, 1] < 128) & (lab[inner, 2] > 128)).mean())
    return delta_e, edge_ratio, green_fraction


def audit_image(image_path: Path, split: str, label_path: Path, imgsz: int) -> tuple[list[dict], dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    polygons = read_polygons(label_path, width, height, imgsz)
    canvas = letterbox(image, imgsz)
    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    masks = [polygon_mask(poly, imgsz) for poly in polygons]
    union = np.maximum.reduce(masks) if masks else np.zeros((imgsz, imgsz), dtype=np.uint8)
    boxes = [tuple(int(v) for v in cv2.boundingRect(poly)) for poly in polygons]
    boxes = [(x, y, x + w - 1, y + h - 1) for x, y, w, h in boxes]
    nearest = [float("inf")] * len(polygons)
    overlap_neighbors = [0] * len(polygons)
    near2_neighbors = [0] * len(polygons)
    near4_neighbors = [0] * len(polygons)
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            lower_bound = bbox_gap(boxes[i], boxes[j])
            gap = raster_gap(masks[i], masks[j], boxes[i], boxes[j]) if lower_bound <= 32 else lower_bound
            nearest[i] = min(nearest[i], gap)
            nearest[j] = min(nearest[j], gap)
            if gap == 0:
                overlap_neighbors[i] += 1
                overlap_neighbors[j] += 1
            if gap <= 2:
                near2_neighbors[i] += 1
                near2_neighbors[j] += 1
            if gap <= 4:
                near4_neighbors[i] += 1
                near4_neighbors[j] += 1

    rows: list[dict] = []
    areas: list[float] = []
    for idx, (poly, mask, box) in enumerate(zip(polygons, masks, boxes)):
        area = float(cv2.contourArea(poly.astype(np.float32)))
        hull_area = float(cv2.contourArea(cv2.convexHull(poly).astype(np.float32)))
        perimeter = float(cv2.arcLength(poly.astype(np.float32), True))
        solidity = area / hull_area if hull_area > 0 else float("nan")
        x0, y0, x1, y1 = box
        box_w, box_h = x1 - x0 + 1, y1 - y0 + 1
        other_masks = union & (1 - mask)
        delta_e, edge_ratio, green_fraction = color_metrics(lab, gradient, mask, other_masks)
        areas.append(area)
        rows.append(
            {
                "split": split,
                "image": image_path.name,
                "instance": idx,
                "area_px2_640": round(area, 4),
                "bbox_w_px_640": box_w,
                "bbox_h_px_640": box_h,
                "bbox_min_side_px_640": min(box_w, box_h),
                "coco_small_lt32sq": int(area < 32**2),
                "ultra_tiny_min_side_lt8": int(min(box_w, box_h) < 8),
                "tiny_min_side_lt16": int(min(box_w, box_h) < 16),
                "solidity": round(solidity, 6),
                "convex_deficit": round(1 - solidity, 6),
                "boundary_complexity": round(perimeter**2 / (4 * math.pi * max(area, 1.0)), 6),
                "nearest_instance_gap_px_640": round(nearest[idx], 4) if math.isfinite(nearest[idx]) else "",
                "overlap_neighbor_count": overlap_neighbors[idx],
                "neighbor_gap_le2_count": near2_neighbors[idx],
                "neighbor_gap_le4_count": near4_neighbors[idx],
                "lab_delta_e_to_context": round(delta_e, 6) if math.isfinite(delta_e) else "",
                "boundary_to_context_gradient_ratio": round(edge_ratio, 6) if math.isfinite(edge_ratio) else "",
                "green_lab_fraction": round(green_fraction, 6) if math.isfinite(green_fraction) else "",
            }
        )
    linear_areas = [math.sqrt(max(a, 1.0)) for a in areas]
    image_row = {
        "split": split,
        "image": image_path.name,
        "instances": len(rows),
        "linear_scale_ratio": round(max(linear_areas) / min(linear_areas), 6) if len(linear_areas) >= 2 else 1.0,
        "area_ratio": round(max(areas) / max(min(areas), 1.0), 6) if len(areas) >= 2 else 1.0,
        "tiny_instances_lt16": sum(int(r["tiny_min_side_lt16"]) for r in rows),
        "concave_instances_solidity_lt085": sum(float(r["solidity"]) < 0.85 for r in rows),
        "near_instances_gap_le2": sum(int(r["neighbor_gap_le2_count"]) > 0 for r in rows),
    }
    return rows, image_row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value != "" and value is not None:
            values.append(float(value))
    return values


def build_summary(instance_rows: list[dict], image_rows: list[dict], imgsz: int) -> dict:
    total = len(instance_rows)
    counts = Counter(row["split"] for row in instance_rows)
    summary = {
        "imgsz": imgsz,
        "images": len(image_rows),
        "instances": total,
        "instances_by_split": dict(counts),
        "fractions": {},
        "distributions": {},
    }
    flags = {
        "coco_small_area": "coco_small_lt32sq",
        "ultra_tiny_min_side_lt8": "ultra_tiny_min_side_lt8",
        "tiny_min_side_lt16": "tiny_min_side_lt16",
        "strong_concavity_solidity_lt085": None,
        "very_strong_concavity_solidity_lt070": None,
        "has_neighbor_gap_le2": None,
        "has_neighbor_gap_le4": None,
        "low_color_contrast_delta_e_lt10": None,
        "weak_boundary_gradient_ratio_lt1": None,
        "combined_tiny_and_low_contrast": None,
        "combined_concave_and_near": None,
    }
    for name, key in flags.items():
        if key:
            count = sum(int(row[key]) for row in instance_rows)
        elif name == "strong_concavity_solidity_lt085":
            count = sum(float(row["solidity"]) < 0.85 for row in instance_rows)
        elif name == "very_strong_concavity_solidity_lt070":
            count = sum(float(row["solidity"]) < 0.70 for row in instance_rows)
        elif name == "has_neighbor_gap_le2":
            count = sum(int(row["neighbor_gap_le2_count"]) > 0 for row in instance_rows)
        elif name == "has_neighbor_gap_le4":
            count = sum(int(row["neighbor_gap_le4_count"]) > 0 for row in instance_rows)
        elif name == "low_color_contrast_delta_e_lt10":
            count = sum(
                row["lab_delta_e_to_context"] != "" and float(row["lab_delta_e_to_context"]) < 10
                for row in instance_rows
            )
        elif name == "weak_boundary_gradient_ratio_lt1":
            count = sum(
                row["boundary_to_context_gradient_ratio"] != ""
                and float(row["boundary_to_context_gradient_ratio"]) < 1
                for row in instance_rows
            )
        elif name == "combined_tiny_and_low_contrast":
            count = sum(
                int(row["tiny_min_side_lt16"])
                and row["lab_delta_e_to_context"] != ""
                and float(row["lab_delta_e_to_context"]) < 10
                for row in instance_rows
            )
        else:
            count = sum(
                float(row["solidity"]) < 0.85 and int(row["neighbor_gap_le2_count"]) > 0
                for row in instance_rows
            )
        summary["fractions"][name] = {"count": count, "fraction": count / total if total else 0.0}
    for key in (
        "area_px2_640",
        "bbox_min_side_px_640",
        "solidity",
        "convex_deficit",
        "boundary_complexity",
        "nearest_instance_gap_px_640",
        "lab_delta_e_to_context",
        "boundary_to_context_gradient_ratio",
        "green_lab_fraction",
    ):
        summary["distributions"][key] = describe(numeric(instance_rows, key))
    summary["image_distributions"] = {
        "instances_per_image": describe(numeric(image_rows, "instances")),
        "linear_scale_ratio": describe(numeric(image_rows, "linear_scale_ratio")),
        "area_ratio": describe(numeric(image_rows, "area_ratio")),
    }
    return summary


def fmt_fraction(item: dict) -> str:
    return f"{item['count']} ({item['fraction'] * 100:.2f}%)"


def write_report(path: Path, dataset: Path, summary: dict) -> None:
    fractions = summary["fractions"]
    distributions = summary["distributions"]
    image_distributions = summary["image_distributions"]
    min_side = distributions["bbox_min_side_px_640"]
    convex_deficit = distributions["convex_deficit"]
    boundary_complexity = distributions["boundary_complexity"]
    lab_delta = distributions["lab_delta_e_to_context"]
    scale_ratio = image_distributions["linear_scale_ratio"]
    lines = [
        "# Citrus dataset difficulty audit",
        "",
        f"- Dataset: `{dataset}`",
        f"- Letterbox input: {summary['imgsz']} × {summary['imgsz']}",
        f"- Images: {summary['images']}",
        f"- Valid polygon instances: {summary['instances']}",
        "",
        "## Task-specific evidence",
        "",
        "| Evidence | Count (fraction) |",
        "|---|---:|",
        f"| COCO-small area (<32² px) | {fmt_fraction(fractions['coco_small_area'])} |",
        f"| Ultra-tiny min side <8 px | {fmt_fraction(fractions['ultra_tiny_min_side_lt8'])} |",
        f"| Tiny min side <16 px | {fmt_fraction(fractions['tiny_min_side_lt16'])} |",
        f"| Strong visible-mask concavity (solidity <0.85) | "
        f"{fmt_fraction(fractions['strong_concavity_solidity_lt085'])} |",
        f"| Very strong concavity (solidity <0.70) | "
        f"{fmt_fraction(fractions['very_strong_concavity_solidity_lt070'])} |",
        f"| Has another instance within 2 px | {fmt_fraction(fractions['has_neighbor_gap_le2'])} |",
        f"| Has another instance within 4 px | {fmt_fraction(fractions['has_neighbor_gap_le4'])} |",
        f"| Low local Lab contrast (ΔE <10) | {fmt_fraction(fractions['low_color_contrast_delta_e_lt10'])} |",
        f"| Boundary weaker than local context (gradient ratio <1) | "
        f"{fmt_fraction(fractions['weak_boundary_gradient_ratio_lt1'])} |",
        f"| Tiny and low-contrast jointly | {fmt_fraction(fractions['combined_tiny_and_low_contrast'])} |",
        f"| Concave and near-neighbor jointly | {fmt_fraction(fractions['combined_concave_and_near'])} |",
        "",
        "## Distribution checkpoints",
        "",
        "| Metric | Median | P90 |",
        "|---|---:|---:|",
        f"| Min bbox side at 640 (px) | {min_side['median']:.2f} | {min_side['p90']:.2f} |",
        f"| Convex deficit | {convex_deficit['median']:.3f} | {convex_deficit['p90']:.3f} |",
        f"| Boundary complexity | {boundary_complexity['median']:.3f} | {boundary_complexity['p90']:.3f} |",
        f"| Local Lab ΔE | {lab_delta['median']:.2f} | {lab_delta['p90']:.2f} |",
        f"| Per-image linear scale ratio | {scale_ratio['median']:.2f} | {scale_ratio['p90']:.2f} |",
        "",
        "Thresholds in this report are descriptive challenge tags, not claims of biological categories. "
        "The per-instance CSV is the source for sensitivity analysis and challenge-subset evaluation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instance_rows: list[dict] = []
    image_rows: list[dict] = []
    errors: list[dict] = []
    for split in SPLITS:
        image_dir = args.dataset / split / "images"
        label_dir = args.dataset / split / "labels"
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_dir / f"{image_path.stem}.txt"
            try:
                rows, image_row = audit_image(image_path, split, label_path, args.imgsz)
                instance_rows.extend(rows)
                image_rows.append(image_row)
            except Exception as exc:  # keep the audit complete and expose every unreadable item
                errors.append({"split": split, "image": str(image_path), "error": repr(exc)})
    summary = build_summary(instance_rows, image_rows, args.imgsz)
    summary["errors"] = errors
    write_csv(args.output / "instances.csv", instance_rows)
    write_csv(args.output / "images.csv", image_rows)
    if errors:
        write_csv(args.output / "errors.csv", errors)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.output / "REPORT.md", args.dataset, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
