"""Read-only J research audit: actual dataset geometry, supervision and counterexamples.

No training, dataset edits, checkpoint inference, or production-model changes.
Geometry is square letterbox640 without random augmentation: not training recall.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import sympy as sp
import torch
import torch.nn.functional as F

from ultralytics.data.utils import polygon2mask, polygons2masks_overlap
from ultralytics.nn.modules.citrus_j import JContext
from ultralytics.utils.metrics import compute_ap
from ultralytics.utils.tal import TaskAlignedAssigner


def mathematical_checks():
    """Verify exact identities and executable examples, not efficacy claims."""
    d, delta, tau = sp.symbols("d delta tau", positive=True)
    square_iou = (d - delta) ** 2 / (2 * d**2 - (d - delta) ** 2)
    a1, a2, v1, v2 = sp.symbols("a1 a2 v1 v2", real=True)
    mixture = (sp.exp(a1) * v1 + sp.exp(a2) * v2) / (sp.exp(a1) + sp.exp(a2))
    derivative = sp.exp(a1) / (sp.exp(a1) + sp.exp(a2)) * (v1 - mixture)
    assert sp.simplify(sp.diff(mixture, a1) - derivative) == 0
    assert sp.simplify(square_iou.subs(delta, d * (1 - sp.sqrt(2 * tau / (1 + tau)))) - tau) == 0
    # A signed zero-row-sum read removes a CONSTANT foreground as well as background.
    signed_read = sp.Matrix([[1, -1]])
    constant = sp.Matrix([[7], [7]])
    assert signed_read * constant == sp.zeros(1, 1)
    compression = sp.Matrix([[1, 1]])
    null_vector = sp.Matrix([[1], [-1]])
    assert compression * null_vector == sp.zeros(1, 1)
    torch.manual_seed(42)
    context = JContext(8, mode=2, grid=2, dim=4).eval()
    context(torch.randn(1, 8, 4, 4)).square().sum().backward()
    gate_grad = float(context.gain.grad.abs().sum())
    branch_grad = float(context.qk.weight.grad.abs().sum())
    assert gate_grad > 0 and branch_grad == 0
    mean_loss = float(F.binary_cross_entropy_with_logits(torch.zeros(1), torch.ones(1)))
    per_anchor_loss = float(F.binary_cross_entropy_with_logits(torch.tensor([8.0, -8.0]), torch.ones(2)))
    recall = np.array([0.2, 0.5, 0.8])
    precision = np.array([1.0, 0.9, 0.8])
    upstream_style_ap, _, _ = compute_ap(recall, precision)
    # Conventional endpoint interpolation for comparison only; NOT COCOeval.
    x = np.linspace(0, 1, 101)
    reference_r = np.r_[0.0, recall, 1.0]
    reference_p = np.maximum.accumulate(np.r_[1.0, precision, 0.0][::-1])[::-1]
    interpolated_ap = float(np.trapz(np.interp(x, reference_r, reference_p), x))
    return {
        "sympy_version": sp.__version__,
        "fusion_gradient_residual": "0",
        "square_iou": str(square_iou),
        "one_pixel_xy_shift_iou": {str(size): str(square_iou.subs({d: size, delta: 1})) for size in (4, 8, 16)},
        "max_shift_over_side_at_iou05": str(1 - sp.sqrt(sp.Rational(2, 3))),
        "constant_signal_difference_read": 0,
        "compression_null_vector": [1, -1],
        "zero_initialized_context_first_backward": {"gain_gradient_l1": gate_grad, "qk_gradient_l1": branch_grad},
        "anchor_averaging_counterexample": {"BCE_at_mean_logit": mean_loss, "mean_anchor_BCE": per_anchor_loss},
        "AP_endpoint_toy_only": {
            "current_ultralytics_style_AP": float(upstream_style_ap),
            "conventional_interpolation_AP": interpolated_ap,
            "scope": "Current official main has the same endpoint; this is not a detected local bug or COCOeval.",
        },
        "scope": "Exact algebra and constructed examples; no proof of AP, recall, real foliage cancellation or convergence.",
    }


def audit_dataset(dataset, output):
    """Count current files and audit train/val only; never select by test outcomes."""
    manifest_file = dataset / "audit/split_manifest.csv"
    with manifest_file.open(encoding="utf-8-sig", newline="") as handle:
        manifest = {(r["new_split"], r["image"]): r for r in csv.DictReader(handle)}
    expected = set(manifest)
    actual = set()
    summary, records, examples, groups, content_members = {}, [], {}, {}, {}
    identity = hashlib.sha256()
    anchors = []
    for stride in (8, 16, 32):
        yy, xx = np.meshgrid(np.arange(640 // stride), np.arange(640 // stride), indexing="ij")
        anchors.append(np.stack((xx, yy), -1).reshape(-1, 2) * stride + stride / 2)
    anchors = torch.tensor(np.concatenate(anchors), dtype=torch.float32)
    assigner = TaskAlignedAssigner(stride=[8, 16, 32])
    for split in ("train", "val", "test"):
        counts, shapes, cohorts, cohort_instances = Counter(), Counter(), Counter(), Counter()
        for image in sorted((dataset / split / "images").iterdir()):
            if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            actual.add((split, image.name))
            entry = manifest[(split, image.name)]
            group = entry["group_id"]
            groups.setdefault(group, set()).add(split)
            image_digest = hashlib.sha256(image.read_bytes()).hexdigest()
            content_members.setdefault(image_digest, []).append([split, image.name])
            cohorts[entry["cohort"]] += 1
            label = dataset / split / "labels" / (image.stem + ".txt")
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            label_digest = hashlib.sha256(label.read_bytes()).hexdigest()
            identity.update(f"{split}/{image.name}\t{image_digest}\t{label_digest}\n".encode("utf-8"))
            with Image.open(image) as original:
                width, height = original.size
            counts["images"] += 1
            counts["instances"] += len(lines)
            counts["empty_images"] += not bool(lines)
            cohort_instances[entry["cohort"]] += len(lines)
            shapes[f"{width}x{height}"] += 1
            # TEST: count membership only; no difficulty/proxy performance stratification.
            if split == "test" or not lines:
                continue
            scale = min(640 / width, 640 / height)
            pad = np.array(
                [round((640 - round(width * scale)) / 2 - 0.1), round((640 - round(height * scale)) / 2 - 0.1)]
            )
            polygons = []
            for line in lines:
                fields = np.asarray([float(value) for value in line.split()])
                if fields[0] != 0 or len(fields[1:]) < 6 or len(fields[1:]) % 2:
                    raise ValueError(f"Audit cannot interpret polygon: {label}")
                polygon = fields[1:].reshape(-1, 2) * [width, height] * scale + pad
                polygons.append(polygon)
            ids_by_ratio = {}
            for ratio in (2, 4):
                combined, order = polygons2masks_overlap((640, 640), polygons, downsample_ratio=ratio)
                ids_by_ratio[ratio] = {int(order[k - 1]) for k in np.unique(combined) if k > 0}
            for index, polygon in enumerate(polygons):
                box = torch.tensor(np.r_[polygon.min(0), polygon.max(0)], dtype=torch.float32)
                sides = box[2:] - box[:2]
                raw = ((anchors > box[:2]) & (anchors < box[2:])).all(1).sum().item()
                eligible = assigner.select_candidates_in_gts(anchors, box[None, None], torch.ones(1, 1, 1))
                row = {
                    "split": split,
                    "image": image.name,
                    "instance": index,
                    "cohort": entry["cohort"],
                    "min_side_640": float(sides.min()),
                    "box_area_640": float(sides.prod()),
                    "raw_candidate_centers": int(raw),
                    "expanded_candidate_centers": int(eligible.sum()),
                }
                for ratio in (2, 4):
                    binary = polygon2mask((640, 640), [polygon.reshape(-1)], downsample_ratio=ratio)
                    row[f"individual_pixels_r{ratio}"] = int(binary.sum())
                    row[f"retained_overlap_r{ratio}"] = index in ids_by_ratio[ratio]
                    counts[f"lost_individual_r{ratio}"] += not bool(binary.any())
                    counts[f"lost_overlap_r{ratio}"] += not row[f"retained_overlap_r{ratio}"]
                counts["min_side_lt8"] += float(sides.min()) < 8
                counts["min_side_lt16"] += float(sides.min()) < 16
                counts["no_raw_center"] += raw == 0
                counts["no_expanded_center"] += int(eligible.sum()) == 0
                counts["tiny_with_1to4_pixels_r2"] += float(sides.min()) < 16 and 0 < row["individual_pixels_r2"] <= 4
                records.append(row)
                # Deterministic diagnostic examples from VAL; no cherry-picked predictions.
                if split == "val" and float(sides.min()) < 8 and len(examples) < 6 and image.name not in examples:
                    examples[image.name] = (image, polygon / scale - pad / scale, width, height)
        summary[split] = {
            "counts": dict(counts),
            "cohorts_images": dict(cohorts),
            "cohorts_instances": dict(cohort_instances),
            "image_shapes": dict(shapes),
        }
        print(split, summary[split]["counts"], flush=True)
    with (output / "instance_supervision_geometry.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Diagnostic original-pixel crops; nearest enlargement is marked, not new image information.
    montage = Image.new("RGB", (900, 600), "white")
    drawing = ImageDraw.Draw(montage)
    for i, (name, (image, polygon, width, height)) in enumerate(examples.items()):
        low, high = polygon.min(0), polygon.max(0)
        center = (low + high) / 2
        crop_size = max(80, int(max(high - low) * 3))
        left, top = max(0, int(center[0] - crop_size / 2)), max(0, int(center[1] - crop_size / 2))
        right, bottom = min(width, left + crop_size), min(height, top + crop_size)
        with Image.open(image) as original:
            crop = original.convert("RGB").crop((left, top, right, bottom))
        painter = ImageDraw.Draw(crop)
        painter.line(
            [tuple(point - [left, top]) for point in polygon] + [tuple(polygon[0] - [left, top])],
            fill=(255, 80, 20),
            width=1,
        )
        crop = crop.resize((280, 260), Image.Resampling.NEAREST)
        x, y = i % 3 * 300 + 10, i // 3 * 300 + 25
        montage.paste(crop, (x, y))
        drawing.text((x, y - 18), f"VAL {name}: original GT outline", fill="black")
        drawing.text((x, y + 264), "Original crop; nearest display enlargement", fill="black")
    montage.save(output / "val_tiny_gt_context.png")
    return {
        "dataset": str(dataset),
        "scope": "Square letterbox640; no augmentation; geometry not recall; test counts only",
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "manifest_missing_files": sorted(expected - actual),
        "manifest_extra_files": sorted(actual - expected),
        "cross_split_groups": {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1},
        "byte_identical_images_across_splits": {
            digest: members for digest, members in content_members.items() if len({split for split, _ in members}) > 1
        },
        "image_and_label_identity_sha256": identity.hexdigest(),
        "current_TAL_min_side_expansion": assigner.stride_val,
        "splits": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new audit output folder")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    result = {"mathematics": mathematical_checks(), "dataset": audit_dataset(args.dataset, args.output)}
    code_root = Path(__file__).resolve().parents[1]
    result["inspected_source_sha256"] = {
        relative: hashlib.sha256((code_root / relative).read_bytes()).hexdigest()
        for relative in (
            "ultralytics/nn/modules/citrus_j.py",
            "ultralytics/utils/citrus_j_loss.py",
            "ultralytics/utils/loss.py",
            "ultralytics/utils/tal.py",
            "ultralytics/utils/metrics.py",
            "ultralytics/data/utils.py",
            "ultralytics/data/augment.py",
        )
    }
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["mathematics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
