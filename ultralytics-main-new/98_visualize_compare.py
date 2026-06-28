"""
Visualize GT and second-stage keypoint predictions on one image.

Pipeline:
YOLO segmentation -> matched flower ROI -> ROI heatmap models -> GT/pred overlay.
"""

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_IMAGE_PATH = Path(r"E:\mastercode\data\shr_watermelon\segmentation\images\val\dsc00005.jpg")
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "98_visualize_compare"

MODEL_SPECS = {
    "013": {
        "module": "013_improved_net_v2.py",
        "weights": RESULTS_DIR / "13_roi_heatmap_v2" / "best.pth",
        "base_arg": "base_channels",
        "default_base": 32,
        "color": (255, 144, 0),
        "marker": "circle",
    },
    "014": {
        "module": "014_improved_net_v2.py",
        "weights": RESULTS_DIR / "14_roi_heatmap_lite" / "best.pth",
        "base_arg": "base_channels",
        "default_base": 16,
        "color": (255, 0, 220),
        "marker": "square",
    },
    "015": {
        "module": "015_improved_net_v2.py",
        "weights": RESULTS_DIR / "15_roi_heatmap_distill" / "best.pth",
        "base_arg": "student_base_channels",
        "default_base": 8,
        "color": (0, 0, 255),
        "marker": "diamond",
    },
}

GT_COLOR = (0, 180, 0)
BOX_COLOR = (255, 190, 0)


def load_module(module_name, filename):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage14 = load_module("stage14_train_utils", "014_train_improved_v2.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GT vs ROI keypoint predictions.")
    parser.add_argument("--image-path", default=str(DEFAULT_IMAGE_PATH), help="Image to visualize.")
    parser.add_argument("--label-path", default="", help="Labelme JSON path. If empty, infer from --label-dir.")
    parser.add_argument("--label-dir", default=stage14.VAL_LABEL_DIR, help="Directory containing Labelme JSON labels.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seg-model-path", default=stage14.SEG_MODEL_PATH, help="YOLO segmentation weights.")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="YOLO segmentation confidence threshold.")
    parser.add_argument("--candidate-class-ids", default="0,3", help="YOLO classes for stage 2, or 'all'.")
    parser.add_argument("--models", default="013,014,015", help="Comma-separated model ids: 013,014,015.")
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--max-instances", type=int, default=0, help="0 means all matched instances.")
    parser.add_argument("--show", action="store_true", help="Open an interactive preview window.")
    return parser.parse_args()


def resolve_label_path(args):
    if args.label_path:
        return Path(args.label_path)
    return Path(args.label_dir) / Path(args.image_path).with_suffix(".json").name


def checkpoint_args(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def load_roi_model(model_id, device, heatmap_size):
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unknown model id '{model_id}'. Available: {', '.join(MODEL_SPECS)}")

    spec = MODEL_SPECS[model_id]
    weights = Path(spec["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Weights for {model_id} not found: {weights}")

    module = load_module(f"net_{model_id}", spec["module"])
    checkpoint = torch.load(weights, map_location=device)
    args = checkpoint_args(checkpoint)
    base_channels = int(args.get(spec["base_arg"], spec["default_base"]))

    net_kwargs = {"in_channels": 4, "base_channels": base_channels}
    if "output_size" in inspect.signature(module.ROIHeatmapNet).parameters:
        net_kwargs["output_size"] = heatmap_size

    model = module.ROIHeatmapNet(**net_kwargs).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, {"base_channels": base_channels, "weights": str(weights), **spec}


def norm_to_pixel(norm_xy, width, height):
    x = int(round(float(norm_xy[0]) * max(width - 1, 1)))
    y = int(round(float(norm_xy[1]) * max(height - 1, 1)))
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


def predict_one(model, image, mask, roi_box, roi_size, device):
    roi_mask = stage14.build_resized_roi_mask(mask, roi_box, roi_size)
    roi = stage14.preprocess_roi(image, roi_box, roi_mask, roi_size)
    roi_tensor = torch.from_numpy(roi).unsqueeze(0).to(device)
    roi_box_tensor = torch.tensor(roi_box, dtype=torch.float32, device=device).unsqueeze(0)
    img_wh_tensor = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(roi_tensor)
        pred_roi_xy = stage14.soft_argmax_2d(logits)
        pred_norm = stage14.roi_xy_to_image_norm(pred_roi_xy, roi_box_tensor, img_wh_tensor)[0]
    return pred_norm.detach().cpu().numpy().astype(float)


def draw_tiny_text(image, text, xy, color, scale=0.35):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(xy[0]), int(xy[1])
    text_size, baseline = cv2.getTextSize(text, font, scale, 1)
    tw, th = text_size
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, max(w - tw - 3, 0)))
    y = int(np.clip(y, th + 3, max(h - baseline - 2, th + 3)))
    cv2.rectangle(image, (x - 1, y - th - 2), (x + tw + 1, y + baseline + 1), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_marker(image, point, color, marker="circle"):
    x, y = point
    if marker == "square":
        cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), color, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3), (255, 255, 255), 1, cv2.LINE_AA)
    elif marker == "diamond":
        pts = np.array([[x, y - 3], [x + 3, y], [x, y + 3], [x - 3, y]], dtype=np.int32)
        cv2.fillConvexPoly(image, pts, color, lineType=cv2.LINE_AA)
        cv2.polylines(image, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.circle(image, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)


def draw_legend(image, loaded_specs):
    x, y = 8, 16
    draw_marker(image, (x + 5, y - 4), GT_COLOR, "circle")
    draw_tiny_text(image, "GT", (x + 15, y), GT_COLOR)
    y += 16
    for model_id, spec in loaded_specs.items():
        draw_marker(image, (x + 5, y - 4), spec["color"], spec["marker"])
        draw_tiny_text(image, model_id, (x + 15, y), spec["color"])
        y += 16


def draw_results(image, records, loaded_specs):
    vis = image.copy()
    h, w = vis.shape[:2]

    for record in records:
        box = np.asarray(record["roi_box"], dtype=np.float32).round().astype(int)
        x1 = int(np.clip(box[0], 0, max(w - 1, 0)))
        y1 = int(np.clip(box[1], 0, max(h - 1, 0)))
        x2 = int(np.clip(box[2], x1 + 1, w))
        y2 = int(np.clip(box[3], y1 + 1, h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 1, cv2.LINE_AA)
        draw_tiny_text(vis, f"#{record['instance_index']} {record['class_name']} {record['confidence']:.2f}", (x1, y1 - 3), BOX_COLOR)

        gt_px = tuple(record["gt_px"])
        draw_marker(vis, gt_px, GT_COLOR, "circle")

        for model_id, pred in record["predictions"].items():
            spec = loaded_specs[model_id]
            pred_px = tuple(pred["pred_px"])
            cv2.line(vis, gt_px, pred_px, (180, 180, 180), 1, cv2.LINE_AA)
            draw_marker(vis, pred_px, spec["color"], spec["marker"])

    draw_legend(vis, loaded_specs)
    return vis


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    label_path = resolve_label_path(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]

    gt_points = stage14.load_gt_points(str(label_path), w, h)
    if not gt_points:
        raise RuntimeError(f"No visible GT points found in label: {label_path}")

    candidate_class_ids = stage14.parse_candidate_class_ids(args.candidate_class_ids)
    seg_model = YOLO(args.seg_model_path)
    instances = stage14.extract_yolo_instances(
        seg_model,
        str(image_path),
        w,
        h,
        candidate_class_ids=candidate_class_ids,
        conf=args.seg_conf,
    )
    masks = [instance["mask"] for instance in instances]
    matches = stage14.match_masks_to_gt(masks, gt_points, w, h)
    if args.max_instances and args.max_instances > 0:
        matches = matches[: args.max_instances]
    if not matches:
        raise RuntimeError("No YOLO masks matched GT points. Check segmentation weights, classes, and labels.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    loaded_models = {}
    loaded_specs = {}
    for model_id in requested_models:
        model, spec = load_roi_model(model_id, device, args.heatmap_size)
        loaded_models[model_id] = model
        loaded_specs[model_id] = spec

    img_wh = np.array([w, h], dtype=np.float32)
    records = []
    for out_idx, (mask_idx, gt_idx) in enumerate(matches):
        instance = instances[mask_idx]
        mask = instance["mask"]
        roi_box = stage14.mask_to_padded_bbox(mask)
        if roi_box is None:
            continue

        gt_norm = gt_points[gt_idx]["norm_pt"].astype(float)
        gt_px = norm_to_pixel(gt_norm, w, h)
        predictions = {}
        for model_id, model in loaded_models.items():
            pred_norm = predict_one(model, image, mask, roi_box, args.roi_size, device)
            pred_px = norm_to_pixel(pred_norm, w, h)
            error_px = float(np.linalg.norm((pred_norm - gt_norm) * img_wh))
            predictions[model_id] = {
                "pred_norm": pred_norm.tolist(),
                "pred_px": [int(pred_px[0]), int(pred_px[1])],
                "error_px": error_px,
            }

        records.append(
            {
                "instance_index": int(out_idx),
                "mask_index": int(mask_idx),
                "gt_index": int(gt_idx),
                "gt_label": gt_points[gt_idx]["label"],
                "gt_norm": gt_norm.tolist(),
                "gt_px": [int(gt_px[0]), int(gt_px[1])],
                "roi_box": np.asarray(roi_box, dtype=float).tolist(),
                "class_id": int(instance["class_id"]),
                "class_name": instance["class_name"],
                "confidence": float(instance["confidence"]),
                "predictions": predictions,
            }
        )

    vis = draw_results(image, records, loaded_specs)
    stem = image_path.stem
    out_image = output_dir / f"{stem}_gt_pred_compare.jpg"
    out_json = output_dir / f"{stem}_gt_pred_compare.json"
    cv2.imwrite(str(out_image), vis)

    summary = {
        "image_path": str(image_path),
        "label_path": str(label_path),
        "seg_model_path": args.seg_model_path,
        "seg_conf": args.seg_conf,
        "candidate_classes": stage14.format_candidate_class_ids(candidate_class_ids),
        "num_detected_instances": len(instances),
        "num_gt_points": len(gt_points),
        "num_matches": len(records),
        "device": str(device),
        "models": {
            model_id: {
                "weights": spec["weights"],
                "base_channels": spec["base_channels"],
            }
            for model_id, spec in loaded_specs.items()
        },
        "records": records,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved visualization: {out_image}")
    print(f"Saved metrics JSON: {out_json}")
    for model_id in loaded_models:
        errors = [record["predictions"][model_id]["error_px"] for record in records]
        print(f"{model_id}: mean_error={np.mean(errors):.2f}px median_error={np.median(errors):.2f}px n={len(errors)}")

    if args.show:
        cv2.imshow("GT vs predictions", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
