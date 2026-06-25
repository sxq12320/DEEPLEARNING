"""
Train the direction-A second-stage model:
YOLO segmentation -> flower ROI image + mask -> pollination keypoint heatmap.
"""

import argparse
import importlib.util
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from keypoint_map_utils import compute_batch_oks, compute_mask_area, summarize_single_keypoint_map


RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
SAVE_DIR = os.path.join(RESULTS_DIR, "13_roi_heatmap_v2")

TRAIN_IMG_DIR = r"E:\mastercode\data\shr_watermelon\segmentation\images\train"
TRAIN_LABEL_DIR = r"E:\mastercode\data\shr_watermelon\pose\labels\train"
VAL_IMG_DIR = r"E:\mastercode\data\shr_watermelon\segmentation\images\val"
VAL_LABEL_DIR = r"E:\mastercode\data\shr_watermelon\pose\labels\val"

MAX_GT_MATCH_DISTANCE_PX = 160
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_roi_heatmap_net_class():
    module_path = os.path.join(os.path.dirname(__file__), "013_improved_net_v2.py")
    spec = importlib.util.spec_from_file_location("improved_net_v2_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ROIHeatmapNet


ROIHeatmapNet = load_roi_heatmap_net_class()


def extract_yolo_masks(seg_model, img_path, img_w, img_h):
    results = seg_model.predict(img_path, conf=0.25, verbose=False)
    if not results or results[0].masks is None:
        return []

    masks = []
    for result_mask in results[0].masks:
        mask_data = result_mask.data.cpu().numpy()[0]
        resized = cv2.resize(mask_data, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        candidate = np.zeros((img_h, img_w), dtype=np.uint8)
        candidate[resized > 0.5] = 255
        area = int(np.count_nonzero(candidate))
        if area > 0:
            masks.append((area, candidate))

    masks.sort(key=lambda item: item[0], reverse=True)
    return [mask for _, mask in masks]


def compute_flower_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flower_center = np.array([0.5, 0.5], dtype=np.float32)
    if not contours:
        return flower_center

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] <= 0:
        return flower_center

    h, w = mask.shape
    return np.array(
        [moments["m10"] / moments["m00"] / w, moments["m01"] / moments["m00"] / h],
        dtype=np.float32,
    )


def mask_to_padded_bbox(mask, margin_ratio=0.25):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    pad_x = int(round(box_w * margin_ratio))
    pad_y = int(round(box_h * margin_ratio))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def select_gt_center(label_path, w, h, flower_center):
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    img_wh = np.array([w, h], dtype=np.float32)
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label = shape.get("label", "")
        if label not in ("fully_visible", "partially_visible", "invisible"):
            continue
        points = shape.get("points") or []
        if not points:
            continue

        point = points[0]
        norm_pt = np.array([point[0] / w, point[1] / h], dtype=np.float32)
        distance_px = float(np.linalg.norm((norm_pt - flower_center) * img_wh))
        candidates.append((distance_px, label, norm_pt))

    if not candidates:
        return None

    distance_px, label, norm_pt = min(candidates, key=lambda item: item[0])
    if label == "invisible" or distance_px > MAX_GT_MATCH_DISTANCE_PX:
        return None
    return norm_pt


def make_gaussian_heatmap(center_xy_norm, heatmap_size, sigma=2.0):
    cx = float(center_xy_norm[0]) * (heatmap_size - 1)
    cy = float(center_xy_norm[1]) * (heatmap_size - 1)

    xs = np.arange(heatmap_size, dtype=np.float32)
    ys = np.arange(heatmap_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heatmap = np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * sigma ** 2))
    peak = float(np.max(heatmap))
    if peak > 0:
        heatmap /= peak
    return heatmap.astype(np.float32)


def build_resized_roi_mask(mask, roi_box, roi_size):
    x1, y1, x2, y2 = roi_box.astype(np.int32).tolist()
    mask_crop = mask[y1:y2, x1:x2]
    return cv2.resize(mask_crop, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)


def preprocess_roi(image, roi_box, roi_mask_resized, roi_size):
    x1, y1, x2, y2 = roi_box.astype(np.int32).tolist()
    image_crop = image[y1:y2, x1:x2]

    image_resized = cv2.resize(image_crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGE_MEAN) / IMAGE_STD
    mask_channel = (roi_mask_resized > 0).astype(np.float32)[..., None]

    roi = np.concatenate([rgb, mask_channel], axis=2)
    return np.transpose(roi, (2, 0, 1)).astype(np.float32)


def gt_to_roi_xy(gt_center_norm, roi_box, img_w, img_h):
    gt_px = np.array([gt_center_norm[0] * img_w, gt_center_norm[1] * img_h], dtype=np.float32)
    x1, y1, x2, y2 = roi_box.astype(np.float32)
    roi_wh = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    if np.any(roi_wh <= 1):
        return None
    roi_xy = (gt_px - np.array([x1, y1], dtype=np.float32)) / roi_wh
    if np.any(roi_xy < 0.0) or np.any(roi_xy > 1.0):
        return None
    return roi_xy.astype(np.float32)


class YOLOROIHeatmapDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        seg_model,
        roi_size=128,
        heatmap_size=64,
        max_samples=None,
        cache=True,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.roi_size = roi_size
        self.heatmap_size = heatmap_size
        self.cache_enabled = cache
        self.cache = {}

        img_files = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg") and not f.startswith("annotations")
        )
        if max_samples is not None and max_samples > 0:
            img_files = img_files[:max_samples]
        self.img_files = img_files
        self.samples = self._index_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache_enabled and idx in self.cache:
            return self.cache[idx]

        sample = self._build_sample(idx)
        if self.cache_enabled:
            self.cache[idx] = sample
        return sample

    def _index_samples(self):
        samples = []
        desc = f"Index {os.path.basename(os.path.normpath(self.img_dir))}"
        for img_file in tqdm(self.img_files, desc=desc, ncols=90):
            img_path = os.path.join(self.img_dir, img_file)
            label_path = os.path.join(self.label_dir, img_file.replace(".jpg", ".json"))

            image = cv2.imread(img_path)
            if image is None:
                continue
            h, w = image.shape[:2]

            masks = extract_yolo_masks(self.seg_model, img_path, w, h)
            for instance_idx, mask in enumerate(masks):
                mask_area = compute_mask_area(mask)
                flower_center = compute_flower_center(mask)
                gt_center = select_gt_center(label_path, w, h, flower_center)
                if gt_center is None:
                    continue

                roi_box = mask_to_padded_bbox(mask)
                if roi_box is None:
                    continue

                gt_roi_xy = gt_to_roi_xy(gt_center, roi_box, w, h)
                if gt_roi_xy is None:
                    continue

                roi_mask = build_resized_roi_mask(mask, roi_box, self.roi_size)
                samples.append(
                    {
                        "img_file": img_file,
                        "img_path": img_path,
                        "instance_idx": instance_idx,
                        "roi_box": roi_box,
                        "roi_mask": roi_mask,
                        "gt_roi_xy": gt_roi_xy,
                        "gt_center": gt_center,
                        "img_wh": np.array([w, h], dtype=np.float32),
                        "mask_area": float(mask_area),
                    }
                )

        return samples

    def _build_sample(self, idx):
        meta = self.samples[idx]
        img_file = meta["img_file"]

        image = cv2.imread(meta["img_path"])
        if image is None:
            return self._dummy(img_file)

        roi = preprocess_roi(image, meta["roi_box"], meta["roi_mask"], self.roi_size)
        heatmap = make_gaussian_heatmap(meta["gt_roi_xy"], self.heatmap_size, sigma=2.0)

        return {
            "roi": torch.tensor(roi, dtype=torch.float32),
            "heatmap": torch.tensor(heatmap[None, :, :], dtype=torch.float32),
            "gt_roi_xy": torch.tensor(meta["gt_roi_xy"], dtype=torch.float32),
            "gt_center": torch.tensor(meta["gt_center"], dtype=torch.float32),
            "roi_box": torch.tensor(meta["roi_box"], dtype=torch.float32),
            "img_wh": torch.tensor(meta["img_wh"], dtype=torch.float32),
            "mask_area": torch.tensor(meta["mask_area"], dtype=torch.float32),
            "valid": True,
            "file": f"{img_file}#{meta['instance_idx']}",
        }

    def _dummy(self, img_file="", img_wh=(1, 1), mask_area=0.0):
        return {
            "roi": torch.zeros(4, self.roi_size, self.roi_size, dtype=torch.float32),
            "heatmap": torch.zeros(1, self.heatmap_size, self.heatmap_size, dtype=torch.float32),
            "gt_roi_xy": torch.zeros(2, dtype=torch.float32),
            "gt_center": torch.zeros(2, dtype=torch.float32),
            "roi_box": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "img_wh": torch.tensor([float(img_wh[0]), float(img_wh[1])], dtype=torch.float32),
            "mask_area": torch.tensor(float(mask_area), dtype=torch.float32),
            "valid": False,
            "file": img_file,
        }


def soft_argmax_2d(logits, beta=20.0):
    b, _, h, w = logits.shape
    flat = logits.reshape(b, -1)
    prob = F.softmax(flat * beta, dim=1)

    y_coords = torch.linspace(0.0, 1.0, h, device=logits.device)
    x_coords = torch.linspace(0.0, 1.0, w, device=logits.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    pred_x = torch.sum(prob * xx[None, :], dim=1)
    pred_y = torch.sum(prob * yy[None, :], dim=1)
    return torch.stack([pred_x, pred_y], dim=1)


def argmax_heatmap_xy(logits):
    b, _, h, w = logits.shape
    flat_idx = torch.argmax(logits.reshape(b, -1), dim=1)
    y = torch.div(flat_idx, w, rounding_mode="floor").float()
    x = (flat_idx % w).float()
    if w > 1:
        x = x / float(w - 1)
    if h > 1:
        y = y / float(h - 1)
    return torch.stack([x, y], dim=1)


def roi_xy_to_image_norm(roi_xy, roi_boxes, img_wh):
    x1y1 = roi_boxes[:, :2]
    roi_wh = torch.clamp(roi_boxes[:, 2:] - roi_boxes[:, :2], min=1.0)
    pred_px = x1y1 + roi_xy * roi_wh
    pred_norm = pred_px / torch.clamp(img_wh, min=1.0)
    return torch.clamp(pred_norm, 0.0, 1.0)


def norm_xy_to_pixel(norm_xy, image_shape):
    h, w = image_shape[:2]
    x = int(round(float(norm_xy[0]) * max(w - 1, 1)))
    y = int(round(float(norm_xy[1]) * max(h - 1, 1)))
    x = int(np.clip(x, 0, max(w - 1, 0)))
    y = int(np.clip(y, 0, max(h - 1, 0)))
    return x, y


def draw_text_with_background(image, text, origin, color, font_scale=0.45, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(origin[0]), int(origin[1])
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    h, w = image.shape[:2]

    x = int(np.clip(x, 0, max(w - text_w - 8, 0)))
    y = int(np.clip(y, text_h + 8, max(h - baseline - 4, text_h + 8)))

    cv2.rectangle(
        image,
        (x - 2, y - text_h - 2),
        (x + text_w + 2, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_point_marker(image, point_xy, color):
    x, y = point_xy
    cv2.circle(image, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 5, color, 1, cv2.LINE_AA)


def draw_prediction_visualization(image, roi_box, pred_center_norm, gt_center_norm, error_px, sample_name):
    vis = image.copy()
    h, w = vis.shape[:2]

    box = np.round(np.asarray(roi_box, dtype=np.float32)).astype(np.int32)
    x1 = int(np.clip(box[0], 0, max(w - 1, 0)))
    y1 = int(np.clip(box[1], 0, max(h - 1, 0)))
    x2 = int(np.clip(box[2], x1 + 1, w))
    y2 = int(np.clip(box[3], y1 + 1, h))

    gt_px = norm_xy_to_pixel(gt_center_norm, vis.shape)
    pred_px = norm_xy_to_pixel(pred_center_norm, vis.shape)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 1, cv2.LINE_AA)
    cv2.line(vis, gt_px, pred_px, (0, 220, 255), 1, cv2.LINE_AA)
    draw_point_marker(vis, gt_px, (0, 220, 0))
    draw_point_marker(vis, pred_px, (0, 0, 255))

    draw_text_with_background(vis, "GT", (gt_px[0] + 6, gt_px[1] - 6), (0, 255, 0))
    draw_text_with_background(vis, "Pred", (pred_px[0] + 6, pred_px[1] + 16), (0, 0, 255))
    draw_text_with_background(
        vis,
        f"{sample_name} | error {error_px:.2f}px",
        (12, 30),
        (255, 255, 255),
        font_scale=0.55,
        thickness=1,
    )
    return vis


def safe_visualization_name(img_file, instance_idx, error_px):
    stem = os.path.splitext(os.path.basename(img_file))[0]
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    err_tag = f"{error_px:.1f}".replace(".", "p")
    return f"{safe_stem}_inst{int(instance_idx):02d}_err{err_tag}px.jpg"


def save_prediction_visualizations(model, dataset, device, save_dir, max_visualizations=0):
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    limit = len(dataset) if max_visualizations is None or max_visualizations <= 0 else max_visualizations
    records = []
    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Visualize", ncols=90):
            if len(records) >= limit:
                break

            meta = dataset.samples[idx]
            sample = dataset[idx]
            if not bool(sample["valid"]):
                continue

            image = cv2.imread(meta["img_path"])
            if image is None:
                continue

            roi = sample["roi"].unsqueeze(0).to(device)
            logits = model(roi)
            pred_roi_xy = argmax_heatmap_xy(logits).cpu()
            pred_center = roi_xy_to_image_norm(
                pred_roi_xy,
                sample["roi_box"].unsqueeze(0),
                sample["img_wh"].unsqueeze(0),
            )[0].numpy()

            gt_center = sample["gt_center"].numpy()
            img_wh = sample["img_wh"].numpy()
            error_px = float(np.linalg.norm((pred_center - gt_center) * img_wh))
            sample_name = str(sample["file"])

            vis = draw_prediction_visualization(
                image,
                meta["roi_box"],
                pred_center,
                gt_center,
                error_px,
                sample_name,
            )

            out_name = safe_visualization_name(meta["img_file"], meta["instance_idx"], error_px)
            out_path = os.path.join(vis_dir, out_name)
            cv2.imwrite(out_path, vis)

            records.append(
                {
                    "file": sample_name,
                    "path": out_path,
                    "error_px": error_px,
                    "gt_center_norm": gt_center.tolist(),
                    "pred_center_norm": pred_center.tolist(),
                    "roi_box": np.asarray(meta["roi_box"], dtype=float).tolist(),
                }
            )

    with open(os.path.join(vis_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return vis_dir, records


def heatmap_coord_loss(logits, target_heatmaps, target_roi_xy):
    pred_heatmaps = torch.sigmoid(logits)
    heatmap_loss = F.mse_loss(pred_heatmaps, target_heatmaps)
    pred_roi_xy = soft_argmax_2d(logits)
    coord_loss = F.smooth_l1_loss(pred_roi_xy, target_roi_xy)
    return heatmap_loss + 0.25 * coord_loss, heatmap_loss.detach(), coord_loss.detach()


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    errors = []
    oks_scores = []

    with torch.no_grad():
        for batch in val_loader:
            valid = batch["valid"].bool()
            if not valid.any():
                continue

            roi = batch["roi"][valid].to(device)
            target_heatmap = batch["heatmap"][valid].to(device)
            target_roi_xy = batch["gt_roi_xy"][valid].to(device)
            gt_center = batch["gt_center"][valid].to(device)
            roi_box = batch["roi_box"][valid].to(device)
            img_wh = batch["img_wh"][valid].to(device)
            mask_area = batch["mask_area"][valid].cpu().numpy()

            logits = model(roi)
            loss, _, _ = heatmap_coord_loss(logits, target_heatmap, target_roi_xy)
            total_loss += float(loss.item())
            total_batches += 1

            pred_roi_xy = argmax_heatmap_xy(logits)
            pred_center = roi_xy_to_image_norm(pred_roi_xy, roi_box, img_wh)

            pixel_errors = torch.norm((pred_center - gt_center) * img_wh, dim=1)
            errors.extend(pixel_errors.cpu().tolist())
            oks_scores.extend(
                compute_batch_oks(
                    pred_center.detach().cpu().numpy(),
                    gt_center.detach().cpu().numpy(),
                    img_wh.detach().cpu().numpy(),
                    mask_area,
                ).tolist()
            )

    mean_loss = total_loss / max(total_batches, 1)
    mean_error = float(np.mean(errors)) if errors else 0.0
    median_error = float(np.median(errors)) if errors else 0.0
    map_metrics = summarize_single_keypoint_map(oks_scores)
    return mean_loss, mean_error, median_error, errors, oks_scores, map_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 60)
    print("Train ROIHeatmapNet")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.save_dir
    print(f"Device: {device}")
    print(f"Segmentation model: {SEG_MODEL_PATH}")
    print(f"Save dir: {save_dir}")

    seg_model = YOLO(SEG_MODEL_PATH)

    train_dataset = YOLOROIHeatmapDataset(
        TRAIN_IMG_DIR,
        TRAIN_LABEL_DIR,
        seg_model,
        roi_size=args.roi_size,
        heatmap_size=args.heatmap_size,
        max_samples=args.max_train_samples,
        cache=not args.no_cache,
    )
    val_dataset = YOLOROIHeatmapDataset(
        VAL_IMG_DIR,
        VAL_LABEL_DIR,
        seg_model,
        roi_size=args.roi_size,
        heatmap_size=args.heatmap_size,
        max_samples=args.max_val_samples,
        cache=not args.no_cache,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Train source images: {len(train_dataset.img_files)}")
    print(f"Val source images: {len(val_dataset.img_files)}")
    print(f"Train ROI samples: {len(train_dataset)}")
    print(f"Val ROI samples: {len(val_dataset)}")
    print(f"ROI size: {args.roi_size}")
    print(f"Heatmap size: {args.heatmap_size}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("No matched ROI samples were found. Check YOLO masks, labels, and GT matching distance.")

    model = ROIHeatmapNet(in_channels=4, base_channels=args.base_channels).to(device)
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Trainable params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.01,
    )

    os.makedirs(save_dir, exist_ok=True)

    best_map = -1.0
    best_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_maps = []

    epoch_pbar = tqdm(range(args.epochs), desc="Training", ncols=110)

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_heatmap_loss = 0.0
        train_coord_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            valid = batch["valid"].bool()
            if not valid.any():
                continue

            roi = batch["roi"][valid].to(device)
            target_heatmap = batch["heatmap"][valid].to(device)
            target_roi_xy = batch["gt_roi_xy"][valid].to(device)

            logits = model(roi)
            loss, heat_loss, coord_loss = heatmap_coord_loss(logits, target_heatmap, target_roi_xy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())
            train_heatmap_loss += float(heat_loss.item())
            train_coord_loss += float(coord_loss.item())
            train_batches += 1

        scheduler.step()

        train_loss /= max(train_batches, 1)
        train_heatmap_loss /= max(train_batches, 1)
        train_coord_loss /= max(train_batches, 1)
        val_loss, mean_error, median_error, _, _, map_metrics = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maps.append(map_metrics["mAP50-95"])

        is_better = (
            map_metrics["mAP50-95"] > best_map
            or (map_metrics["mAP50-95"] == best_map and val_loss < best_loss)
        )
        if is_better:
            best_map = map_metrics["mAP50-95"]
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch + 1,
                    "best_mAP50-95": best_map,
                    "best_val_loss": best_loss,
                },
                os.path.join(save_dir, "best.pth"),
            )

        epoch_pbar.set_postfix(
            {
                "loss": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "err": f"{mean_error:.1f}px",
                "mAP50": f"{map_metrics['mAP50']:.3f}",
                "mAP95": f"{map_metrics['mAP50-95']:.3f}",
            }
        )

    epoch_pbar.close()

    if os.path.exists(os.path.join(save_dir, "best.pth")):
        checkpoint = torch.load(os.path.join(save_dir, "best.pth"), map_location=device)
        model.load_state_dict(checkpoint["model"])

    final_loss, mean_error, median_error, all_errors, all_oks, map_metrics = evaluate(model, val_loader, device)
    all_errors_arr = np.asarray(all_errors, dtype=np.float32)

    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best mAP50-95: {best_map:.4f}")
    print(f"Final val loss: {final_loss:.6f}")
    print(f"Samples: {len(all_errors_arr)}")
    if len(all_errors_arr) > 0:
        print(f"Mean error: {mean_error:.2f} px")
        print(f"Median error: {median_error:.2f} px")
        print(f"<10px: {np.sum(all_errors_arr < 10)} ({np.mean(all_errors_arr < 10) * 100:.1f}%)")
        print(f"<20px: {np.sum(all_errors_arr < 20)} ({np.mean(all_errors_arr < 20) * 100:.1f}%)")
        print(f"<30px: {np.sum(all_errors_arr < 30)} ({np.mean(all_errors_arr < 30) * 100:.1f}%)")
    else:
        print("Mean error: N/A")
        print("Median error: N/A")
    print(f"mAP50: {map_metrics['mAP50']:.4f}")
    print(f"mAP50-95: {map_metrics['mAP50-95']:.4f}")

    visualization_dir, visualization_records = save_prediction_visualizations(
        model,
        val_dataset,
        device,
        save_dir,
        max_visualizations=args.max_visualizations,
    )
    print(f"Visualizations: {visualization_dir} ({len(visualization_records)} images)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(train_losses, label="Train Loss")
        axes[0].plot(val_losses, label="Val Loss")
        axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best {best_epoch + 1}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(val_maps, label="Val mAP50-95")
        axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP50-95")
        axes[1].set_title("Validation mAP")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Skipped training curve: {exc}")

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "ROIHeatmapNet",
                "direction": "YOLO segmentation + ROI RGB/mask + keypoint heatmap",
                "best_epoch": best_epoch + 1,
                "best_val_loss": float(best_loss),
                "best_mAP50-95": float(best_map),
                "final_val_loss": float(final_loss),
                "num_samples": int(len(all_errors_arr)),
                "mean_error_px": float(mean_error) if len(all_errors_arr) > 0 else 0.0,
                "median_error_px": float(median_error) if len(all_errors_arr) > 0 else 0.0,
                "mAP50": map_metrics["mAP50"],
                "mAP50-95": map_metrics["mAP50-95"],
                "oks_mean": map_metrics["oks_mean"],
                "oks_median": map_metrics["oks_median"],
                "ap_by_threshold": map_metrics["ap_by_threshold"],
                "visualization_dir": visualization_dir,
                "num_visualizations": int(len(visualization_records)),
                "params": int(total_params),
                "args": vars(args),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Checkpoint: {os.path.join(save_dir, 'best.pth')}")
    print(f"Results: {os.path.join(save_dir, 'results.json')}")
    print(f"Visualization index: {os.path.join(visualization_dir, 'index.json')}")


if __name__ == "__main__":
    main()
