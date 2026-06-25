"""
Improved contour-based pollination point localization with CNN + Transformer.
"""

import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from keypoint_map_utils import compute_batch_oks, compute_mask_area, summarize_single_keypoint_map


RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
NUM_BOUNDARY_POINTS = 64
MAX_GT_MATCH_DISTANCE_PX = 160


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BoundaryEncoder1DCNN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=128):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv1d(x)


class PointTransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class ImprovedContourToPollinationNet(nn.Module):
    def __init__(self, num_boundary_points=64, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.num_boundary_points = num_boundary_points

        self.cnn_encoder = BoundaryEncoder1DCNN(in_channels=2, hidden_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=num_boundary_points)
        self.transformer = nn.Sequential(
            *[PointTransformerBlock(d_model, nhead) for _ in range(num_layers)]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.hsv_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.predictor = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, boundary_points, hsv_features):
        batch_size = boundary_points.shape[0]
        pts = boundary_points.view(batch_size, 2, self.num_boundary_points)
        cnn_features = self.cnn_encoder(pts)
        features = cnn_features.permute(0, 2, 1)
        features = self.pos_encoding(features)
        features = self.transformer(features)
        features = self.global_pool(features.permute(0, 2, 1)).squeeze(-1)
        hsv_feat = self.hsv_encoder(hsv_features)
        combined = torch.cat([features, hsv_feat], dim=1)
        return self.predictor(combined)


def extract_boundary_points(mask, num_points=64):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    if len(contour_points) < 3:
        return None

    closed_contour = np.vstack([contour_points, contour_points[0]])
    segment_vectors = np.diff(closed_contour, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative_length[-1])
    if total_length <= 0:
        return None

    target_lengths = np.linspace(0.0, total_length, num_points, endpoint=False, dtype=np.float32)
    sampled_points = np.zeros((num_points, 2), dtype=np.float32)

    for i, target_len in enumerate(target_lengths):
        segment_idx = int(np.searchsorted(cumulative_length, target_len, side="right") - 1)
        segment_idx = min(max(segment_idx, 0), len(segment_lengths) - 1)
        start_len = cumulative_length[segment_idx]
        end_len = cumulative_length[segment_idx + 1]
        ratio = 0.0 if end_len <= start_len else (target_len - start_len) / (end_len - start_len)
        sampled_points[i] = closed_contour[segment_idx] * (1.0 - ratio) + closed_contour[segment_idx + 1] * ratio

    h, w = mask.shape
    sampled_points[:, 0] /= w
    sampled_points[:, 1] /= h
    return sampled_points.flatten()


def extract_hsv_features(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flower_pixels = hsv[mask > 0]
    if len(flower_pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.array(
        [
            np.mean(flower_pixels[:, 0]) / 180.0,
            np.mean(flower_pixels[:, 1]) / 255.0,
            np.mean(flower_pixels[:, 2]) / 255.0,
        ],
        dtype=np.float32,
    )


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


class YOLOSegPollinationDataset(Dataset):
    def __init__(self, img_dir, label_dir, seg_model, num_boundary_points=64):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.num_boundary_points = num_boundary_points
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".jpg") and not f.startswith("annotations")]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            return self._dummy()

        h, w = image.shape[:2]
        results = self.seg_model.predict(img_path, conf=0.25, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for result_mask in results[0].masks:
                mask_data = result_mask.data.cpu().numpy()[0]
                mask_resized = cv2.resize(mask_data, (w, h))
                mask[mask_resized > 0.5] = 255

        mask_area = compute_mask_area(mask)
        boundary = extract_boundary_points(mask, self.num_boundary_points)
        hsv_feat = extract_hsv_features(image, mask)
        flower_center = compute_flower_center(mask)

        label_path = os.path.join(self.label_dir, img_file.replace(".jpg", ".json"))
        gt_center = select_gt_center(label_path, w, h, flower_center)
        gt_valid = gt_center is not None
        if not gt_valid:
            gt_center = np.array([0.0, 0.0], dtype=np.float32)

        return {
            "boundary": torch.tensor(boundary, dtype=torch.float32)
            if boundary is not None
            else torch.zeros(self.num_boundary_points * 2, dtype=torch.float32),
            "hsv": torch.tensor(hsv_feat, dtype=torch.float32),
            "flower_center": torch.tensor(flower_center, dtype=torch.float32),
            "gt_center": torch.tensor(gt_center, dtype=torch.float32),
            "img_wh": torch.tensor([w, h], dtype=torch.float32),
            "mask_area": torch.tensor(mask_area, dtype=torch.float32),
            "valid": bool(boundary is not None and gt_valid),
        }

    def _dummy(self):
        return {
            "boundary": torch.zeros(self.num_boundary_points * 2, dtype=torch.float32),
            "hsv": torch.zeros(3, dtype=torch.float32),
            "flower_center": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "gt_center": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "img_wh": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "mask_area": torch.tensor(0.0, dtype=torch.float32),
            "valid": False,
        }


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("YOLO segmentation + CNN/Transformer pollination localization")
    print("=" * 60)

    print(f"Loading segmentation model: {SEG_MODEL_PATH}")
    seg_model = YOLO(SEG_MODEL_PATH)

    train_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\train",
        r"E:\mastercode\data\shr_watermelon\pose\labels\train",
        seg_model,
        NUM_BOUNDARY_POINTS,
    )
    val_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\val",
        r"E:\mastercode\data\shr_watermelon\pose\labels\val",
        seg_model,
        NUM_BOUNDARY_POINTS,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedContourToPollinationNet(num_boundary_points=NUM_BOUNDARY_POINTS).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    save_dir = os.path.join(RESULTS_DIR, "11_cnn_transformer_pollination")
    os.makedirs(save_dir, exist_ok=True)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Trainable params: {params:,}")

    best_loss = float("inf")
    epoch_pbar = tqdm(range(100), desc="Training", ncols=100)

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=80, leave=False):
            valid = batch["valid"]
            if not valid.any():
                continue

            boundary = batch["boundary"][valid].to(device)
            hsv = batch["hsv"][valid].to(device)
            flower_center = batch["flower_center"][valid].to(device)
            gt_center = batch["gt_center"][valid].to(device)

            offset = model(boundary, hsv)
            pred_center = flower_center + offset
            loss = criterion(pred_center, gt_center)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        model.eval()
        val_loss = 0.0
        val_count = 0
        errors = []
        oks_scores = []

        with torch.no_grad():
            for batch in val_loader:
                valid = batch["valid"]
                if not valid.any():
                    continue

                boundary = batch["boundary"][valid].to(device)
                hsv = batch["hsv"][valid].to(device)
                flower_center = batch["flower_center"][valid].to(device)
                gt_center = batch["gt_center"][valid].to(device)
                img_wh = batch["img_wh"][valid].to(device)
                mask_area = batch["mask_area"][valid].cpu().numpy()

                offset = model(boundary, hsv)
                pred_center = flower_center + offset

                loss = criterion(pred_center, gt_center)
                val_loss += loss.item()
                val_count += 1

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

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        mean_error = float(np.mean(errors)) if errors else 0.0
        map_metrics = summarize_single_keypoint_map(oks_scores)

        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "error": f"{mean_error:.1f}px",
                "mAP50": f"{map_metrics['mAP50']:.3f}",
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

    epoch_pbar.close()

    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth"), map_location=device))
    model.eval()

    all_errors = []
    all_oks = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=80):
            valid = batch["valid"]
            if not valid.any():
                continue

            boundary = batch["boundary"][valid].to(device)
            hsv = batch["hsv"][valid].to(device)
            flower_center = batch["flower_center"][valid].to(device)
            gt_center = batch["gt_center"][valid].to(device)
            img_wh = batch["img_wh"][valid].to(device)
            mask_area = batch["mask_area"][valid].cpu().numpy()

            offset = model(boundary, hsv)
            pred_center = flower_center + offset

            pixel_errors = torch.norm((pred_center - gt_center) * img_wh, dim=1)
            all_errors.extend(pixel_errors.cpu().tolist())
            all_oks.extend(
                compute_batch_oks(
                    pred_center.detach().cpu().numpy(),
                    gt_center.detach().cpu().numpy(),
                    img_wh.detach().cpu().numpy(),
                    mask_area,
                ).tolist()
            )

    all_errors = np.asarray(all_errors, dtype=np.float32)
    map_metrics = summarize_single_keypoint_map(all_oks)

    print(f"Samples: {len(all_errors)}")
    if len(all_errors) > 0:
        print(f"Mean error: {np.mean(all_errors):.2f} px")
        print(f"Median error: {np.median(all_errors):.2f} px")
        print(f"<10px: {np.sum(all_errors < 10)} ({np.mean(all_errors < 10) * 100:.1f}%)")
        print(f"<20px: {np.sum(all_errors < 20)} ({np.mean(all_errors < 20) * 100:.1f}%)")
    else:
        print("Mean error: N/A")
        print("Median error: N/A")
        print("<10px: 0 (0.0%)")
        print("<20px: 0 (0.0%)")
    print(f"mAP50: {map_metrics['mAP50']:.4f}")
    print(f"mAP50-95: {map_metrics['mAP50-95']:.4f}")
    print(f"Best checkpoint: {os.path.join(save_dir, 'best.pth')}")

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": int(len(all_errors)),
                "mean_error_px": float(np.mean(all_errors)) if len(all_errors) > 0 else 0.0,
                "median_error_px": float(np.median(all_errors)) if len(all_errors) > 0 else 0.0,
                "mAP50": map_metrics["mAP50"],
                "mAP50-95": map_metrics["mAP50-95"],
                "oks_mean": map_metrics["oks_mean"],
                "oks_median": map_metrics["oks_median"],
                "ap_by_threshold": map_metrics["ap_by_threshold"],
                "params": params,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
