"""
Visualize and evaluate 010 ContourToPollinationNet on the validation set.
"""

import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

from keypoint_map_utils import compute_mask_area, compute_oks, summarize_single_keypoint_map


RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
MODEL_PATH = os.path.join(RESULTS_DIR, "10_contour_pollination", "best.pth")
VAL_IMG_DIR = r"E:\mastercode\data\shr_watermelon\segmentation\images\val"
VAL_LABEL_DIR = r"E:\mastercode\data\shr_watermelon\pose\labels\val"
VIS_DIR = os.path.join(RESULTS_DIR, "10_contour_pollination", "visualizations")
NUM_BOUNDARY_POINTS = 64
MAX_GT_MATCH_DISTANCE_PX = 160


class ContourToPollinationNet(nn.Module):
    def __init__(self, num_boundary_points=64, hidden_dim=128):
        super().__init__()
        self.num_boundary_points = num_boundary_points
        self.boundary_encoder = nn.Sequential(
            nn.Linear(num_boundary_points * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.hsv_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
        )
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh(),
        )

    def forward(self, boundary_points, hsv_features):
        boundary_feat = self.boundary_encoder(boundary_points)
        hsv_feat = self.hsv_encoder(hsv_features)
        fused = torch.cat([boundary_feat, hsv_feat], dim=1)
        fused = self.fusion(fused)
        return self.predictor(fused)


def extract_boundary_points(mask, num_points=64):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    indices = np.linspace(0, len(contour_points) - 1, num_points).astype(int)
    sampled_points = contour_points[indices]
    h, w = mask.shape
    normalized = sampled_points.astype(np.float32)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized


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


def as_float_point(point):
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    return [float(arr[0]), float(arr[1])]


def draw_visualization(
    image,
    contour_pts_norm,
    flower_center_norm,
    pred_center_norm,
    gt_center_norm,
    error_px,
    img_w,
    img_h,
):
    vis = image.copy()

    if contour_pts_norm is not None:
        for pt in contour_pts_norm:
            px = int(pt[0] * img_w)
            py = int(pt[1] * img_h)
            cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)

    fc_x = int(flower_center_norm[0] * img_w)
    fc_y = int(flower_center_norm[1] * img_h)
    cv2.circle(vis, (fc_x, fc_y), 6, (255, 0, 0), -1)
    cv2.putText(vis, "Center", (fc_x + 8, fc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    pred_x = int(pred_center_norm[0] * img_w)
    pred_y = int(pred_center_norm[1] * img_h)
    cv2.circle(vis, (pred_x, pred_y), 8, (0, 0, 255), 2)
    cv2.putText(vis, "Pred", (pred_x + 10, pred_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if gt_center_norm is not None:
        gt_x = int(gt_center_norm[0] * img_w)
        gt_y = int(gt_center_norm[1] * img_h)
        cv2.circle(vis, (gt_x, gt_y), 8, (0, 255, 0), 2)
        cv2.putText(vis, "GT", (gt_x + 10, gt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (0, 0, 255), 1)

    info_text = f"Error: {error_px:.1f}px" if error_px is not None else "GT skipped"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def summarize_errors(errors):
    if len(errors) == 0:
        return {
            "total": 0,
            "mean_error_px": 0.0,
            "median_error_px": 0.0,
            "std_error_px": 0.0,
            "max_error_px": 0.0,
            "min_error_px": 0.0,
        }

    arr = np.asarray(errors, dtype=np.float32)
    return {
        "total": int(len(arr)),
        "mean_error_px": float(np.mean(arr)),
        "median_error_px": float(np.median(arr)),
        "std_error_px": float(np.std(arr)),
        "max_error_px": float(np.max(arr)),
        "min_error_px": float(np.min(arr)),
    }


def main():
    os.makedirs(VIS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seg_model = YOLO(SEG_MODEL_PATH)
    model = ContourToPollinationNet(num_boundary_points=NUM_BOUNDARY_POINTS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    img_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith(".jpg") and not f.startswith("annotations")])
    print(f"Validation images: {len(img_files)}")

    all_errors = []
    all_oks = []
    results_data = []
    skipped_data = []

    for img_file in tqdm(img_files, desc="Running inference"):
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        label_path = os.path.join(VAL_LABEL_DIR, img_file.replace(".jpg", ".json"))

        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        results = seg_model.predict(img_path, conf=0.25, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for result_mask in results[0].masks:
                mask_data = result_mask.data.cpu().numpy()[0]  # type: ignore
                mask_resized = cv2.resize(mask_data, (w, h))
                mask[mask_resized > 0.5] = 255

        mask_area = compute_mask_area(mask)
        contour_pts = extract_boundary_points(mask, NUM_BOUNDARY_POINTS)
        hsv_feat = extract_hsv_features(image, mask)
        if contour_pts is None:
            skipped_data.append({"file": img_file, "reason": "no contour extracted"})
            continue

        flower_center = compute_flower_center(mask)

        boundary_tensor = torch.tensor(contour_pts.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        hsv_tensor = torch.tensor(hsv_feat, dtype=torch.float32).unsqueeze(0).to(device)
        flower_center_tensor = torch.tensor(flower_center, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            offset = model(boundary_tensor, hsv_tensor)
            pred_center = (flower_center_tensor + offset).cpu().numpy()[0]

        gt_center = select_gt_center(label_path, w, h, flower_center)
        if gt_center is None:
            skipped_data.append(
                {
                    "file": img_file,
                    "reason": "no matched visible pollination point",
                    "pred": as_float_point(pred_center),
                    "flower_center": as_float_point(flower_center),
                }
            )
            vis = draw_visualization(image, contour_pts, flower_center, pred_center, None, None, w, h)
            cv2.imwrite(os.path.join(VIS_DIR, img_file), vis)
            continue

        img_wh = np.array([w, h], dtype=np.float32)
        error_px = float(np.linalg.norm((pred_center - gt_center) * img_wh))
        oks = compute_oks(pred_center, gt_center, img_wh, mask_area)

        all_errors.append(error_px)
        all_oks.append(oks)
        results_data.append(
            {
                "file": img_file,
                "error_px": error_px,
                "oks": float(oks),
                "pred": as_float_point(pred_center),
                "gt": as_float_point(gt_center),
                "flower_center": as_float_point(flower_center),
                "mask_area_px": float(mask_area),
            }
        )

        vis = draw_visualization(image, contour_pts, flower_center, pred_center, gt_center, error_px, w, h)
        cv2.imwrite(os.path.join(VIS_DIR, img_file), vis)

    error_summary = summarize_errors(all_errors)
    map_metrics = summarize_single_keypoint_map(all_oks)

    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print(f"Samples: {error_summary['total']}")
    print(f"Mean error: {error_summary['mean_error_px']:.2f} px")
    print(f"Median error: {error_summary['median_error_px']:.2f} px")
    print(f"Max error: {error_summary['max_error_px']:.2f} px")
    print(f"Min error: {error_summary['min_error_px']:.2f} px")
    print(f"Std error: {error_summary['std_error_px']:.2f} px")

    if error_summary["total"] > 0:
        all_errors_arr = np.asarray(all_errors, dtype=np.float32)
        print(f"<10px: {np.sum(all_errors_arr < 10)} ({np.mean(all_errors_arr < 10) * 100:.1f}%)")
        print(f"<20px: {np.sum(all_errors_arr < 20)} ({np.mean(all_errors_arr < 20) * 100:.1f}%)")
        print(f"<30px: {np.sum(all_errors_arr < 30)} ({np.mean(all_errors_arr < 30) * 100:.1f}%)")
        print(f"<50px: {np.sum(all_errors_arr < 50)} ({np.mean(all_errors_arr < 50) * 100:.1f}%)")

    print(f"mAP50: {map_metrics['mAP50']:.4f}")
    print(f"mAP50-95: {map_metrics['mAP50-95']:.4f}")

    results_data.sort(key=lambda item: item["error_px"], reverse=True)
    if results_data:
        print("\nWorst Top-5:")
        for item in results_data[:5]:
            print(f"  {item['file']}: {item['error_px']:.1f}px, OKS={item['oks']:.4f}")

        print("\nBest Top-5:")
        for item in results_data[-5:]:
            print(f"  {item['file']}: {item['error_px']:.1f}px, OKS={item['oks']:.4f}")

    if all_errors:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_errors_arr = np.asarray(all_errors, dtype=np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(all_errors_arr, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].axvline(np.mean(all_errors_arr), color="red", linestyle="--", label=f"Mean: {np.mean(all_errors_arr):.1f}px")
        axes[0].axvline(
            np.median(all_errors_arr),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(all_errors_arr):.1f}px",
        )
        axes[0].set_xlabel("Error (px)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Error Distribution")
        axes[0].legend()

        sorted_errors = np.sort(all_errors_arr)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1].plot(sorted_errors, cdf, "b-", linewidth=2)
        axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.5)
        axes[1].set_xlabel("Error (px)")
        axes[1].set_ylabel("Cumulative Proportion")
        axes[1].set_title("CDF of Error")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "error_distribution.png"), dpi=150)
        plt.close(fig)
        print(f"Saved plot: {os.path.join(VIS_DIR, 'error_distribution.png')}")

    with open(os.path.join(VIS_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    **error_summary,
                    "skipped": len(skipped_data),
                    "mAP50": map_metrics["mAP50"],
                    "mAP50-95": map_metrics["mAP50-95"],
                    "oks_mean": map_metrics["oks_mean"],
                    "oks_median": map_metrics["oks_median"],
                    "ap_by_threshold": map_metrics["ap_by_threshold"],
                },
                "per_sample": results_data,
                "skipped_sample": skipped_data,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved detailed results: {os.path.join(VIS_DIR, 'eval_results.json')}")
    print(f"Saved visualizations: {VIS_DIR}")


if __name__ == "__main__":
    main()
