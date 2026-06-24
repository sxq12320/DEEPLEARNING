"""
可视化评估：010 ContourToPollinationNet 训练结果
===============================================
1. 在验证集上逐样本推理
2. 可视化预测授粉点 vs GT授粉点
3. 误差分布统计
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

# ============ 配置 ============
RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
MODEL_PATH = os.path.join(RESULTS_DIR, "10_contour_pollination", "best.pth")
VAL_IMG_DIR = r"E:\mastercode\data\shr_watermelon\segmentation\images\val"
VAL_LABEL_DIR = r"E:\mastercode\data\shr_watermelon\pose\labels\val"
VIS_DIR = os.path.join(RESULTS_DIR, "10_contour_pollination", "visualizations")
NUM_BOUNDARY_POINTS = 64
IMG_SIZE = 960  # 假设宽度，会从json读取实际值


# ============ 网络（必须和训练时一致） ============
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
            nn.Tanh()
        )

    def forward(self, boundary_points, hsv_features):
        boundary_feat = self.boundary_encoder(boundary_points)
        hsv_feat = self.hsv_encoder(hsv_features)
        fused = torch.cat([boundary_feat, hsv_feat], dim=1)
        fused = self.fusion(fused)
        offset = self.predictor(fused)
        return offset


# ============ 工具函数 ============
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
    return np.array([
        np.mean(flower_pixels[:, 0]) / 180.0,
        np.mean(flower_pixels[:, 1]) / 255.0,
        np.mean(flower_pixels[:, 2]) / 255.0
    ], dtype=np.float32)


def draw_visualization(image, contour_pts_norm, flower_center_norm, pred_center_norm,
                       gt_center_norm, error_px, img_w, img_h):
    """绘制单张图的可视化结果"""
    vis = image.copy()

    # 画轮廓点
    if contour_pts_norm is not None:
        for pt in contour_pts_norm:
            px = int(pt[0] * img_w)
            py = int(pt[1] * img_h)
            cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)  # 黄色轮廓点

    # 画花朵中心
    fc_x = int(flower_center_norm[0] * img_w)
    fc_y = int(flower_center_norm[1] * img_h)
    cv2.circle(vis, (fc_x, fc_y), 6, (255, 0, 0), -1)  # 蓝色花朵中心
    cv2.putText(vis, "Center", (fc_x + 8, fc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 画GT授粉点
    gt_x = int(gt_center_norm[0] * img_w)
    gt_y = int(gt_center_norm[1] * img_h)
    cv2.circle(vis, (gt_x, gt_y), 8, (0, 255, 0), 2)  # 绿色圆圈 = GT
    cv2.putText(vis, "GT", (gt_x + 10, gt_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 画预测授粉点
    pred_x = int(pred_center_norm[0] * img_w)
    pred_y = int(pred_center_norm[1] * img_h)
    cv2.circle(vis, (pred_x, pred_y), 8, (0, 0, 255), 2)  # 红色圆圈 = Pred
    cv2.putText(vis, f"Pred", (pred_x + 10, pred_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 画误差线
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (0, 0, 255), 1)

    # 误差信息
    info_text = f"Error: {error_px:.1f}px"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


def main():
    os.makedirs(VIS_DIR, exist_ok=True)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    seg_model = YOLO(SEG_MODEL_PATH)

    model = ContourToPollinationNet(num_boundary_points=NUM_BOUNDARY_POINTS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 获取验证图片
    img_files = sorted([f for f in os.listdir(VAL_IMG_DIR)
                        if f.endswith('.jpg') and not f.startswith('annotations')])
    print(f"验证图片数: {len(img_files)}")

    all_errors = []
    results_data = []

    # 推理并可视化
    for img_file in tqdm(img_files, desc="推理中"):
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        json_name = img_file.replace('.jpg', '.json')
        label_path = os.path.join(VAL_LABEL_DIR, json_name)

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        # YOLO分割
        results = seg_model.predict(img_path, conf=0.25, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for r in results[0].masks:
                mask_data = r.data.cpu().numpy()[0]
                mask_resized = cv2.resize(mask_data, (w, h))
                mask[mask_resized > 0.5] = 255

        # 提取特征
        contour_pts = extract_boundary_points(mask, NUM_BOUNDARY_POINTS)
        hsv_feat = extract_hsv_features(image, mask)

        if contour_pts is None:
            continue

        # 花朵中心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flower_center = np.array([0.5, 0.5])
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                flower_center = np.array([M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h])

        # GT
        gt_center = np.array([0.5, 0.5])
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data['shapes']:
                if shape['label'] == 'fully_visible':
                    gt_center = np.array([shape['points'][0][0] / w, shape['points'][0][1] / h])
                    break

        # 推理
        boundary_tensor = torch.tensor(contour_pts.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        hsv_tensor = torch.tensor(hsv_feat, dtype=torch.float32).unsqueeze(0).to(device)
        flower_center_tensor = torch.tensor(flower_center, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            offset = model(boundary_tensor, hsv_tensor)
            pred_center = flower_center_tensor + offset
            pred_center = pred_center.cpu().numpy()[0]

        # 计算误差
        error_px = np.sqrt(((pred_center - gt_center) * max(w, h)) ** 2).sum()
        all_errors.append(error_px)
        results_data.append({
            'file': img_file,
            'error_px': error_px,
            'pred': pred_center.tolist(),
            'gt': gt_center.tolist(),
            'flower_center': flower_center.tolist(),
        })

        # 可视化
        vis = draw_visualization(image, contour_pts, flower_center, pred_center,
                                 gt_center, error_px, w, h)
        cv2.imwrite(os.path.join(VIS_DIR, img_file), vis)

    # ========== 统计报告 ==========
    all_errors = np.array(all_errors)

    print("\n" + "=" * 60)
    print("评估结果统计")
    print("=" * 60)
    print(f"  总样本数:   {len(all_errors)}")
    print(f"  平均误差:   {np.mean(all_errors):.2f} px")
    print(f"  中位数误差: {np.median(all_errors):.2f} px")
    print(f"  最大误差:   {np.max(all_errors):.2f} px")
    print(f"  最小误差:   {np.min(all_errors):.2f} px")
    print(f"  标准差:     {np.std(all_errors):.2f} px")
    print(f"  <10px:      {np.sum(all_errors < 10)} ({np.sum(all_errors < 10) / len(all_errors) * 100:.1f}%)")
    print(f"  <20px:      {np.sum(all_errors < 20)} ({np.sum(all_errors < 20) / len(all_errors) * 100:.1f}%)")
    print(f"  <30px:      {np.sum(all_errors < 30)} ({np.sum(all_errors < 30) / len(all_errors) * 100:.1f}%)")
    print(f"  <50px:      {np.sum(all_errors < 50)} ({np.sum(all_errors < 50) / len(all_errors) * 100:.1f}%)")

    # 排序打印最差的样本
    results_data.sort(key=lambda x: x['error_px'], reverse=True)
    print(f"\n  误差最大 Top-5:")
    for r in results_data[:5]:
        print(f"    {r['file']}: {r['error_px']:.1f}px")
    print(f"\n  误差最小 Top-5:")
    for r in results_data[-5:]:
        print(f"    {r['file']}: {r['error_px']:.1f}px")

    # 误差分布直方图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(all_errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(all_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors):.1f}px')
    axes[0].axvline(np.median(all_errors), color='orange', linestyle='--', label=f'Median: {np.median(all_errors):.1f}px')
    axes[0].set_xlabel('Error (px)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].legend()

    # 累积分布
    sorted_errors = np.sort(all_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cdf, 'b-', linewidth=2)
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Error (px)')
    axes[1].set_ylabel('Cumulative Proportion')
    axes[1].set_title('CDF of Error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "error_distribution.png"), dpi=150)
    print(f"\n  误差分布图已保存: {VIS_DIR}/error_distribution.png")

    # 保存详细结果JSON
    with open(os.path.join(VIS_DIR, "eval_results.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(all_errors),
                'mean_error_px': float(np.mean(all_errors)),
                'median_error_px': float(np.median(all_errors)),
                'std_error_px': float(np.std(all_errors)),
                'max_error_px': float(np.max(all_errors)),
                'min_error_px': float(np.min(all_errors)),
            },
            'per_sample': results_data
        }, f, indent=2, ensure_ascii=False)

    print(f"  详细结果已保存: {VIS_DIR}/eval_results.json")
    print(f"  可视化图片已保存: {VIS_DIR}/ ({len(all_errors)}张)")


if __name__ == "__main__":
    main()
