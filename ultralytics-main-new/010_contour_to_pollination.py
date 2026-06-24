"""
完整流程：YOLO分割 + 轮廓提取 + 深度学习定位授粉点
====================================================
1. YOLO分割模型得到花朵掩膜
2. 从掩膜提取轮廓边界点
3. 轮廓点 + HSV特征 → 深度学习网络 → 授粉点
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============ 配置 ============
RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
NUM_BOUNDARY_POINTS = 64
MAX_GT_MATCH_DISTANCE_PX = 160


# ============ 网络 ============
class ContourToPollinationNet(nn.Module):
    """从轮廓边界点 + HSV特征预测授粉点偏移量"""
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
    """从掩膜提取轮廓边界点（固定长度）"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    
    # 均匀采样到固定点数
    indices = np.linspace(0, len(contour_points)-1, num_points).astype(int)
    sampled_points = contour_points[indices]
    
    # 归一化到[0,1]
    h, w = mask.shape
    normalized = sampled_points.astype(np.float32)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    
    return normalized.flatten()


def extract_hsv_features(image, mask):
    """从花朵区域提取HSV颜色特征"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flower_pixels = hsv[mask > 0]
    
    if len(flower_pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    
    return np.array([
        np.mean(flower_pixels[:, 0]) / 180.0,
        np.mean(flower_pixels[:, 1]) / 255.0,
        np.mean(flower_pixels[:, 2]) / 255.0
    ], dtype=np.float32)


def select_gt_center(label_path, w, h, flower_center):
    """Select the pollination point that belongs to the current segmented flower."""
    if not os.path.exists(label_path):
        return None

    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    candidates = []
    img_wh = np.array([w, h], dtype=np.float32)
    for shape in data.get('shapes', []):
        if shape.get('shape_type') != 'point':
            continue
        lbl = shape.get('label', '')
        if lbl not in ('fully_visible', 'partially_visible', 'invisible'):
            continue
        points = shape.get('points') or []
        if not points:
            continue

        pt = points[0]
        norm_pt = np.array([pt[0] / w, pt[1] / h], dtype=np.float32)
        distance_px = float(np.linalg.norm((norm_pt - flower_center) * img_wh))
        candidates.append((distance_px, lbl, norm_pt))

    if not candidates:
        return None

    distance_px, lbl, norm_pt = min(candidates, key=lambda item: item[0])
    if lbl == 'invisible' or distance_px > MAX_GT_MATCH_DISTANCE_PX:
        return None
    return norm_pt


# ============ 数据集（使用YOLO分割） ============
class YOLOSegPollinationDataset(Dataset):
    """使用YOLO分割模型生成掩膜的数据集"""
    def __init__(self, img_dir, label_dir, seg_model, num_boundary_points=64, show_progress=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.num_boundary_points = num_boundary_points
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.cache = {}
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        # 读取图像
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        # YOLO分割得到掩膜
        results = self.seg_model.predict(img_path, conf=0.25, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for r in results[0].masks:
                mask_data = r.data.cpu().numpy()[0]
                mask_resized = cv2.resize(mask_data, (w, h))
                mask[mask_resized > 0.5] = 255
        
        # 提取特征
        boundary = extract_boundary_points(mask, self.num_boundary_points)
        hsv_feat = extract_hsv_features(image, mask)
        
        # 计算花朵中心（用于偏移量计算）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flower_center = np.array([0.5, 0.5])
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                flower_center = np.array([M["m10"]/M["m00"]/w, M["m01"]/M["m00"]/h])

        # 读取GT授粉点（优先 fully_visible > partially_visible，多候选取离花朵中心最近的）
        json_name = img_file.replace('.jpg', '.json')
        label_path = os.path.join(self.label_dir, json_name)
        gt_center = select_gt_center(label_path, w, h, flower_center)
        gt_valid = gt_center is not None
        if not gt_valid:
            gt_center = np.array([0.0, 0.0], dtype=np.float32)

        sample = {
            'boundary': torch.tensor(boundary) if boundary is not None else torch.zeros(self.num_boundary_points * 2),
            'hsv': torch.tensor(hsv_feat, dtype=torch.float32),
            'flower_center': torch.tensor(flower_center, dtype=torch.float32),
            'gt_center': torch.tensor(gt_center, dtype=torch.float32),
            'img_wh': torch.tensor([w, h], dtype=torch.float32),
            'valid': boundary is not None and gt_valid
        }
        self.cache[idx] = sample
        return sample


# ============ 训练 ============
def main():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 60)
    print("YOLO分割 + 轮廓提取 + 深度学习定位授粉点")
    print("=" * 60)
    
    # 1. 加载YOLO分割模型
    print(f"\n加载分割模型: {SEG_MODEL_PATH}")
    seg_model = YOLO(SEG_MODEL_PATH)
    
    # 2. 创建数据集
    print("创建数据集（使用YOLO分割生成掩膜）...")
    print("正在为训练集生成分割掩膜...")
    train_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\train",
        r"E:\mastercode\data\shr_watermelon\pose\labels\train",
        seg_model, NUM_BOUNDARY_POINTS
    )
    val_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\val",
        r"E:\mastercode\data\shr_watermelon\pose\labels\val",
        seg_model, NUM_BOUNDARY_POINTS
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 3. 创建网络
    model = ContourToPollinationNet(num_boundary_points=NUM_BOUNDARY_POINTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 4. 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    save_dir = os.path.join(RESULTS_DIR, "10_contour_pollination")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始训练... 设备: {device}")
    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    
    best_loss = float('inf')
    
    # 训练循环
    epoch_pbar = tqdm(range(100), desc="训练进度", ncols=100)
    
    for epoch in epoch_pbar:
        # 训练
        model.train()
        train_loss = 0
        train_count = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/100", ncols=80, leave=False)
        
        for batch in batch_pbar:
            if not batch['valid'].any():
                continue
            
            boundary = batch['boundary'][batch['valid']].to(device)
            hsv = batch['hsv'][batch['valid']].to(device)
            flower_center = batch['flower_center'][batch['valid']].to(device)
            gt_center = batch['gt_center'][batch['valid']].to(device)
            
            offset = model(boundary, hsv)
            pred_center = flower_center + offset
            
            loss = criterion(pred_center, gt_center)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
            
            batch_pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        batch_pbar.close()
        
        # 验证
        model.eval()
        val_loss = 0
        val_count = 0
        errors = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中", ncols=80, leave=False):
                if not batch['valid'].any():
                    continue
                
                boundary = batch['boundary'][batch['valid']].to(device)
                hsv = batch['hsv'][batch['valid']].to(device)
                flower_center = batch['flower_center'][batch['valid']].to(device)
                gt_center = batch['gt_center'][batch['valid']].to(device)
                img_wh = batch['img_wh'][batch['valid']].to(device)
                
                offset = model(boundary, hsv)
                pred_center = flower_center + offset
                
                loss = criterion(pred_center, gt_center)
                val_loss += loss.item()
                val_count += 1
                
                # 计算像素误差
                pixel_errors = torch.norm((pred_center - gt_center) * img_wh, dim=1)
                errors.extend(pixel_errors.cpu().tolist())
        
        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        mean_error = np.mean(errors) if errors else 0
        
        # 更新进度条
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'error': f"{mean_error:.1f}px"
        })
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
    
    epoch_pbar.close()
    
    print(f"\n训练完成！")
    print(f"模型保存: {save_dir}/best.pth")
    print(f"最佳验证损失: {best_loss:.6f}")
    
    # 5. 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pth")))
    model.eval()
    
    all_errors = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="评估中", ncols=80):
            if not batch['valid'].any():
                continue
            
            boundary = batch['boundary'][batch['valid']].to(device)
            hsv = batch['hsv'][batch['valid']].to(device)
            flower_center = batch['flower_center'][batch['valid']].to(device)
            gt_center = batch['gt_center'][batch['valid']].to(device)
            img_wh = batch['img_wh'][batch['valid']].to(device)
            
            offset = model(boundary, hsv)
            pred_center = flower_center + offset
            
            pixel_errors = torch.norm((pred_center - gt_center) * img_wh, dim=1)
            all_errors.extend(pixel_errors.cpu().tolist())
    
    all_errors = np.array(all_errors)
    print(f"  总样本数: {len(all_errors)}")
    print(f"  平均误差: {np.mean(all_errors):.2f} px")
    print(f"  中位数误差: {np.median(all_errors):.2f} px")
    print(f"  <10px: {np.sum(all_errors < 10)} ({np.sum(all_errors < 10)/len(all_errors)*100:.1f}%)")
    print(f"  <20px: {np.sum(all_errors < 20)} ({np.sum(all_errors < 20)/len(all_errors)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("全流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
