"""
改进版：YOLO分割 + 轮廓提取 + CNN+Transformer定位授粉点
========================================================
修复GT问题：partially_visible也作为有效标注
使用CNN+Transformer替代MLP
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
import math
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============ 配置 ============
RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
NUM_BOUNDARY_POINTS = 64


# ============ 网络 ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


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
            nn.Linear(3, 32), nn.ReLU(inplace=True), nn.Linear(32, 32),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model + 32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(64, 2), nn.Tanh()
        )
    
    def forward(self, boundary_points, hsv_features):
        B = boundary_points.shape[0]
        pts = boundary_points.view(B, 2, self.num_boundary_points)
        cnn_features = self.cnn_encoder(pts)
        features = cnn_features.permute(0, 2, 1)
        features = self.pos_encoding(features)
        features = self.transformer(features)
        features = self.global_pool(features.permute(0, 2, 1)).squeeze(-1)
        hsv_feat = self.hsv_encoder(hsv_features)
        combined = torch.cat([features, hsv_feat], dim=1)
        return self.predictor(combined)


# ============ 工具函数 ============
def extract_boundary_points(mask, num_points=64):
    """
    从掩膜提取轮廓边界点（弧长等间距采样）
    
    关键改进：不是按索引等间距，而是按弧长等间距
    这样保证每个点之间的实际距离相等
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 取最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    
    if len(contour_points) < 3:
        return None
    
    # 计算累积弧长
    diff = np.diff(contour_points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length == 0:
        return None
    
    # 等弧长采样
    target_lengths = np.linspace(0, total_length, num_points)
    sampled_points = np.zeros((num_points, 2))
    
    for i, target_len in enumerate(target_lengths):
        # 找到目标弧长所在的线段
        idx = np.searchsorted(cumulative_length, target_len)
        idx = min(idx, len(contour_points) - 1)
        
        if idx == 0:
            sampled_points[i] = contour_points[0]
        else:
            # 线性插值
            t = (target_len - cumulative_length[idx-1]) / (cumulative_length[idx] - cumulative_length[idx-1] + 1e-7)
            sampled_points[i] = contour_points[idx-1] * (1 - t) + contour_points[idx] * t
    
    # 归一化到[0,1]
    h, w = mask.shape
    normalized = sampled_points.astype(np.float32)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    
    return normalized.flatten()


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


# ============ 数据集 ============
class YOLOSegPollinationDataset(Dataset):
    def __init__(self, img_dir, label_dir, seg_model, num_boundary_points=64):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.num_boundary_points = num_boundary_points
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        # YOLO分割
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
        
        # 读取GT（修复：partially_visible也作为有效标注）
        json_name = img_file.replace('.jpg', '.json')
        label_path = os.path.join(self.label_dir, json_name)
        gt_center = np.array([0.5, 0.5])
        
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data['shapes']:
                if shape['shape_type'] == 'point' and shape['label'] in ['fully_visible', 'partially_visible']:
                    gt_center = np.array([shape['points'][0][0] / w, shape['points'][0][1] / h])
                    break
        
        # 花朵中心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flower_center = np.array([0.5, 0.5])
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                flower_center = np.array([M["m10"]/M["m00"]/w, M["m01"]/M["m00"]/h])
        
        return {
            'boundary': torch.tensor(boundary) if boundary is not None else torch.zeros(self.num_boundary_points * 2),
            'hsv': torch.tensor(hsv_feat),
            'flower_center': torch.tensor(flower_center),
            'gt_center': torch.tensor(gt_center),
            'valid': boundary is not None
        }


# ============ 训练 ============
def main():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 60)
    print("YOLO分割 + CNN+Transformer定位授粉点")
    print("=" * 60)
    
    # 加载YOLO分割模型
    print(f"加载分割模型: {SEG_MODEL_PATH}")
    seg_model = YOLO(SEG_MODEL_PATH)
    
    # 创建数据集
    print("创建数据集...")
    print("  GT修复: partially_visible也作为有效标注")
    
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
    
    # 创建网络
    model = ImprovedContourToPollinationNet(num_boundary_points=NUM_BOUNDARY_POINTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    save_dir = os.path.join(RESULTS_DIR, "11_cnn_transformer_pollination")
    os.makedirs(save_dir, exist_ok=True)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n网络参数量: {params:,}")
    print(f"设备: {device}")
    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    
    best_loss = float('inf')
    
    # 训练
    epoch_pbar = tqdm(range(100), desc="训练进度", ncols=100)
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        train_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=80, leave=False):
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
        
        # 验证
        model.eval()
        val_loss = 0
        val_count = 0
        errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                if not batch['valid'].any():
                    continue
                
                boundary = batch['boundary'][batch['valid']].to(device)
                hsv = batch['hsv'][batch['valid']].to(device)
                flower_center = batch['flower_center'][batch['valid']].to(device)
                gt_center = batch['gt_center'][batch['valid']].to(device)
                
                offset = model(boundary, hsv)
                pred_center = flower_center + offset
                
                loss = criterion(pred_center, gt_center)
                val_loss += loss.item()
                val_count += 1
                
                for i in range(pred_center.shape[0]):
                    err = torch.sqrt(((pred_center[i] - gt_center[i]) * 640) ** 2).sum().item()
                    errors.append(err)
        
        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        mean_error = np.mean(errors) if errors else 0
        
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'error': f"{mean_error:.1f}px"
        })
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
    
    epoch_pbar.close()
    
    # 评估
    print("\n" + "=" * 60)
    print("最终评估")
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
            
            offset = model(boundary, hsv)
            pred_center = flower_center + offset
            
            for i in range(pred_center.shape[0]):
                err = torch.sqrt(((pred_center[i] - gt_center[i]) * 640) ** 2).sum().item()
                all_errors.append(err)
    
    all_errors = np.array(all_errors)
    print(f"  总样本数: {len(all_errors)}")
    print(f"  平均误差: {np.mean(all_errors):.2f} px")
    print(f"  中位数误差: {np.median(all_errors):.2f} px")
    print(f"  <10px: {np.sum(all_errors < 10)} ({np.sum(all_errors < 10)/len(all_errors)*100:.1f}%)")
    print(f"  <20px: {np.sum(all_errors < 20)} ({np.sum(all_errors < 20)/len(all_errors)*100:.1f}%)")
    
    print(f"\n模型保存: {save_dir}/best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
