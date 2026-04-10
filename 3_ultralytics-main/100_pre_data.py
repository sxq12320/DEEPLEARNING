import numpy as np
import cv2
import torch

def load_rgbd(img_path, depth_npy_path):
    # 1. 读取 RGB
    rgb = cv2.imread(img_path)               # (H, W, 3) uint8
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # 2. 读取深度图并归一化
    depth = np.load(depth_npy_path)          # (H, W) float32
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_norm = (depth_norm * 255).astype(np.uint8)  # 统一到 0-255

    # 3. 缩放到同一尺寸（YOLO 默认 640x640）
    H, W = rgb.shape[:2]
    depth_resized = cv2.resize(depth_norm, (W, H))

    # 4. 拼接为 4 通道 (H, W, 4)
    rgbd = np.concatenate([rgb, depth_resized[:, :, np.newaxis]], axis=-1)
    return rgbd  # shape: (H, W, 4)