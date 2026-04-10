import os
import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset

class RGBDDataset(YOLODataset):
    def __init__(self, depth_dir, *args, **kwargs):
        self.depth_dir = depth_dir
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        rgb_img = label['img']  # shape: (H, W, 3)
        img_path = label['im_file']
        
        # 获取深度图路径
        depth_path = self.get_depth_path(img_path) 
        
        # --- 新增点：兼容读取 .npy 或图像格式 ---
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"致命错误: 找不到对应的深度图文件 -> {depth_path}")

        if depth_path.endswith('.npy'):
            depth_img = np.load(depth_path)
        else:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
        # 确保深度图维度正确 (H, W) -> (H, W, 1)
        if len(depth_img.shape) == 2:
            depth_img = np.expand_dims(depth_img, axis=-1) 
            
        # 确保深度图与RGB图分辨率绝对对齐
        if depth_img.shape[:2] != rgb_img.shape[:2]:
            depth_img = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if len(depth_img.shape) == 2:
                 depth_img = np.expand_dims(depth_img, axis=-1)

        # 在通道维度拼接，形成 C=4 的张量
        rgbd_img = np.concatenate([rgb_img, depth_img], axis=-1)
        label['img'] = rgbd_img

        return label

    def get_depth_path(self, rgb_path):
        """
        严谨的路径映射逻辑
        """
        # 1. 提取纯文件名（舍弃原有的 .jpg / .JPG / .png 等后缀）
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        
        # 2. 拼接新的后缀 (请根据你的实际情况修改这里！如果是npy就用 .npy)
        depth_filename = base_name + '.npy'  # <--- 请确认这里是 .npy 还是 .png 或 .tiff
        
        # 3. 检查是否需要区分 train / val 子目录
        # 如果你的 depth_maps 文件夹里面还有 train 和 val 文件夹，取消下面两行的注释：
        # if 'val' in rgb_path.replace('\\', '/').split('/'):
        #     return os.path.join(self.depth_dir, 'val', depth_filename)
        # else:
        #     return os.path.join(self.depth_dir, 'train', depth_filename)

        return os.path.join(self.depth_dir, depth_filename)