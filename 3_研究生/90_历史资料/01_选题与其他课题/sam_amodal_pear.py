"""
SAM + 形状先验 Amodal 分割 for 梨采摘点定位
=====================================================
功能：
1. 用 SAM 分割梨的可见区域
2. 用椭圆形状先验推断完整形状（amodal 分割）
3. 输出 amodal 分割掩码 + 可视化结果

使用方法：
    python sam_amodal_pear.py --input_dir ./pear_images --output_dir ./amodal_masks

日期：2026-06-28
"""

import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from scipy import ndimage
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. SAM 可见区域分割
# ===========================

class SAMSegmentor:
    def __init__(self, model_path, device='cuda'):
        """
        初始化 SAM 模型
        
        Args:
            model_path: SAM 预训练权重路径
            device: 'cuda' 或 'cpu'
        """
        self.device = device
        
        # 根据模型文件名选择模型类型
        if 'vit_h' in model_path:
            model_type = 'vit_h'
        elif 'vit_l' in model_path:
            model_type = 'vit_l'
        else:
            model_type = 'vit_b'
        
        # 加载 SAM 模型
        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        self.sam.to(device=self.device)
        
        # 创建自动掩码生成器
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        
        print(f"✅ SAM ({model_type}) 加载成功，运行在 {device}")
    
    def segment_image(self, image_path):
        """
        对单张图像进行分割
        
        Args:
            image_path: 图像路径
            
        Returns:
            masks: 分割掩码列表（每个掩码是一个二值 numpy 数组）
            bboxes: 边界框列表 [x1, y1, x2, y2]
        """
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 生成分割掩码
        masks = self.mask_generator.generate(image)
        
        # 过滤出可能是梨的掩码（基于面积和形状）
        pear_masks = []
        bboxes = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            
            # 过滤条件：面积在合理范围内（可根据实际调整）
            if 1000 < area < 50000:
                pear_masks.append(mask)
                
                # 计算边界框
                y_indices, x_indices = np.where(mask)
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()
                bboxes.append([x1, y1, x2, y2])
        
        return pear_masks, bboxes, image


# ===========================
# 2. 形状先验拟合（椭圆）
# ===========================

class ShapePriorFitter:
    def __init__(self):
        """
        形状先验拟合器（使用椭圆近似梨的形状）
        """
        pass
    
    def fit_ellipse(self, visible_mask):
        """
        根据可见掩码拟合椭圆
        
        Args:
            visible_mask: 可见区域二值掩码
            
        Returns:
            ellipse_params: 椭圆参数 (center_x, center_y, major_axis, minor_axis, angle)
            full_mask: 完整椭圆掩码（amodal）
        """
        # 找到可见区域的轮廓
        contours = measure.find_contours(visible_mask, 0.5)
        
        if len(contours) == 0:
            return None, visible_mask
        
        # 使用最大轮廓
        contour = max(contours, key=len)
        contour_points = contour[:, [1, 0]]  # 转换为 (x, y) 格式
        
        # 拟合椭圆
        if len(contour_points) >= 5:
            ellipse = cv2.fitEllipse(contour_points.astype(np.float32))
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            
            # 确保 major_axis >= minor_axis
            if major_axis < minor_axis:
                major_axis, minor_axis = minor_axis, major_axis
                angle = angle + 90
            
            ellipse_params = (center_x, center_y, major_axis, minor_axis, angle)
            
            # 生成完整椭圆掩码（amodal）
            full_mask = np.zeros_like(visible_mask, dtype=np.uint8)
            cv2.ellipse(
                full_mask,
                (int(center_x), int(center_y)),
                (int(major_axis / 2), int(minor_axis / 2)),
                angle,
                0, 360,
                1, -1
            )
            
            return ellipse_params, full_mask
        else:
            # 点数不足，无法拟合椭圆
            return None, visible_mask
    
    def infer_amodal_mask(self, visible_mask, ellipse_params):
        """
        根据椭圆参数推断 amodal 掩码
        
        Args:
            visible_mask: 可见区域二值掩码
            ellipse_params: 椭圆参数
            
        Returns:
            amodal_mask: amodal 二值掩码
        """
        if ellipse_params is None:
            return visible_mask
        
        center_x, center_y, major_axis, minor_axis, angle = ellipse_params
        
        # 生成完整椭圆掩码
        amodal_mask = np.zeros_like(visible_mask, dtype=np.uint8)
        cv2.ellipse(
            amodal_mask,
            (int(center_x), int(center_y)),
            (int(major_axis / 2), int(minor_axis / 2)),
            angle,
            0, 360,
            1, -1
        )
        
        return amodal_mask


# ===========================
# 3. 采摘点定位（梨柄位置）
# ===========================

class HarvestPointLocator:
    def __init__(self):
        """
        采摘点定位器（基于 amodal 分割掩码）
        """
        pass
    
    def find_harvest_point(self, amodal_mask, visible_mask):
        """
        找到采摘点（梨柄位置）
        
        策略：
        1. 找到 amodal 掩码的"顶部"区域（梨柄通常在果实顶部）
        2. 在顶部区域寻找轮廓的"凹点"或"极值点"
        
        Args:
            amodal_mask: amodal 二值掩码
            visible_mask: 可见区域二值掩码（用于参考）
            
        Returns:
            harvest_point: 采摘点坐标 (x, y)
        """
        # 找到 amodal 掩码的顶部
        y_indices, x_indices = np.where(amodal_mask)
        
        if len(y_indices) == 0:
            return None
        
        # 顶部的点（y 最小）
        top_y = y_indices.min()
        top_x_candidates = x_indices[y_indices == top_y]
        top_x = int(np.mean(top_x_candidates))
        
        # 梨柄通常在果实顶部的中心附近
        # 这里简化为顶部中心点
        harvest_point = (top_x, top_y)
        
        return harvest_point


# ===========================
# 4. 主流程
# ===========================

def process_images(input_dir, output_dir, model_path, device='cuda'):
    """
    批量处理图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        model_path: SAM 模型路径
        device: 'cuda' 或 'cpu'
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'amodal_masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 初始化 SAM
    sam_segmentor = SAMSegmentor(model_path, device)
    
    # 初始化形状先验拟合器
    shape_fitter = ShapePriorFitter()
    
    # 初始化采摘点定位器
    harvest_locator = HarvestPointLocator()
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"\n📂 找到 {len(image_files)} 张图像")
    print("=" * 50)
    
    for idx, image_file in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] 处理: {image_file}")
        
        image_path = os.path.join(input_dir, image_file)
        
        # 1. SAM 分割可见区域
        visible_masks, bboxes, image = sam_segmentor.segment_image(image_path)
        
        print(f"  ✅ 找到 {len(visible_masks)} 个梨实例")
        
        # 2. 对每个实例进行 amodal 分割
        amodal_masks = []
        harvest_points = []
        
        for i, visible_mask in enumerate(visible_masks):
            # 拟合椭圆形状先验
            ellipse_params, _ = shape_fitter.fit_ellipse(visible_mask)
            
            # 推断 amodal 掩码
            amodal_mask = shape_fitter.infer_amodal_mask(visible_mask, ellipse_params)
            amodal_masks.append(amodal_mask)
            
            # 定位采摘点
            harvest_point = harvest_locator.find_harvest_point(amodal_mask, visible_mask)
            harvest_points.append(harvest_point)
            
            print(f"    实例 {i+1}: 采摘点 = {harvest_point}")
        
        # 3. 保存结果
        # 保存 amodal 掩码
        for i, amodal_mask in enumerate(amodal_masks):
            mask_path = os.path.join(output_dir, 'amodal_masks', 
                                     f'{os.path.splitext(image_file)[0]}_instance_{i}.png')
            cv2.imwrite(mask_path, amodal_mask * 255)
        
        # 4. 可视化
        visualize_results(image, visible_masks, amodal_masks, harvest_points, 
                         os.path.join(output_dir, 'visualizations', image_file))
        
        print(f"  ✅ 结果已保存到: {output_dir}")
    
    print("\n" + "=" * 50)
    print("🎉 批量处理完成！")


def visualize_results(image, visible_masks, amodal_masks, harvest_points, save_path):
    """
    可视化结果
    
    Args:
        image: 原始图像
        visible_masks: 可见区域掩码列表
        amodal_masks: amodal 掩码列表
        harvest_points: 采摘点列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 原始图像
    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 2. 可见区域分割
    visible_overlay = image.copy()
    for mask in visible_masks:
        # 生成随机颜色
        color = np.random.randint(0, 255, 3)
        visible_overlay[mask] = 0.5 * visible_overlay[mask] + 0.5 * color
    
    axes[1].imshow(visible_overlay.astype(np.uint8))
    axes[1].set_title('可见区域分割 (SAM)')
    axes[1].axis('off')
    
    # 3. Amodal 分割 + 采摘点
    amodal_overlay = image.copy()
    for i, (mask, point) in enumerate(zip(amodal_masks, harvest_points)):
        # 生成随机颜色
        color = np.random.randint(0, 255, 3)
        amodal_overlay[mask] = 0.5 * amodal_overlay[mask] + 0.5 * color
        
        # 绘制采摘点
        if point is not None:
            cv2.circle(amodal_overlay, point, 10, (255, 0, 0), -1)
            cv2.putText(amodal_overlay, f'P{i}', 
                       (point[0] + 15, point[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    axes[2].imshow(amodal_overlay.astype(np.uint8))
    axes[2].set_title('Amodal 分割 + 采摘点')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ===========================
# 5. 命令行入口
# ===========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM + 形状先验 Amodal 分割 for 梨采摘点定位')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入图像目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--model_path', type=str, 
                       default='./models/sam_vit_h_4b8939.pth',
                       help='SAM 预训练权重路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='运行设备')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: SAM 模型文件不存在: {args.model_path}")
        print("请先下载 SAM 预训练权重:")
        print("  wget -P ./models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        exit(1)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        exit(1)
    
    # 开始处理
    print("=" * 60)
    print("🍐 SAM + 形状先验 Amodal 分割 for 梨采摘点定位")
    print("=" * 60)
    
    process_images(args.input_dir, args.output_dir, args.model_path, args.device)
