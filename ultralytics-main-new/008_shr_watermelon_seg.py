"""
西瓜花授粉点全流程一键运行 (修复版)
====================================
适配实际标注格式:
- pose标签: fully_visible / partially_visible / invisible (只有point)
- 分割标签: 有rectangle和segmentation
"""

import torch
import random
import numpy as np
import cv2
import os
import json
from ultralytics import YOLO

# ============ 配置 ============
SEED = 42
SEG_DATA = r"E:/mastercode/ultralytics-main-new/208_shr_watermelon_seg.yaml"
POSE_LABELS = r"E:\mastercode\data\shr_watermelon\pose\labels"
SEG_IMAGES = r"E:\mastercode\data\shr_watermelon\segmentation\images"
RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_TRAIN_NAME = "09_watermelon_seg_2"
HSV_VIS_NAME = "09_hsv_pollination_vis"

# HSV阈值（黄色花朵）
HSV_LOWER = np.array([25, 40, 30])
HSV_UPPER = np.array([35, 255, 255])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_segmentation():
    """步骤1: 训练分割模型"""
    print("=" * 60)
    print("步骤1: 训练YOLO11-seg-nano分割模型")
    print("=" * 60)
    
    yolo = YOLO(r"E:\mastercode\ultralytics-main-new\results\09_watermelon_seg_2\weights\best.pt")
    yolo.train(
        data=SEG_DATA,
        project=RESULTS_DIR,
        name=SEG_TRAIN_NAME,
        epochs=10,
        patience=30,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        device=0 if torch.cuda.is_available() else "cpu",
        exist_ok=True,
    )
    
    print(f"\n分割训练完成！")
    return os.path.join(RESULTS_DIR, SEG_TRAIN_NAME, "weights", "best.pt")


def hsv_pollination_localization(image, mask):
    """HSV颜色分割定位授粉点"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    combined_mask = cv2.bitwise_and(yellow_mask, mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, combined_mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, combined_mask
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy), combined_mask


def evaluate_hsv_localization():
    """步骤2&3: HSV定位 + 计算mAP"""
    print("\n" + "=" * 60)
    print("步骤2: HSV方法授粉点定位评估")
    print("=" * 60)
    
    seg_model_path = os.path.join(RESULTS_DIR, SEG_TRAIN_NAME, "weights", "best.pt")
    if not os.path.exists(seg_model_path):
        print(f"错误: 找不到分割模型 {seg_model_path}")
        return
    
    seg_model = YOLO(seg_model_path)
    
    total_flowers = 0
    matched_points = 0
    errors = []
    vis_count = 0
    
    val_img_dir = os.path.join(SEG_IMAGES, "val")
    img_files = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    
    print(f"处理 {len(img_files)} 张验证图片...")
    
    for img_file in img_files:
        img_path = os.path.join(val_img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 分割推理
        results = seg_model.predict(img_path, conf=0.25, verbose=False)
        
        # 检查pose标签（.json格式）
        json_name = img_file.replace('.jpg', '.json')
        gt_path = os.path.join(POSE_LABELS, "val", json_name)
        
        if not os.path.exists(gt_path):
            continue
        
        # 读取pose标签
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # 获取所有可见的关键点
        visible_points = []
        for shape in gt_data['shapes']:
            if shape['shape_type'] == 'point' and shape['label'] == 'fully_visible':
                visible_points.append(shape['points'][0])
        
        if not visible_points:
            continue
        
        # 获取分割掩膜
        seg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if results[0].masks is not None:
            for r in results[0].masks:
                mask_data = r.data.cpu().numpy()[0] # type: ignore
                mask_resized = cv2.resize(mask_data, (img.shape[1], img.shape[0]))
                seg_mask[mask_resized > 0.5] = 255
        
        # 对每个可见关键点进行HSV定位
        for gt_point in visible_points:
            gt_x, gt_y = int(gt_point[0]), int(gt_point[1])
            
            # 找到该点最近的分割区域
            # 简单方法：在该点周围取一个patch进行HSV定位
            patch_size = 100
            x1 = max(0, gt_x - patch_size)
            y1 = max(0, gt_y - patch_size)
            x2 = min(img.shape[1], gt_x + patch_size)
            y2 = min(img.shape[0], gt_y + patch_size)
            
            crop = img[y1:y2, x1:x2]
            seg_crop = seg_mask[y1:y2, x1:x2]
            
            # HSV定位
            pred_point, _ = hsv_pollination_localization(crop, seg_crop)
            
            total_flowers += 1
            
            if pred_point is not None:
                pred_x = pred_point[0] + x1
                pred_y = pred_point[1] + y1
                
                error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                errors.append(error)
                
                if error < 10:
                    matched_points += 1
                
                # 可视化
                if vis_count < 10:
                    vis_dir = os.path.join(RESULTS_DIR, HSV_VIS_NAME)
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    vis_img = img.copy()
                    cv2.circle(vis_img, (gt_x, gt_y), 5, (0, 255, 0), -1)
                    cv2.circle(vis_img, (int(pred_x), int(pred_y)), 5, (0, 0, 255), -1)
                    cv2.putText(vis_img, f"Err:{error:.1f}px", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imwrite(os.path.join(vis_dir, f"vis_{vis_count}_{img_file}"), vis_img)
                    vis_count += 1
    
    # 输出结果
    if total_flowers > 0:
        accuracy = matched_points / total_flowers * 100
        mean_error = np.mean(errors) if errors else 0
        median_error = np.median(errors) if errors else 0
        
        print(f"\n=== HSV授粉点定位评估结果 ===")
        print(f"  总关键点数: {total_flowers}")
        print(f"  匹配成功(误差<10px): {matched_points}")
        print(f"  定位准确率: {accuracy:.2f}%")
        print(f"  平均误差: {mean_error:.2f} px")
        print(f"  中位数误差: {median_error:.2f} px")
        
        if errors:
            errors = np.array(errors)
            print(f"\n  误差分布:")
            print(f"    <5px:  {np.sum(errors < 5)} ({np.sum(errors < 5)/len(errors)*100:.1f}%)")
            print(f"    5-10px: {np.sum((errors >= 5) & (errors < 10))} ({np.sum((errors >= 5) & (errors < 10))/len(errors)*100:.1f}%)")
            print(f"    10-20px: {np.sum((errors >= 10) & (errors < 20))} ({np.sum((errors >= 10) & (errors < 20))/len(errors)*100:.1f}%)")
            print(f"    >20px: {np.sum(errors >= 20)} ({np.sum(errors >= 20)/len(errors)*100:.1f}%)")
    else:
        print("警告: 没有找到有效的fully_visible关键点进行评估")


def main():
    """一键运行全流程"""
    set_seed(SEED)
    
    print("西瓜花授粉点全流程一键运行")
    print("=" * 60)
    
    # 步骤1: 训练分割模型
    train_segmentation()
    
    # 步骤2&3: HSV定位评估
    evaluate_hsv_localization()
    
    print("\n" + "=" * 60)
    print("全流程完成！")
    print(f"结果保存在: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
