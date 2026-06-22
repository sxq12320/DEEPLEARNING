"""
西瓜花关键点 - GT vs Prediction 对比可视化
==========================================
左图：Ground Truth（标注）
右图：Model Prediction（预测）
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# 配置
MODEL_PATH = r"E:\mastercode\ultralytics-main-new\results\08_shr_watermelon_6pt_nano\weights\best.pt"
VAL_IMG_DIR = r"E:\mastercode\data\shr_watermelon\pose\yolo_6pt\images\val"
VAL_LBL_DIR = r"E:\mastercode\data\shr_watermelon\pose\yolo_6pt\labels\val"
OUTPUT_DIR = r"E:\mastercode\ultralytics-main-new\results\vis_compare"

# 类别
CLASS_NAMES = ["blooming_male", "unknown", "closed_male", "blooming_female", "closed_female"]
KPT_NAMES = ["stigma", "petal1", "petal2", "petal3", "petal4", "petal5"]

# 颜色
KPT_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
]


def draw_gt(img, label_path, img_w, img_h):
    """绘制Ground Truth"""
    if not os.path.exists(label_path):
        return img
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            
            cls_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            cv2.putText(img, f"GT:{label}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            # 画关键点
            kpt_data = parts[5:]
            for j in range(6):
                idx = j * 3
                if idx + 2 >= len(kpt_data):
                    break
                kx = float(kpt_data[idx]) * img_w
                ky = float(kpt_data[idx + 1]) * img_h
                vis = int(kpt_data[idx + 2])
                
                if vis > 0:
                    color = KPT_COLORS[j]
                    cv2.circle(img, (int(kx), int(ky)), 3, color, -1)
                    cv2.circle(img, (int(kx), int(ky)), 3, (255, 255, 255), 1)
    
    return img


def draw_pred(img, results):
    """绘制Prediction"""
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        
        if boxes is None or len(boxes) == 0:
            continue
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            cls = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img, f"Pred:{label}:{conf:.2f}", (x1, y2 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            if keypoints is not None and len(keypoints) > 0:
                kpts = keypoints.xy[i].cpu().numpy()
                kpts_conf = keypoints.conf[i].cpu().numpy()
                
                for j in range(len(kpts)):
                    kx, ky = int(kpts[j, 0]), int(kpts[j, 1])
                    kc = kpts_conf[j] if j < len(kpts_conf) else 0
                    
                    if kc > 0.2:
                        color = KPT_COLORS[j]
                        cv2.circle(img, (kx, ky), 3, color, -1)
                        cv2.circle(img, (kx, ky), 3, (255, 255, 255), 1)
    
    return img


def main():
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    img_files = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.jpg')]
    print(f"处理 {len(img_files)} 张图片...")
    
    for img_file in img_files[:10]:
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        label_path = os.path.join(VAL_LBL_DIR, img_file.replace('.jpg', '.txt'))
        
        # 读取原图
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # GT图
        gt_img = img.copy()
        gt_img = draw_gt(gt_img, label_path, w, h)
        
        # Pred图
        pred_img = img.copy()
        results = model.predict(img_path, conf=0.25, verbose=False)
        pred_img = draw_pred(pred_img, results)
        
        # 拼接
        combined = np.hstack([gt_img, pred_img])
        
        # 添加标题
        title_h = 30
        title_img = np.zeros((title_h, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_img, "Ground Truth", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(title_img, "Prediction", (w + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        final = np.vstack([title_img, combined])
        
        # 保存
        output_path = os.path.join(OUTPUT_DIR, f"compare_{img_file}")
        cv2.imwrite(output_path, final)
        print(f"  {img_file}")
    
    print(f"\n完成！对比图保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
