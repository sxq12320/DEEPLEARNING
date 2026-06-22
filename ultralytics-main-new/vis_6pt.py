"""
西瓜花关键点推理可视化
======================
在单张图片上绘制预测的框+关键点
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# 配置
MODEL_PATH = r"E:\mastercode\ultralytics-main-new\results\08_shr_watermelon_6pt_nano\weights\best.pt"
VAL_IMG_DIR = r"E:\mastercode\data\shr_watermelon\pose\yolo_6pt\images\val"
OUTPUT_DIR = r"E:\mastercode\ultralytics-main-new\results\vis_6pt"

# 关键点名称
KPT_NAMES = ["stigma", "petal1", "petal2", "petal3", "petal4", "petal5"]

# 关键点颜色 (BGR)
KPT_COLORS = [
    (0, 0, 255),    # 红 - stigma
    (0, 255, 0),    # 绿 - petal1
    (255, 0, 0),    # 蓝 - petal2
    (0, 255, 255),  # 黄 - petal3
    (255, 0, 255),  # 紫 - petal4
    (255, 255, 0),  # 青 - petal5
]


def visualize_single_image(img_path, model, output_dir):
    """在单张图片上可视化预测结果"""
    # 推理
    results = model.predict(img_path, conf=0.25, verbose=False)
    
    # 读取原图
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        
        if boxes is None or len(boxes) == 0:
            continue
        
        for i in range(len(boxes)):
            # 绘制框
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            cls = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            
            class_names = ["blooming_male", "unknown", "closed_male", "blooming_female", "closed_female"]
            label = f"{class_names[cls]} {conf:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # type: ignore
            
            # 绘制关键点
            if keypoints is not None and len(keypoints) > 0:
                kpts = keypoints.xy[i].cpu().numpy()  # (num_kpts, 2)
                kpts_conf = keypoints.conf[i].cpu().numpy()  # (num_kpts,)
                
                for j in range(len(kpts)):
                    kx, ky = int(kpts[j, 0]), int(kpts[j, 1])
                    kc = kpts_conf[j] if j < len(kpts_conf) else 0
                    
                    if kc > 0.3:  # 只绘制置信度>0.3的点
                        color = KPT_COLORS[j % len(KPT_COLORS)]
                        
                        # 画小圆点
                        cv2.circle(img, (kx, ky), 3, color, -1)
                        cv2.circle(img, (kx, ky), 3, (255, 255, 255), 1) # type: ignore
                        
                        # 小标签
                        kpt_label = f"{KPT_NAMES[j]}:{kc:.2f}"
                        cv2.putText(img, kpt_label, (kx + 6, ky - 3), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1) # type: ignore
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, f"pred_{filename}")
    cv2.imwrite(output_path, img)
    
    return output_path


def main():
    # 加载模型
    model = YOLO(MODEL_PATH)
    
    # 获取验证集图片
    img_files = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.jpg')]
    
    print(f"找到 {len(img_files)} 张验证图片")
    print(f"开始推理可视化...")
    
    # 处理前10张
    for i, img_file in enumerate(img_files[:10]):
        img_path = os.path.join(VAL_IMG_DIR, img_file)
        output_path = visualize_single_image(img_path, model, OUTPUT_DIR)
        print(f"  [{i+1}/10] {img_file} -> {output_path}")
    
    print(f"\n完成！结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
