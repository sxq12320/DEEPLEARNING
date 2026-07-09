"""
YOLO + GrabCut 半自动标注脚本（优化版）
用法: python yolo_grabcut_annotate.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# ============ 配置 ============
YOLO_MODEL = r"E:\mastercode\FWQ\ultralytics-main\0_results\JEURK_SEG\000_yolo11nano_base-2\weights\best.pt"
IMAGE_DIR = Path(r"E:\mastercode\data\jeruk_split\yolo\train\images")
OUTPUT_DIR = Path(r"E:\mastercode\data\jeruk_split\yolo\labels_yc")
VIS_DIR = Path(r"E:\mastercode\data\jeruk_split\yolo\vis_yc")

YOLO_CONF = 0.25
MIN_AREA = 200
MAX_POLY_POINTS = 50
SIMPLIFY_EPS = 0.005
GRABCUT_ITER = 3
# ==============================


def process_one(args):
    img_path, yolo = args
    image = cv2.imread(str(img_path))
    if image is None:
        return img_path, [], None
    h, w = image.shape[:2]

    results = yolo(image, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return img_path, [], image

    xyxy = boxes.xyxy.cpu().numpy()
    yolo_lines = []
    vis = image.copy()

    for box in xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1c, y1c = max(0, x1-8), max(0, y1-8)
        x2c, y2c = min(w, x2+8), min(h, y2+8)
        rect = (x1c, y1c, x2c-x1c, y2c-y1c)
        if rect[2] <= 0 or rect[3] <= 0:
            continue

        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, rect, bgd, fgd, GRABCUT_ITER, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except Exception:
            fg = np.zeros((h, w), np.uint8)
            fg[y1:y2, x1:x2] = 1

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, SIMPLIFY_EPS * peri, True)
            if len(approx) > MAX_POLY_POINTS:
                step = max(1, len(approx) // MAX_POLY_POINTS)
                approx = approx[::step]
            coords = approx.reshape(-1, 2).astype(float) / [w, h]
            coords = np.clip(coords, 0, 1)
            line = "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
            yolo_lines.append(line)
            pts = (approx.reshape(-1, 2)).astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    return img_path, yolo_lines, vis


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("加载YOLO模型...")
    yolo = YOLO(YOLO_MODEL)

    images = sorted(IMAGE_DIR.glob("*.jpg"))
    print(f"共 {len(images)} 张图片\n")

    total = 0
    for idx, img_path in enumerate(images):
        _, lines, vis = process_one((img_path, yolo))
        total += len(lines)

        label_path = OUTPUT_DIR / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines) if lines else "")

        if vis is not None:
            cv2.imwrite(str(VIS_DIR / img_path.name), vis)

        print(f"[{idx+1}/{len(images)}] {img_path.name} → {len(lines)} 个标注")

    print(f"\n完成! 共 {total} 个实例")
    print(f"标签: {OUTPUT_DIR}")
    print(f"可视化: {VIS_DIR}")


if __name__ == "__main__":
    main()
