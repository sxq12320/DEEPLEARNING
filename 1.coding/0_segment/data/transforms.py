import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def image_transform(image_path, target_size=(640, 640)):
    """修改图片大小使其与目标大小一致（letterbox 方式）。"""
    image = Image.open(image_path)
    w, h = image.size
    max_len = max(w, h)
    black_mask = Image.new("RGB", (max_len, max_len), (0, 0, 0))
    black_mask.paste(image, (0, 0))
    return black_mask.resize(target_size)


def TXT2MASK(label_dir, image_name, target_size):
    """读取 YOLO 格式 TXT 标签并渲染为 mask。"""
    stem = Path(image_name).stem
    txt_path = os.path.join(label_dir, f"{stem}.txt")
    w, h = target_size[0], target_size[1]
    label = np.zeros((h, w, 1), dtype=np.uint8)

    if not os.path.exists(txt_path):
        return label

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            vals = [float(v) for v in line.split()]
            if len(vals) < 5:
                continue
            cls_id = int(vals[0])
            if len(vals) == 5:
                cx, cy, bw, bh = vals[1:5]
                if max(cx, cy, bw, bh) <= 1.0:
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                x1, y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
                x2, y2 = int(min(w - 1, cx + bw / 2)), int(min(h - 1, cy + bh / 2))
                cv2.rectangle(label, (x1, y1), (x2, y2), color=cls_id, thickness=-1)
            else:
                pts = np.array(vals[1:], dtype=np.float32).reshape(-1, 2)
                if pts.max() <= 1.0:
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(label, [pts], color=cls_id)
    except Exception as e:
        print(f"读取 TXT 标签失败: {txt_path}, 错误: {e}")

    return label


def JSON2MASK(label_dir, image_name, target_size):
    """读取 COCO 格式 JSON 标签并渲染为 mask。"""
    w, h = target_size[0], target_size[1]
    label = np.zeros((h, w, 1), dtype=np.uint8)

    json_path = None
    if os.path.isfile(label_dir) and label_dir.lower().endswith('.json'):
        json_path = label_dir
    else:
        json_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.json')]
        if json_files:
            json_path = os.path.join(label_dir, json_files[0])

    if json_path is None or not os.path.exists(json_path):
        return label

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        img_id_map = {img['file_name']: img['id'] for img in coco_data.get('images', [])}
        img_info_map = {img['id']: img for img in coco_data.get('images', [])}

        image_id = img_id_map.get(image_name)
        if image_id is None:
            return label

        img_w = img_info_map[image_id].get('width', w)
        img_h = img_info_map[image_id].get('height', h)

        for ann in coco_data.get('annotations', []):
            if ann.get('image_id') != image_id:
                continue
            cls_id = int(ann.get('category_id', 1))
            segmentation = ann.get('segmentation', [])

            if isinstance(segmentation, list) and segmentation:
                for poly in segmentation:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    max_len = max(img_w, img_h)
                    pts[:, 0] = pts[:, 0] / max_len * w
                    pts[:, 1] = pts[:, 1] / max_len * h
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(label, [pts], color=cls_id)
            elif isinstance(segmentation, dict) and 'counts' in segmentation:
                pass
            else:
                bbox = ann.get('bbox', None)
                if bbox and len(bbox) == 4:
                    x, y, bw, bh = bbox
                    max_len = max(img_w, img_h)
                    x1, y1 = int(max(0, x / max_len * w)), int(max(0, y / max_len * h))
                    x2, y2 = int(min(w - 1, (x + bw) / max_len * w)), int(min(h - 1, (y + bh) / max_len * h))
                    cv2.rectangle(label, (x1, y1), (x2, y2), color=cls_id, thickness=-1)
    except Exception as e:
        print(f"读取 JSON 标签失败: {json_path}, 错误: {e}")

    return label


def NPY2MASK(label_dir, image_name, target_size):
    """读取 NPY 格式标签并渲染为 mask。"""
    stem = Path(image_name).stem
    npy_path = os.path.join(label_dir, f"{stem}.npy")
    w, h = target_size[0], target_size[1]
    label = np.zeros((h, w, 1), dtype=np.uint8)

    if not os.path.exists(npy_path):
        return label

    try:
        data = np.load(npy_path)
        if data.ndim == 2:
            data = data[..., np.newaxis]
        if data.shape != (h, w, 1):
            data_2d = data[..., 0] if data.ndim == 3 else data
            resized = cv2.resize(data_2d, target_size, interpolation=cv2.INTER_NEAREST)
            label = resized[..., np.newaxis]
        else:
            label = data.astype(np.uint8)
    except Exception as e:
        print(f"读取 NPY 标签失败: {npy_path}, 错误: {e}")

    return label
