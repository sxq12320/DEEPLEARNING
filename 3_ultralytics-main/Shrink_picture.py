import os
import cv2
import numpy as np
from pathlib import Path
import random

# ─────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────
INPUT_IMG_DIR   = "images/train"          # 原始图片目录
INPUT_LABEL_DIR = "labels/train"          # 原始标签目录
OUTPUT_IMG_DIR  = "small_target/images/train"
OUTPUT_LABEL_DIR= "small_target/labels/train"

CANVAS_SIZE  = 640          # 目标画布大小
SMALL_SIZE   = 200          # 缩小后的尺寸
LABEL_TYPE   = "segment"    # "detect" 或 "segment"

# 改进版：是否使用真实背景
USE_REAL_BG     = False
BG_IMG_DIR      = "backgrounds"   # 背景图片目录（USE_REAL_BG=True 时生效）

# 改进版：是否随机位置（False = 固定右上角）
RANDOM_POSITION = False


# ─────────────────────────────────────────
# 坐标变换核心函数
# ─────────────────────────────────────────
def transform_coords(coords, scale, x_offset, y_offset, canvas_size):
    """
    coords: list of (x, y) 归一化坐标
    返回变换后的归一化坐标
    """
    new_coords = []
    for x, y in coords:
        new_x = (x * scale + x_offset) / canvas_size
        new_y = (y * scale + y_offset) / canvas_size
        new_coords.append((new_x, new_y))
    return new_coords


def transform_detect_label(line, scale, x_offset, y_offset, canvas_size):
    """处理检测格式: class cx cy w h"""
    parts = line.strip().split()
    cls = parts[0]
    cx, cy, w, h = map(float, parts[1:5])

    new_cx = (cx * scale + x_offset) / canvas_size
    new_cy = (cy * scale + y_offset) / canvas_size
    new_w  = w * scale / canvas_size
    new_h  = h * scale / canvas_size

    return f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"


def transform_segment_label(line, scale, x_offset, y_offset, canvas_size):
    """处理分割格式: class x1 y1 x2 y2 ... xn yn"""
    parts = line.strip().split()
    cls = parts[0]
    coords_flat = list(map(float, parts[1:]))

    # 拆成 (x, y) 对
    points = [(coords_flat[i], coords_flat[i+1])
              for i in range(0, len(coords_flat), 2)]

    new_points = transform_coords(points, scale, x_offset, y_offset, canvas_size)

    coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in new_points)
    return f"{cls} {coords_str}"


# ─────────────────────────────────────────
# 计算粘贴位置
# ─────────────────────────────────────────
def get_paste_position(canvas_size, small_size, random_pos=False):
    """
    返回 (x_offset, y_offset)
    固定模式：右上角
    随机模式：随机位置（确保不超出边界）
    """
    if random_pos:
        max_offset = canvas_size - small_size
        x_offset = random.randint(0, max_offset)
        y_offset = random.randint(0, max_offset)
    else:
        # 右上角
        x_offset = canvas_size - small_size   # 440
        y_offset = 0
    return x_offset, y_offset


# ─────────────────────────────────────────
# 主处理函数
# ─────────────────────────────────────────
def process_single(img_path, label_path, out_img_path, out_label_path,
                   canvas_size=640, small_size=200,
                   use_real_bg=False, bg_img_dir=None,
                   random_pos=False):

    # 1. 读取原图并缩小
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 读取失败: {img_path}")
        return

    small_img = cv2.resize(img, (small_size, small_size))

    # 2. 创建画布
    if use_real_bg and bg_img_dir:
        bg_files = list(Path(bg_img_dir).glob("*.jpg")) + \
                   list(Path(bg_img_dir).glob("*.png"))
        if bg_files:
            bg = cv2.imread(str(random.choice(bg_files)))
            canvas = cv2.resize(bg, (canvas_size, canvas_size))
        else:
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # 3. 确定粘贴位置
    x_offset, y_offset = get_paste_position(canvas_size, small_size, random_pos)

    # 4. 粘贴小图到画布
    canvas[y_offset:y_offset+small_size,
           x_offset:x_offset+small_size] = small_img

    # 5. 处理标签
    new_lines = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if LABEL_TYPE == "detect":
                    new_line = transform_detect_label(
                        line, small_size, x_offset, y_offset, canvas_size)
                else:
                    new_line = transform_segment_label(
                        line, small_size, x_offset, y_offset, canvas_size)
                new_lines.append(new_line)

    # 6. 保存结果
    cv2.imwrite(str(out_img_path), canvas)
    with open(out_label_path, "w") as f:
        f.write("\n".join(new_lines))


# ─────────────────────────────────────────
# 批量处理入口
# ─────────────────────────────────────────
def process_dataset():
    img_dir   = Path(INPUT_IMG_DIR)
    label_dir = Path(INPUT_LABEL_DIR)
    out_img_dir   = Path(OUTPUT_IMG_DIR)
    out_label_dir = Path(OUTPUT_LABEL_DIR)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"共找到 {len(img_files)} 张图片，开始处理...")

    for img_path in img_files:
        label_path = label_dir / (img_path.stem + ".txt")
        out_img_path   = out_img_dir   / img_path.name
        out_label_path = out_label_dir / (img_path.stem + ".txt")

        process_single(
            img_path, label_path,
            out_img_path, out_label_path,
            canvas_size    = CANVAS_SIZE,
            small_size     = SMALL_SIZE,
            use_real_bg    = USE_REAL_BG,
            bg_img_dir     = BG_IMG_DIR,
            random_pos     = RANDOM_POSITION
        )

    print("✅ 全部处理完成！")


# ─────────────────────────────────────────
# 可视化验证（处理后随机抽一张看看对不对）
# ─────────────────────────────────────────
def visualize_result(out_img_dir, out_label_dir, canvas_size=640):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img_files = list(Path(out_img_dir).glob("*.jpg")) + \
                list(Path(out_img_dir).glob("*.png"))
    if not img_files:
        print("没有找到输出图片")
        return

    img_path   = random.choice(img_files)
    label_path = Path(out_label_dir) / (img_path.stem + ".txt")

    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img_rgb)

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                coords = list(map(float, parts[1:]))

                if LABEL_TYPE == "detect":
                    cx, cy, w, h = coords
                    x1 = (cx - w/2) * canvas_size
                    y1 = (cy - h/2) * canvas_size
                    rect = patches.Rectangle(
                        (x1, y1), w*canvas_size, h*canvas_size,
                        linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                else:
                    # 分割：画多边形
                    pts = np.array([(coords[i]*canvas_size, coords[i+1]*canvas_size)
                                    for i in range(0, len(coords), 2)])
                    poly = plt.Polygon(pts, closed=True,
                                       edgecolor='lime', facecolor='none', linewidth=2)
                    ax.add_patch(poly)

    ax.set_title(f"验证: {img_path.name}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("verify_result.png", dpi=150)
    plt.show()
    print("验证图已保存为 verify_result.png")


if __name__ == "__main__":
    process_dataset()
    visualize_result(OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR)
