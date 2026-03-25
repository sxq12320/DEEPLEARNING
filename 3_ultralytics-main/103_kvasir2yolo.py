"""
Kvasir-SEG 完整转换脚本
支持：
  1. 目标检测（bbox → YOLO det格式）
  2. 实例分割（mask → YOLO seg格式 polygon）

数据集目录结构（你的现有结构）：
    E:/mastercode/data/Kvasir-SEG/
    ├── images/                  ← 原始图片
    ├── masks/                   ← 二值掩码图（白色=息肉区域）
    └── kavsir_bboxes.json       ← bbox标注

用法：
    1. 修改下方【配置区】的路径
    2. 选择 MODE = "det" 或 "seg"
    3. python kvasir2yolo.py

输出目录结构：
    Kvasir-SEG-yolo-seg/  或  Kvasir-SEG-yolo-det/
    ├── images/
    │   ├── yolo26n_origin/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── yolo26n_origin/
    │   ├── val/
    │   └── test/
    └── 203_kvasir_data.yaml
"""

import json
import shutil
import random
import cv2
from pathlib import Path


# ========== 配置区（修改为你的实际路径）==========
DATA_ROOT    = r"E:\mastercode\data\Kvasir-SEG"  # 数据集根目录
JSON_FILE    = "kavsir_bboxes.json"              # JSON文件名
IMAGE_SUBDIR = "images"                          # 图片子目录名
MASK_SUBDIR  = "masks"                           # mask子目录名

MODE         = "seg"   # "det" = 目标检测  |  "seg" = 实例分割

# 数据集划分比例（三者之和必须 = 1.0）
TRAIN_RATIO  = 0.80    # 训练集 70%
VAL_RATIO    = 0.1    # 验证集 15%
TEST_RATIO   = 0.1    # 测试集 15%

RANDOM_SEED  = 42      # 随机种子（保证可复现）
# ================================================


IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp"]


def find_file(directory: Path, stem: str):
    for suffix in IMAGE_SUFFIX:
        p = directory / (stem + suffix)
        if p.exists():
            return p
    return None


def mask_to_polygon(mask_path: Path, img_w: int, img_h: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    if mask.shape[0] != img_h or mask.shape[1] != img_w:
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = approx.reshape(-1, 2)
        coords = []
        for x, y in points:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append("0 " + " ".join(coords))

    return lines


def bbox_to_yolo(box: dict, img_w: int, img_h: int, cls_id: int):
    xmin = max(0, min(box["xmin"], img_w))
    xmax = max(0, min(box["xmax"], img_w))
    ymin = max(0, min(box["ymin"], img_h))
    ymax = max(0, min(box["ymax"], img_h))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_c = (xmin + xmax) / 2 / img_w
    y_c = (ymin + ymax) / 2 / img_h
    bw  = (xmax - xmin) / img_w
    bh  = (ymax - ymin) / img_h

    return f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"


def convert():
    data_root  = Path(DATA_ROOT)
    image_dir  = data_root / IMAGE_SUBDIR
    mask_dir   = data_root / MASK_SUBDIR
    json_path  = data_root / JSON_FILE
    output_dir = data_root.parent / f"Kvasir-SEG-yolo-{MODE}"

    print("=" * 55)
    print(f"  Kvasir-SEG → YOLO {'分割(seg)' if MODE=='seg' else '检测(det)'} 转换脚本")
    print("=" * 55)

    # 读取JSON
    print(f"\n[1/5] 读取标注: {json_path.name}")
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    total = len(annotations)
    print(f"      共 {total} 张图片")

    # 划分数据集 yolo26n_origin / val / test
    print(f"\n[2/5] 划分 yolo26n_origin/val/test  ({TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%})")
    keys = list(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(keys)

    n_val   = max(1, int(total * VAL_RATIO))
    n_test  = max(1, int(total * TEST_RATIO))
    n_train = total - n_val - n_test

    val_set   = set(keys[:n_val])
    test_set  = set(keys[n_val:n_val + n_test])
    train_set = set(keys[n_val + n_test:])

    print(f"      yolo26n_origin: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

    # 创建输出目录
    print(f"\n[3/5] 创建输出目录: {output_dir.name}")
    for split in ("yolo26n_origin", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 转换
    print(f"\n[4/5] 开始转换...")
    stats = {"yolo26n_origin": 0, "val": 0, "test": 0}
    skip, warn = 0, 0

    for img_id, info in annotations.items():
        if img_id in val_set:
            split = "val"
        elif img_id in test_set:
            split = "test"
        else:
            split = "yolo26n_origin"

        img_w = info["width"]
        img_h = info["height"]
        boxes = info.get("bbox", [])

        # 查找图片
        src_img = find_file(image_dir, img_id)
        if src_img is None:
            print(f"  [警告] 找不到图片: {img_id}")
            warn += 1
            continue

        # 生成label内容
        if MODE == "seg":
            mask_path = find_file(mask_dir, img_id)
            if mask_path is not None:
                label_lines = mask_to_polygon(mask_path, img_w, img_h)
            else:
                print(f"  [提示] 无mask，使用bbox代替: {img_id}")
                label_lines = [l for box in boxes
                               for l in [bbox_to_yolo(box, img_w, img_h, 0)] if l]
        else:
            label_lines = [l for box in boxes
                           for l in [bbox_to_yolo(box, img_w, img_h, 0)] if l]

        if not label_lines:
            skip += 1
            continue

        # 写 label 文件
        label_path = output_dir / "labels" / split / (img_id + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        # 复制图片
        dst_img = output_dir / "images" / split / (img_id + src_img.suffix)
        shutil.copy2(src_img, dst_img)

        stats[split] += 1

    print(f"      yolo26n_origin: {stats['yolo26n_origin']}  val: {stats['val']}  test: {stats['test']}")
    if skip: print(f"      跳过(无有效标注): {skip}")
    if warn: print(f"      警告(找不到图片): {warn}")

    # 生成 203_kvasir_data.yaml
    print(f"\n[5/5] 生成 203_kvasir_data.yaml")
    yaml_path = output_dir / "203_kvasir_data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# Kvasir-SEG YOLO数据集配置  mode={MODE}
path: {output_dir.resolve()}
yolo26n_origin: images/yolo26n_origin
val:   images/val
test:  images/test

nc: 1
names: ['polyp']
""")

    model = "yolov8s-seg.pt" if MODE == "seg" else "yolov8s.pt"
    print(f"\n{'='*55}")
    print(f"  ✅ 转换完成！输出: {output_dir}")
    print(f"\n  训练命令:")
    print(f"    yolo yolo26n_origin model={model} data={yaml_path} epochs=100 imgsz=640 batch=8")
    print(f"\n  测试命令:")
    print(f"    yolo val model=runs/segment/yolo26n_origin/weights/best.pt data={yaml_path} split=test")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("❌ 缺少依赖，请先运行: pip install opencv-python")
        exit(1)

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        f"❌ 比例之和必须=1.0，当前={TRAIN_RATIO+VAL_RATIO+TEST_RATIO:.2f}"

    convert()