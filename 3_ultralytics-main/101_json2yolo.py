import json
import os
import re
from PIL import Image

def clean_filename(via_key):
    """去除VIA键名中的数字后缀"""
    match = re.match(r'(.+\.png)\d+$', via_key, re.IGNORECASE)
    if match:
        return match.group(1)
    return via_key

def via_json_to_yolo_seg(json_path, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 只需要一个类别：苹果
    class_id = 0 
    print("设定总类别数 (nc): 1")
    print("类别映射: {0: 'apple'}")

    for via_key, info in data.items():
        img_filename = clean_filename(via_key)
        img_path = os.path.join(image_dir, img_filename)

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"跳过 {img_path}: {e}")
            continue

        yolo_lines = []
        for region in info.get('regions', {}).values():
            shape = region.get('shape_attributes', {})
            if shape.get('name') != 'polygon':
                continue

            xs = shape.get('all_points_x', [])
            ys = shape.get('all_points_y', [])
            if not xs or not ys:
                continue

            # 归一化坐标
            norm_points = []
            for x, y in zip(xs, ys):
                norm_points.append(f"{x/img_w:.6f}")
                norm_points.append(f"{y/img_h:.6f}")

            # 核心修改：不再读取 apple_ID 作为类别，直接统一定义为 class_id (0)
            yolo_lines.append(f"{class_id} " + " ".join(norm_points))

        txt_name = os.path.splitext(img_filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, 'w') as f_out:
            f_out.write("\n".join(yolo_lines))

        print(f"已生成: {txt_path}")

    print("\n请在 data.yaml 中设置:")
    print("nc: 1")
    print("names: [\"apple\"]")

# 使用示例
if __name__ == "__main__":
    via_json_to_yolo_seg(
        json_path=r"E:\mastercode\data\Apple_RGB_D_Amoal\gt_json\test\via_region_data_amodal.json",   # 你的 JSON 文件路径
        image_dir=r"E:\\mastercode\\data\\Apple_RGB_D_Amoal\\yolo\\test\\images",         # 原始图片所在目录
        output_dir=r"E:\\mastercode\\data\\Apple_RGB_D_Amoal\\yolo\\test\\labels"        # 输出目录
    )