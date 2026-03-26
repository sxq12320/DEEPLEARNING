import json
import os


def coco_seg_to_yolo(json_path, output_dir):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 建立 image_id 到 宽高的映射字典
    images_info = {}
    for image in data['images']:
        images_info[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height']
        }

    # 2. 遍历 annotations 提取分割坐标
    count = 0
    for ann in data['annotations']:
        # 检查是否包含 segmentation 数据
        if 'segmentation' not in ann or not ann['segmentation']:
            continue

        # 如果 segmentation 是 RLE 格式（通常是字典，跳过）
        if isinstance(ann['segmentation'], dict):
            print(f"⚠️ 警告：跳过 RLE 格式的标注 (ID: {ann['id']})")
            continue

        image_id = ann['image_id']
        category_id = ann['category_id']

        # YOLO 类别索引从 0 开始
        yolo_class_id = category_id - 1

        # 获取当前图片的实际宽和高
        image_w = images_info[image_id]['width']
        image_h = images_info[image_id]['height']

        # 获取对应的文件名，把后缀换成 .txt
        file_name = images_info[image_id]['file_name']
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        # 写入文件（'a' 模式表示追加）
        with open(txt_path, 'a') as txt_file:
            # segmentation 通常是一个列表的列表，例如 [[x1, y1, x2, y2, ...]]
            for polygon in ann['segmentation']:
                # 过滤掉点数太少无法构成多边形的异常数据 (至少得有3个点，即6个坐标值)
                if len(polygon) < 6:
                    continue

                normalized_coords = []
                # 步长为 2 遍历坐标点，进行归一化 (x除以宽，y除以高)
                for i in range(0, len(polygon), 2):
                    x_norm = polygon[i] / image_w
                    y_norm = polygon[i + 1] / image_h
                    normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")

                # 将类别 ID 和归一化后的坐标组合成 YOLO 分割格式
                line = f"{yolo_class_id} " + " ".join(normalized_coords) + "\n"
                txt_file.write(line)
                count += 1

    print(f"✅ 转换完成！共处理了 {count} 个多边形轮廓。")
    print(f"📁 YOLO 分割格式的 txt 文件已保存在: {output_dir}")


# --- 运行代码 ---
# 把 'your_dataset.json' 换成你这个 JSON 文件的实际路径
# 把 'yolo_seg_labels_output' 换成你想要保存新 txt 标签的文件夹路径
json_file_path = r"E:\mastercode\data\jeruk_split\annotations\instances_val.json"
output_folder = r'E:\mastercode\data\jeruk_split\labels\val'

coco_seg_to_yolo(json_file_path, output_folder)