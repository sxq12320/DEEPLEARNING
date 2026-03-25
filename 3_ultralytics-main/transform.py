import os
import json
import glob
import shutil
import random

def create_yolo_dataset(input_dir, output_dir, class_mapping, val_ratio=0.1, test_ratio=0.1):
    """
    将 Labelme 的 JPG+JSON 混合目录转换为标准的 YOLO 分割数据集结构 (包含 Train/Val/Test)
    """
    # 1. 创建 YOLO 所需的目录结构，新增了 'test' 文件夹
    for split in ['yolo26n_origin', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 2. 获取所有 json 文件并打乱顺序，用于随机划分
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    if not json_files:
        print("未找到 JSON 文件，请检查 input_dir 路径！")
        return
        
    random.shuffle(json_files)
    total_files = len(json_files)
    
    # 计算验证集和测试集的数量
    val_count = int(total_files * val_ratio)
    test_count = int(total_files * test_ratio)
    
    # 切片划分文件列表
    val_files = json_files[:val_count]
    test_files = json_files[val_count : val_count + test_count]
    train_files = json_files[val_count + test_count:]

    def process_files(files, split_name):
        processed_count = 0
        for json_path in files:
            # 读取 JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_width = data['imageWidth']
            img_height = data['imageHeight']
            
            # 对应的图片路径 (假设图片和 json 同名，且后缀为 .jpg)
            img_name = os.path.basename(json_path).replace('.json', '.jpg')
            img_path = os.path.join(input_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"警告：找不到对应的图片 {img_path}，已跳过。")
                continue

            # 准备输出的 TXT 内容
            txt_content = []
            for shape in data['shapes']:
                label = shape['label']
                if shape['shape_type'] != 'polygon' or label not in class_mapping:
                    continue

                class_id = class_mapping[label]
                normalized_points = []
                for point in shape['points']:
                    x = max(0.0, min(1.0, point[0] / img_width))
                    y = max(0.0, min(1.0, point[1] / img_height))
                    normalized_points.extend([f"{x:.6f}", f"{y:.6f}"])
                
                txt_content.append(f"{class_id} " + " ".join(normalized_points) + "\n")

            # 如果这个 JSON 里有有效的标注，才进行复制和生成
            if txt_content:
                # 复制图片到对应的 images 文件夹
                shutil.copy(img_path, os.path.join(output_dir, 'images', split_name, img_name))
                
                # 写入 TXT 到对应的 labels 文件夹
                txt_name = img_name.replace('.jpg', '.txt')
                with open(os.path.join(output_dir, 'labels', split_name, txt_name), 'w', encoding='utf-8') as txt_file:
                    txt_file.writelines(txt_content)
                processed_count += 1
                
        return processed_count

    # 3. 开始处理三个数据集
    print(f"检测到总标注文件数: {total_files}")
    print("开始处理训练集 (Train)...")
    train_count = process_files(train_files, 'yolo26n_origin')
    
    print("开始处理验证集 (Val)...")
    val_count = process_files(val_files, 'val')
    
    print("开始处理测试集 (Test)...")
    test_count = process_files(test_files, 'test')
    
    print("\n================ 转换完成 ================")
    print(f"成功生成 训练集: {train_count} 张, 验证集: {val_count} 张, 测试集: {test_count} 张。")
    print(f"数据集已保存在: {output_dir}")
    print("==========================================")


if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 1. 你图里那个包含 jpg 和 json 的文件夹路径
    INPUT_DIR = r"E:\mastercode\data\caomei\annotions" 
    
    # 2. 你想把生成好的 YOLO 数据集放在哪里
    OUTPUT_DIR = r"E:\mastercode\data\caomei\final"
    
    # 3. 类别映射表 (已配置好草莓的7种状态/病害)
    MY_CLASSES = {
        "Angular Leafspot": 0,  
        "Anthracnose Fruit Rot": 1,
        "Blossom Blight": 2,
        "Gray Mold": 3,
        "Leaf Spot": 4,
        "Powdery Mildew Fruit": 5,
        "Powdery Mildew Leaf": 6
    }
    
    # 4. 执行脚本 (按 8:1:1 比例划分)
    create_yolo_dataset(INPUT_DIR, OUTPUT_DIR, MY_CLASSES, val_ratio=0.1, test_ratio=0.1)