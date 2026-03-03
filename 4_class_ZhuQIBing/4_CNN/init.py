import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

def calculate_stats_with_sampling(train_folder, sample_ratio=0.1, img_size=224, seed=42):
    """
    随机抽样部分图片计算统计信息
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 处理路径
    train_folder = train_folder.replace('\\', '/')
    
    print(f"📁 正在扫描文件夹: {train_folder}")
    print(f"📊 抽样比例: {sample_ratio*100:.1f}%")
    
    # 收集所有图片路径
    all_image_paths = []
    for root, dirs, files in os.walk(train_folder):
        root = root.replace('\\', '/')
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(root, file).replace('\\', '/')
                all_image_paths.append(img_path)
    
    total_images = len(all_image_paths)
    print(f"📷 总共找到 {total_images} 张图片")
    
    if total_images == 0:
        print("❌ 错误: 没有找到任何图片！")
        return None, None
    
    # 随机抽样
    sample_size = int(total_images * sample_ratio)
    if sample_size < 1:
        sample_size = 1
    
    # 确保至少抽样1张图片，最多抽样10000张（防止内存不足）
    sample_size = max(1, min(sample_size, 10000))
    
    print(f"🎯 随机抽样 {sample_size} 张图片进行计算")
    
    # 随机选择图片
    sampled_paths = random.sample(all_image_paths, sample_size)
    
    # 计算统计信息
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    total_pixels = 0
    
    print("\n🔢 正在计算抽样图片的统计信息...")
    
    # 使用进度条
    pbar = tqdm(
        sampled_paths, 
        desc="处理抽样图片", 
        unit="张",
        bar_format="{l_bar}{bar:40}{r_bar}",
        ncols=80
    )
    
    processed_count = 0
    error_count = 0
    
    for img_path in pbar:
        try:
            # 打开图片并调整大小
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0  # 归一化到[0,1]
            
            # 计算统计量
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            
            h, w = r.shape
            total_pixels += h * w
            
            sum_r += r.sum()
            sum_g += g.sum()
            sum_b += b.sum()
            
            sum_sq_r += (r ** 2).sum()
            sum_sq_g += (g ** 2).sum()
            sum_sq_b += (b ** 2).sum()
            
            processed_count += 1
            
            # 更新进度条
            pbar.set_postfix({
                "成功": processed_count,
                "失败": error_count,
                "总抽样": sample_size
            })
            
        except Exception as e:
            error_count += 1
    
    pbar.close()
    
    print(f"✅ 抽样计算完成！成功: {processed_count}, 失败: {error_count}")
    
    if total_pixels == 0:
        print("❌ 错误: 没有处理任何有效图片！")
        return None, None
    
    # 计算均值和标准差
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels
    
    std_r = np.sqrt(sum_sq_r / total_pixels - mean_r ** 2)
    std_g = np.sqrt(sum_sq_g / total_pixels - mean_g ** 2)
    std_b = np.sqrt(sum_sq_b / total_pixels - mean_b ** 2)
    
    mean = [float(mean_r), float(mean_g), float(mean_b)]
    std = [float(std_r), float(std_g), float(std_b)]
    
    return mean, std, total_images, sample_size

# 主程序
if __name__ == "__main__":
    # 直接在这里设置您的路径
    train_folder = r"E:\mastercode\data\RSCD\train"
    
    print("=" * 60)
    print("📊 抽样计算数据集统计信息")
    print("=" * 60)
    
    mean, std, total_images, sample_size = calculate_stats_with_sampling(
        train_folder=train_folder,
        sample_ratio=0.1,  # 10%
        img_size=224,
        seed=42
    )
    
    if mean is not None and std is not None:
        print("\n" + "=" * 60)
        print("📋 抽样计算结果")
        print("=" * 60)
        print(f"📷 总图片数: {total_images}")
        print(f"🎯 抽样图片数: {sample_size}")
        print(f"📊 抽样比例: {sample_size/total_images*100:.2f}%")
        print(f"🎨 均值 (RGB): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
        print(f"📏 标准差 (RGB): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
        print("=" * 60)
        
        # 保存结果
        results = {
            "total_images": total_images,
            "sampled_images": sample_size,
            "sampling_ratio": sample_size / total_images,
            "mean": mean,
            "std": std,
            "transform_usage": {
                "normalize_mean": mean,
                "normalize_std": std
            }
        }
        
        # 保存为JSON文件
        with open("sampled_stats.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON结果已保存到: sampled_stats.json")
        
        # 保存为TXT文件
        with open("sampled_stats.txt", "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("抽样计算结果\n")
            f.write("=" * 60 + "\n")
            f.write(f"总图片数: {total_images}\n")
            f.write(f"抽样图片数: {sample_size}\n")
            f.write(f"抽样比例: {sample_size/total_images*100:.2f}%\n")
            f.write(f"均值: {mean}\n")
            f.write(f"标准差: {std}\n")
            f.write("\n")
            f.write("在transform中使用:\n")
            f.write("=" * 60 + "\n")
            f.write("transforms.Normalize(\n")
            f.write(f"    mean={mean},\n")
            f.write(f"    std={std}\n")
            f.write(")\n")
        
        print(f"📝 文本结果已保存到: sampled_stats.txt")
        
    else:
        print("❌ 计算失败！")