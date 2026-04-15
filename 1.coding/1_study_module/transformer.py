import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm # 用于显示进度条，如果没有可以 pip install tqdm

def batch_convert_npy_to_png(dataset_dir: str, scale_factor: float = 1000.0, delete_original: bool = False):
    """
    批量将目录下的 .npy 深度图转换为 16-bit PNG。
    
    参数:
        dataset_dir: 包含 .npy 文件的根目录路径（支持遍历子文件夹）。
        scale_factor: 物理量缩放系数，默认 1000.0 (将米转换为毫米)。
        delete_original: 是否在成功保存 png 后删除原 .npy 文件。危险操作，请谨慎开启。
    """
    # 将字符串路径转换为 Path 对象，方便跨平台操作
    root_path = Path(dataset_dir)
    
    # 查找所有后缀为 .npy 的文件 (rglob 会递归查找所有子目录)
    npy_files = list(root_path.rglob("*.npy"))
    
    if not npy_files:
        print(f"在目录 {dataset_dir} 中未找到任何 .npy 文件。")
        return

    print(f"共找到 {len(npy_files)} 个 .npy 文件，开始转换...")
    
    # 记录转换失败的文件
    error_logs = []

    for npy_path in tqdm(npy_files, desc="Converting"):
        try:
            # 1. 读取深度矩阵
            depth_array = np.load(npy_path)
            
            # 2. 数据清洗 (Sanitize)
            # 传感器在盲区或反光区域可能返回 NaN 或 Inf，将其安全地置为 0
            depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 3. 缩放与量化 (Quantization)
            # 乘以缩放系数并四舍五入
            scaled_depth = np.round(depth_array * scale_factor)
            
            # 裁剪到 16-bit 无符号整数的安全范围内 [0, 65535]，防止数值溢出卷绕
            clipped_depth = np.clip(scaled_depth, 0, 65535)
            
            # 强制转换数据类型为 uint16
            depth_uint16 = clipped_depth.astype(np.uint16)
            
            # 4. 构建输出路径 (将 .npy 替换为 .png)
            png_path = npy_path.with_suffix('.png')
            
            # 5. 落盘保存
            # cv2.imwrite 遇到 uint16 矩阵会自动按 16-bit PNG 标准无损保存
            success = cv2.imwrite(str(png_path), depth_uint16)
            
            if success:
                # 6. 安全删除逻辑
                if delete_original:
                    os.remove(npy_path)
            else:
                error_logs.append(f"保存失败 (OpenCV写入错误): {png_path}")

        except Exception as e:
            error_logs.append(f"处理异常 {npy_path.name}: {str(e)}")

    # 总结报告
    print("\n转换任务结束！")
    if error_logs:
        print(f"有 {len(error_logs)} 个文件处理失败，详情如下：")
        for err in error_logs[:10]: # 只打印前 10 个错误防止刷屏
            print(err)
        if len(error_logs) > 10:
            print("...")
    else:
        print("所有文件均已完美转换！")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 替换为你实际存放深度图的数据集路径
    YOUR_DATASET_PATH = "E:/mastercode/data/Apple_RGB_D_Amoal/depth_maps"
    
    # 第一次运行建议保持 delete_original=False 进行测试
    # 确认生成的 png 能够被正常读取且数值正确后，再改为 True 进行彻底替换
    batch_convert_npy_to_png(
        dataset_dir=YOUR_DATASET_PATH, 
        scale_factor=1000.0, 
        delete_original=False 
    )