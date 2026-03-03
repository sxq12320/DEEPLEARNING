# from PIL import Image
# import os
# import numpy as np
#
# # 文件夹路径，包含所有图片文件
# folder_path = r'C:\Users\17480\Desktop\InceptionResNet\data\Tomato_Dataset\train'
#
# # 初始化累积变量
# total_pixels = 0
# sum_normalized_pixel_values = np.zeros(3)  # 如果是RGB图像，需要三个通道的均值和方差
#
# # 遍历文件夹中的图片文件
# for root, dirs, files in os.walk(folder_path):
#     for filename in files:
#         if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 可根据实际情况添加其他格式
#             image_path = os.path.join(root, filename)
#             image = Image.open(image_path)
#             image_array = np.array(image)
#
#             # 归一化像素值到0-1之间
#             normalized_image_array = image_array / 255.0
#
#             # print(image_path)
#             # print(normalized_image_array.shape)
#             # 累积归一化后的像素值和像素数量
#             total_pixels += normalized_image_array.size
#             sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))
#
# # 计算均值和方差
# mean = sum_normalized_pixel_values / total_pixels
#
#
# sum_squared_diff = np.zeros(3)
# for root, dirs, files in os.walk(folder_path):
#     for filename in files:
#         if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#             image_path = os.path.join(root, filename)
#             image = Image.open(image_path)
#             image_array = np.array(image)
#             # 归一化像素值到0-1之间
#             normalized_image_array = image_array / 255.0
#             # print(normalized_image_array.shape)
#             # print(mean.shape)
#             # print(image_path)
#
#             try:
#                 diff = (normalized_image_array - mean) ** 2
#                 sum_squared_diff += np.sum(diff, axis=(0, 1))
#             except:
#                 print(f"捕获到自定义异常")
#             # diff = (normalized_image_array - mean) ** 2
#             # sum_squared_diff += np.sum(diff, axis=(0, 1))
#
# variance = sum_squared_diff / total_pixels
#
# print("Mean:", mean)
# print("Variance:", variance)
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= 配置区域 =================
# 注意：这里改成相对路径，前面加个点 '.' 表示当前目录
# 或者写完整的绝对路径 C:/Users/17480/Desktop/...
TRAIN_DATA_PATH = './data/Tomato_Dataset/val'


# ===========================================

def get_mean_std(data_path):
	print(f"正在扫描数据集: {data_path} ...")
	
	# 只需要把图片转成Tensor，不需要Resize（计算原始数据的均值更准）
	# 也不需要Normalize，因为我们就是来算它的
	transform = transforms.Compose([
		transforms.Resize((224, 224)),  # 为了统一计算方便，建议还是Resize一下
		transforms.ToTensor()
	])
	
	try:
		# 加载数据集
		dataset = ImageFolder(data_path, transform=transform)
		dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
	except FileNotFoundError:
		print(f"❌ 错误：找不到路径 {data_path}，请检查文件夹是否存在！")
		return None, None
	except Exception as e:
		print(f"❌ 加载数据出错: {e}")
		return None, None
	
	if len(dataset) == 0:
		print("❌ 错误：文件夹为空，未找到任何图片！")
		return None, None
	
	print(f"✅ 成功加载 {len(dataset)} 张图片，开始计算...")
	
	# 初始化变量
	channels_sum = torch.zeros(3)
	channels_sq_sum = torch.zeros(3)
	num_batches = 0
	total_pixels = 0
	
	# 这里的 tqdm 是进度条
	for data, _ in tqdm(dataloader):
		# data维度: [batch_size, 3, 224, 224]
		
		# 累加每个通道的像素值
		# dim=[0, 2, 3] 表示在 batch, height, width 维度上求和，只保留 channel 维度
		channels_sum += torch.sum(data, dim=[0, 2, 3])
		
		# 累加平方和（用于算方差）
		channels_sq_sum += torch.sum(data ** 2, dim=[0, 2, 3])
		
		# 统计总像素点个数 (batch_size * 224 * 224)
		# data.size(0)是batch大小, size(2)是H, size(3)是W
		total_pixels += data.size(0) * data.size(2) * data.size(3)
	
	# 计算均值
	mean = channels_sum / total_pixels
	
	# 计算标准差: std = sqrt(E[x^2] - (E[x])^2)
	std = (channels_sq_sum / total_pixels - mean ** 2) ** 0.5
	
	return mean, std


if __name__ == '__main__':
	# 计算
	mean, std = get_mean_std(TRAIN_DATA_PATH)
	
	if mean is not None:
		print("\n" + "=" * 30)
		print("🎉 计算完成！请将以下参数填入你的代码中：")
		print("=" * 30)
		print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
		print(f"std  = [{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}]")
		print("=" * 30)
		
		print("\n也就是修改 model_train.py 中的 transforms.Normalize 为：")
		print(f"transforms.Normalize(mean=[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}],")
		print(f"                     std=[{std[0]:.4f},  {std[1]:.4f},  {std[2]:.4f}])")