import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os
import random
import numpy as np

# ================= 配置 =================
# 指向你的训练集路径
DATA_DIR = r'./data/Tomato_Dataset/train'
SAVE_NAME = 'dataset_samples_1.png'


# =======================================

def show_dataset_samples():
	# 加载数据集结构
	dataset = ImageFolder(DATA_DIR)
	classes = dataset.classes
	
	# 设置画布：2行5列（共10类）
	fig, axes = plt.subplots(2, 5, figsize=(15, 6))
	fig.suptitle('Sample Images from PlantVillage Tomato Dataset', fontsize=16)
	
	# 稍微调整子图间距
	plt.subplots_adjust(wspace=0.1, hspace=0.3)
	
	# 遍历每一类，随机选一张画出来
	for i, class_name in enumerate(classes):
		# 找到该类对应的文件夹
		class_dir = os.path.join(DATA_DIR, class_name)
		images = os.listdir(class_dir)
		
		# 随机选一张图
		img_name = random.choice(images)
		img_path = os.path.join(class_dir, img_name)
		
		# 读取图片
		img = plt.imread(img_path)
		
		# 计算行列位置
		row = i // 5
		col = i % 5
		ax = axes[row, col]
		
		# 显示图片
		ax.imshow(img)
		ax.axis('off')  # 去掉坐标轴
		
		# 处理一下类别名字（去掉 Tomato___ 前缀，太长了不好看）
		clean_name = class_name.replace('Tomato___', '').replace('_', ' ')
		
		# 加上标题
		ax.set_title(clean_name, fontsize=10, pad=5)
	
	plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight')
	print(f"✅ 数据集展示图已生成: {SAVE_NAME}")
	plt.show()


if __name__ == '__main__':
	show_dataset_samples()