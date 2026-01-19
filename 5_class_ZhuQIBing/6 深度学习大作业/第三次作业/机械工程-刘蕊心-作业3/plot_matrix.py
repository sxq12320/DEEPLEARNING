# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# import numpy as np
# from model import InceptionResNet
#
# # ================= 配置 =================
# ROOT_TEST = r'./data/Tomato_Dataset/val'
# MODEL_PATH = 'best_model.pth'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # =======================================
#
# def plot_confusion_matrix():
# 	# 预处理
# 	transform = transforms.Compose([
# 		transforms.Resize((224, 224)),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187],
# 		                     std=[0.1802, 0.1568, 0.1944])
# 	])
#
# 	dataset = datasets.ImageFolder(ROOT_TEST, transform=transform)
# 	dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
# 	# 简化的类别名（为了画图不拥挤，去掉前面的Tomato___）
# 	class_names = [c.replace('Tomato___', '') for c in dataset.classes]
#
# 	model = InceptionResNet(num_classes=10).to(DEVICE)
# 	model.load_state_dict(torch.load(MODEL_PATH))
# 	model.eval()
#
# 	y_true = []
# 	y_pred = []
#
# 	print("正在生成混淆矩阵数据...")
# 	with torch.no_grad():
# 		for inputs, labels in dataloader:
# 			inputs = inputs.to(DEVICE)
# 			outputs = model(inputs)
# 			_, preds = torch.max(outputs, 1)
# 			y_true.extend(labels.cpu().numpy())
# 			y_pred.extend(preds.cpu().numpy())
#
# 	# 计算混淆矩阵
# 	cm = confusion_matrix(y_true, y_pred)
# 	# 归一化（看百分比更直观）
# 	cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
# 	# 画图
# 	plt.figure(figsize=(12, 10))
# 	sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
# 	            xticklabels=class_names, yticklabels=class_names)
#
# 	plt.title('Confusion Matrix (Normalized)')
# 	plt.ylabel('True Label')
# 	plt.xlabel('Predicted Label')
# 	plt.xticks(rotation=45, ha='right')
# 	plt.tight_layout()
# 	plt.savefig('confusion_matrix.png', dpi=300)
# 	print("✅ 混淆矩阵已保存为 confusion_matrix.png")
# 	plt.show()
#
#
# if __name__ == '__main__':
# 	plot_confusion_matrix()


import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os
from model import InceptionResNet, GoogLeNet

# ================= 【🔴 修改这里】 =================
MODEL_TYPE = 'GoogLeNet'
MODEL_PATH = './results/GoogLeNet/best_model.pth'
SAVE_NAME = 'confusion_matrix_GoogLeNet.png'  # 图片名字也改一下，别覆盖了
# ===============================================

ROOT_TEST = r'./data/Tomato_Dataset/val'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix():
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187], std=[0.1802, 0.1568, 0.1944])
	])
	
	dataset = datasets.ImageFolder(ROOT_TEST, transform=transform)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
	class_names = [c.replace('Tomato___', '') for c in dataset.classes]
	
	if MODEL_TYPE == 'GoogLeNet':
		model = GoogLeNet(num_classes=10).to(DEVICE)
	else:
		model = InceptionResNet(num_classes=10).to(DEVICE)
	
	if os.path.exists(MODEL_PATH):
		model.load_state_dict(torch.load(MODEL_PATH))
	else:
		print(f"❌ 找不到权重 {MODEL_PATH}")
		return
	
	model.eval()
	y_true = []
	y_pred = []
	
	print(f"正在生成混淆矩阵 | 模型: {MODEL_TYPE}")
	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs = inputs.to(DEVICE)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			y_true.extend(labels.cpu().numpy())
			y_pred.extend(preds.cpu().numpy())
	
	cm = confusion_matrix(y_true, y_pred)
	cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	
	plt.figure(figsize=(12, 10))
	sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
	            xticklabels=class_names, yticklabels=class_names)
	
	plt.title(f'Confusion Matrix ({MODEL_TYPE})')
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()
	plt.savefig(SAVE_NAME, dpi=300)
	print(f"✅ 图片已保存: {SAVE_NAME}")
	plt.show()


if __name__ == '__main__':
	plot_confusion_matrix()