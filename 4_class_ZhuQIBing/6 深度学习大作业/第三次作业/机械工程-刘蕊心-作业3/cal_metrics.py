# import torch
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report
# import numpy as np
# from model import InceptionResNet
#
# # ================= 配置 =================
# ROOT_TEST = r'./data/Tomato_Dataset/val'  # 用验证集做测试
# MODEL_PATH = 'best_model.pth'
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # =======================================
#
# def get_metrics():
# 	# 保持和训练一致的预处理
# 	transform = transforms.Compose([
# 		transforms.Resize((224, 224)),
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187],
# 		                     std=[0.1802, 0.1568, 0.1944])
# 	])
#
# 	dataset = ImageFolder(ROOT_TEST, transform=transform)
# 	dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
# 	print(f"正在计算详细指标... (共 {len(dataset)} 张图片)")
#
# 	model = InceptionResNet(num_classes=10).to(DEVICE)
# 	model.load_state_dict(torch.load(MODEL_PATH))
# 	model.eval()
#
# 	y_true = []
# 	y_pred = []
#
# 	with torch.no_grad():
# 		for inputs, labels in dataloader:
# 			inputs = inputs.to(DEVICE)
# 			outputs = model(inputs)
# 			_, preds = torch.max(outputs, 1)
#
# 			y_true.extend(labels.cpu().numpy())
# 			y_pred.extend(preds.cpu().numpy())
#
# 	# 生成报告
# 	target_names = dataset.classes
# 	report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
# 	print("\n" + "=" * 20 + " 详细分类报告 " + "=" * 20)
# 	print(report)
# 	print("=" * 50)
#
#
# if __name__ == '__main__':
# 	get_metrics()


import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from model import InceptionResNet, GoogLeNet

# ================= 【🔴 修改这里】 =================
MODEL_TYPE = 'GoogLeNet'
MODEL_PATH = './results/GoogLeNet/best_model.pth'  # 确保路径对应
# ===============================================

ROOT_TEST = r'./data/Tomato_Dataset/val'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_metrics():
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187], std=[0.1802, 0.1568, 0.1944])
	])
	
	dataset = ImageFolder(ROOT_TEST, transform=transform)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
	
	print(f"正在计算详细指标 | 模型: {MODEL_TYPE}")
	
	if MODEL_TYPE == 'GoogLeNet':
		model = GoogLeNet(num_classes=10).to(DEVICE)
	else:
		model = InceptionResNet(num_classes=10).to(DEVICE)
	
	if os.path.exists(MODEL_PATH):
		model.load_state_dict(torch.load(MODEL_PATH))
	else:
		print(f"❌ 错误: 找不到权重文件 {MODEL_PATH}")
		return
	
	model.eval()
	y_true = []
	y_pred = []
	
	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs = inputs.to(DEVICE)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			y_true.extend(labels.cpu().numpy())
			y_pred.extend(preds.cpu().numpy())
	
	target_names = dataset.classes
	# 生成 4 位小数的报告，方便填论文
	report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
	print("\n" + "=" * 50)
	print(f"分类报告 ({MODEL_TYPE})")
	print(report)
	print("=" * 50)


if __name__ == '__main__':
	get_metrics()