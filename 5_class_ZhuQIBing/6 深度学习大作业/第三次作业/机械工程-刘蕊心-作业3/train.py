# import copy
# import time
# import os
# from zipfile import sizeEndCentDir
#
# import pandas as pd
# import torch
# from torch import nn
# from torchvision.datasets import ImageFolder
# from torchvision import transforms
# import torch.utils.data as Data
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from model import InceptionResNet
#
# # 数据集根目录 (下面必须包含 train 和 val 两个子文件夹)
# ROOT_DATASET = r'./data/Tomato_Dataset'
#
# def train_val_data_process():
# 	print(f"正在加载数据集，路径: {ROOT_DATASET} ...")
#
# 	#定义图像预处理：训练集数据加强、验证集只做标准化
# 	data_transforms = {
# 		'train': transforms.Compose([
# 			transforms.Resize((224, 224)),
# 			transforms.RandomHorizontalFlip(),
# 			transforms.ToTensor(),
# 			transforms.Normalize(mean=[0.4526, 0.4632, 0.4187],std=[0.1802,  0.1568,  0.1944])
# 		]),
#
# 		'val': transforms.Compose([
# 			transforms.Resize((224, 224)),
# 			transforms.ToTensor(),
# 			transforms.Normalize(mean=[0.4526, 0.4632, 0.4187], std=[0.1802, 0.1568, 0.1944])
# 		])
# 	}
#
#
# 	# 构造具体路径
# 	train_dir = os.path.join(ROOT_DATASET, 'train')
# 	val_dir = os.path.join(ROOT_DATASET, 'val')
# 	#检查路径是否存在
# 	if not os.path.exists(train_dir):
# 		raise FileNotFoundError(f"❌错误：找不到训练文件夹{train_dir}")
# 	if not os.path.exists(val_dir):
# 		raise FileNotFoundError(f"❌错误：找不到验证文件夹{val_dir}")
# 	#直接读取两个文件夹
# 	train_data = ImageFolder(root=train_dir, transform=data_transforms['train'])
# 	val_data = ImageFolder(root=val_dir, transform=data_transforms['val'])
#
# 	#获取类别名称
# 	class_names = train_data.classes
# 	print(f"检测到{len(class_names)}个类别：{class_names}")
# 	print(f"训练集数量：{len(train_data)}|验证集数量：{len(val_data)}")
#
# 	#创建 DataLoader
# 	train_dataloader = Data.DataLoader(train_data,
# 	                                   batch_size=BATCH_SIZE,
# 	                                   shuffle=True,# 训练集必须打乱
# 	                                   num_workers=0)
#
# 	val_dataloader = Data.DataLoader(val_data,
# 	                                 batch_size=BATCH_SIZE,
# 	                                 shuffle=False,
# 	                                 num_workers=0)
# 	return train_dataloader, val_dataloader, len(class_names)
#
# def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	print(f"使用设备: {device} ")
#
# 	# 使用 Adam 优化器，学习率设为 0.001
# 	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 	criterion = nn.CrossEntropyLoss()
# 	model = model.to(device)
#
# 	# 复制当前模型参数用于保存最佳模型
# 	best_model_wts = copy.deepcopy(model.state_dict())
# 	best_acc = 0.0
#
# 	#记录训练过程数据
# 	train_loss_all = []
# 	train_acc_all = []
# 	val_loss_all = []
# 	val_acc_all = []
# 	since = time.time()
#
# 	for epoch in range(num_epochs):
# 		print(f"\nEPOCH {epoch+1}/{num_epochs}")
# 		print('-'*10)
#
# 		#初始化统计数量
# 		train_loss = 0
# 		val_corrects = 0
# 		train_corrects = 0
# 		val_loss = 0
# 		train_num = 0
# 		val_num = 0
#
#
# 		#=============训练阶段=============
# 		model.train()
# 		train_pbar = tqdm(train_dataloader, desc='Training', ncols=100, leave=True)
#
# 		for step,(b_x, b_y) in enumerate(train_pbar):
# 			#GPU运算
# 			b_x = b_x.to(device)
# 			b_y = b_y.to(device)
#
# 			output = model(b_x)
# 			pre_lab = torch.argmax(output, dim=1)
# 			loss = criterion(output, b_y)
#
# 			optimizer.zero_grad()
# 			loss.backward()
# 			optimizer.step()
#
# 			train_loss += loss.item() * b_x.size(0)
# 			train_corrects += torch.sum(pre_lab == b_y.data)##########这行什么意思
# 			train_num += b_x.size(0)
#
# 			train_pbar.set_postfix({'Loss':f'{loss.item():.4f}'})
#
# 		train_pbar.close()
#
#
# 		# ==================== 验证阶段 ====================
# 		model.eval()
# 		val_pbar = tqdm(val_dataloader, desc='Validation', ncols=100, leave=True)
#
# 		with torch.no_grad():
# 			for step, (b_x, b_y) in enumerate(val_pbar):
# 				b_x = b_x.to(device)
# 				b_y = b_y.to(device)
#
# 				output = model(b_x)
# 				pre_lab = torch.argmax(output, dim=1)
# 				loss = criterion(output, b_y)
#
# 				val_loss += loss.item() * b_x.size(0)
# 				val_corrects += torch.sum(pre_lab == b_y.data)
# 				val_num += b_x.size(0)
#
# 				val_pbar.set_postfix({'Loss':f'{loss.item():.4f}'})
#
# 		val_pbar.close()
#
# 		# ==================== 计算指标 ================
# 		epoch_train_loss = train_loss / train_num
# 		epoch_val_loss = val_loss / val_num
# 		epoch_train_acc = (train_corrects.double() / train_num).cpu().item()
# 		epoch_val_acc = (val_corrects.double() / val_num).cpu().item()
#
# 		train_loss_all.append(epoch_train_loss)
# 		val_loss_all.append(epoch_val_loss)
# 		train_acc_all.append(epoch_train_acc)
# 		val_acc_all.append(epoch_val_acc)
#
# 		print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
# 		print(f'Val   Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
#
# 		#保存最佳模型参数
# 		if epoch_val_acc > best_acc:
# 			best_acc = epoch_val_acc
# 			best_model_wts = copy.deepcopy(model.state_dict())
# 			print(f'🏆 出现最高准确率！ 保存最佳权重... (Acc: {best_acc:.4f})')
#
# 	#训练结束
# 	time_used = time.time() - since
# 	print(f'\n训练结束 总耗时：{time_used // 60:.0f}m {time_used % 60:.0f} s')
# 	print(f'最佳验证集准确率；{best_acc:.4f}')
#
# 	#保存文件
# 	torch.save(best_model_wts, 'best_model.pth')
#
# 	#保存训练日志到CSV，方便后续绘图
# 	process_df = pd.DataFrame({
# 		'epoch': range(1, num_epochs + 1),
# 		'train_loss_all': train_loss_all,
# 		'train_acc_all': train_acc_all,
# 		'val_loss_all': val_loss_all,
# 		'val_acc_all': val_acc_all,
# 	})
# 	process_df.to_csv('log_improved.csv', index=False)
#
# 	return process_df
#
# def matplot_acc_loss(train_process):
# 	plt.figure(figsize=[12,4])
#
# 	plt.subplot(1, 2, 1)
# 	plt.plot(train_process['epoch'], train_process.train_loss_all, 'r-',label='Train_loss')
# 	plt.plot(train_process['epoch'], train_process.val_loss_all, 'b-',label='Val_loss')
# 	plt.title('Loss Curve')
# 	plt.legend()
# 	plt.grid(True)
#
# 	plt.subplot(1, 2, 2)
# 	plt.plot(train_process["epoch"], train_process.train_acc_all, 'r-', label='Train Acc')
# 	plt.plot(train_process["epoch"], train_process.val_acc_all, 'b-', label='Val Acc')
# 	plt.title("Accuracy Curve")
# 	plt.legend()
# 	plt.grid(True)
#
# 	plt.savefig('training_result.png')
# 	plt.show()
#
# BATCH_SIZE = 32
# NUM_EPOCHS = 30  # 建议跑30轮以上以看到明显效果
#
# if __name__ == '__main__':
# 	# 1. 加载数据
# 	train_loader, val_loader, num_classes = train_val_data_process()
#
# 	# 2. 初始化改进后的模型
# 	# 注意：这里调用的是 InceptionResNet，不是 ResNet18
# 	model = InceptionResNet(num_classes=num_classes)
#
# 	# 3. 开始训练
# 	train_process = train_model_process(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
#
# 	# 4. 画图
# 	matplot_acc_loss(train_process)


import copy
import time
import os
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from tqdm import tqdm
# 导入两个模型
from model import InceptionResNet, GoogLeNet

# ================= 【🔴 重点修改区域】 =================
# 1. 结果保存路径 (请手动创建 results 文件夹，或让代码自动创建)
#    如果要跑对照实验，请改成 './results/GoogLeNet'
RESULT_DIR = './results/GoogLeNet'

# 2. 选择模型类型: 'InceptionResNet' (改进) 或 'GoogLeNet' (原始)
MODEL_TYPE = 'GoogLeNet'

# 3. 数据集路径
ROOT_DATASET = r'./data/Tomato_Dataset'

# 4. 其他参数
BATCH_SIZE = 32
NUM_EPOCHS = 100
# ======================================================

# 自动创建结果文件夹 (防止报错)
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)
	print(f"📂 已自动创建文件夹: {RESULT_DIR}")


def train_val_data_process():
	print(f"正在加载数据集，路径: {ROOT_DATASET} ...")
	
	data_transforms = {
		'train': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# 使用你之前算好的均值方差
			transforms.Normalize(mean=[0.4526, 0.4632, 0.4187], std=[0.1802, 0.1568, 0.1944])
		]),
		'val': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.4526, 0.4632, 0.4187], std=[0.1802, 0.1568, 0.1944])
		])
	}
	
	train_dir = os.path.join(ROOT_DATASET, 'train')
	val_dir = os.path.join(ROOT_DATASET, 'val')
	
	train_data = ImageFolder(root=train_dir, transform=data_transforms['train'])
	val_data = ImageFolder(root=val_dir, transform=data_transforms['val'])
	
	class_names = train_data.classes
	print(f"检测到 {len(class_names)} 个类别")
	
	train_dataloader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	val_dataloader = Data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
	
	return train_dataloader, val_dataloader, len(class_names)


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"使用设备: {device} | 当前训练模型: {MODEL_TYPE}")
	
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()
	model = model.to(device)
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	
	train_loss_all = []
	train_acc_all = []
	val_loss_all = []
	val_acc_all = []
	since = time.time()
	
	for epoch in range(num_epochs):
		print(f"\nEPOCH {epoch + 1}/{num_epochs}")
		print('-' * 10)
		
		train_loss = 0
		train_corrects = 0
		train_num = 0
		val_loss = 0
		val_corrects = 0
		val_num = 0
		
		# --- 训练 ---
		model.train()
		train_pbar = tqdm(train_dataloader, desc='Training', ncols=100, leave=True)
		for step, (b_x, b_y) in enumerate(train_pbar):
			b_x = b_x.to(device)
			b_y = b_y.to(device)
			output = model(b_x)
			pre_lab = torch.argmax(output, dim=1)
			loss = criterion(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item() * b_x.size(0)
			train_corrects += torch.sum(pre_lab == b_y.data)
			train_num += b_x.size(0)
			train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
		train_pbar.close()
		
		# --- 验证 ---
		model.eval()
		val_pbar = tqdm(val_dataloader, desc='Validation', ncols=100, leave=True)
		with torch.no_grad():
			for step, (b_x, b_y) in enumerate(val_pbar):
				b_x = b_x.to(device)
				b_y = b_y.to(device)
				output = model(b_x)
				pre_lab = torch.argmax(output, dim=1)
				loss = criterion(output, b_y)
				val_loss += loss.item() * b_x.size(0)
				val_corrects += torch.sum(pre_lab == b_y.data)
				val_num += b_x.size(0)
				val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
		val_pbar.close()
		
		# --- 计算指标 ---
		epoch_train_loss = train_loss / train_num
		epoch_val_loss = val_loss / val_num
		epoch_train_acc = (train_corrects.double() / train_num).cpu().item()
		epoch_val_acc = (val_corrects.double() / val_num).cpu().item()
		
		train_loss_all.append(epoch_train_loss)
		val_loss_all.append(epoch_val_loss)
		train_acc_all.append(epoch_train_acc)
		val_acc_all.append(epoch_val_acc)
		
		print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
		print(f'Val   Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
		
		if epoch_val_acc > best_acc:
			best_acc = epoch_val_acc
			best_model_wts = copy.deepcopy(model.state_dict())
			print(f'🏆 出现最高准确率！ 保存最佳权重... (Acc: {best_acc:.4f})')
	
	time_used = time.time() - since
	print(f'\n训练结束 总耗时：{time_used // 60:.0f}m {time_used % 60:.0f} s')
	
	# 【🔴 修改点】保存模型到指定文件夹
	save_path = os.path.join(RESULT_DIR, 'best_model.pth')
	torch.save(best_model_wts, save_path)
	print(f"💾 模型已保存至: {save_path}")
	
	# 【🔴 修改点】保存CSV到指定文件夹
	process_df = pd.DataFrame({
		'epoch': range(1, num_epochs + 1),
		'train_loss_all': train_loss_all,
		'train_acc_all': train_acc_all,
		'val_loss_all': val_loss_all,
		'val_acc_all': val_acc_all,
	})
	csv_path = os.path.join(RESULT_DIR, 'train_log.csv')
	process_df.to_csv(csv_path, index=False)
	print(f"📊 日志已保存至: {csv_path}")
	
	return process_df


def matplot_acc_loss(train_process):
	plt.figure(figsize=[12, 4])
	plt.subplot(1, 2, 1)
	plt.plot(train_process['epoch'], train_process['train_loss_all'], 'r-', label='Train_loss')
	plt.plot(train_process['epoch'], train_process['val_loss_all'], 'b-', label='Val_loss')
	plt.title(f'Loss Curve ({MODEL_TYPE})')
	plt.legend()
	plt.grid(True)
	
	plt.subplot(1, 2, 2)
	plt.plot(train_process["epoch"], train_process['train_acc_all'], 'r-', label='Train Acc')
	plt.plot(train_process["epoch"], train_process['val_acc_all'], 'b-', label='Val Acc')
	plt.title(f'Accuracy Curve ({MODEL_TYPE})')
	plt.legend()
	plt.grid(True)
	
	# 【🔴 修改点】保存图片到指定文件夹
	png_path = os.path.join(RESULT_DIR, 'training_result.png')
	plt.savefig(png_path)
	print(f"📈 曲线图已保存至: {png_path}")
	plt.show()


if __name__ == '__main__':
	train_loader, val_loader, num_classes = train_val_data_process()
	
	# 根据配置选择模型
	if MODEL_TYPE == 'GoogLeNet':
		print(">>> 正在初始化原始 ddGoogLeNet (对照组)...")
		model = GoogLeNet(num_classes=num_classes)
	else:
		print(">>> 正在初始化改进版 InceptionResNet (实验组)...")
		model = InceptionResNet(num_classes=num_classes)
	
	train_process = train_model_process(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
	matplot_acc_loss(train_process)