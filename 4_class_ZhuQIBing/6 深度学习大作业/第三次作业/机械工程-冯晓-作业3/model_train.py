# import copy
# import time
#
# import torch
# from torchvision.datasets import ImageFolder
# from torchvision import transforms
# import torch.utils.data as Data
# import numpy as np
# import matplotlib.pyplot as plt
# from model import GoogLeNet, Inception
# import torch.nn as nn
# import pandas as pd
#
# def train_val_data_process():
#     # 定义数据集的路径
#     ROOT_TRAIN = r'data\train'
#
#     normalize = transforms.Normalize([0.22890568,0.19639583,0.1433638 ], [0.09950783, 0.07997292, 0.06596899])
#     # 定义数据集处理方法变量
#     train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
#     # 加载数据集
#     train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)
#
#     train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
#     train_dataloader = Data.DataLoader(dataset=train_data,
#                                        batch_size=32,
#                                        shuffle=True,
#                                        num_workers=2)
#
#     val_dataloader = Data.DataLoader(dataset=val_data,
#                                        batch_size=32,
#                                        shuffle=True,
#                                        num_workers=2)
#
#     return train_dataloader, val_dataloader
#
#
#
# def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
#     # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 使用Adam优化器，学习率为0.001
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     # 损失函数为交叉熵函数
#     criterion = nn.CrossEntropyLoss()
#     # 将模型放入到训练设备中
#     model = model.to(device)
#     # 复制当前模型的参数
#     best_model_wts = copy.deepcopy(model.state_dict())
#
#     # 初始化参数
#     # 最高准确度
#     best_acc = 0.0
#     # 训练集损失列表
#     train_loss_all = []
#     # 验证集损失列表
#     val_loss_all = []
#     # 训练集准确度列表
#     train_acc_all = []
#     # 验证集准确度列表
#     val_acc_all = []
#     # 当前时间
#     since = time.time()
#
#     for epoch in range(num_epochs):
#         print("Epoch {}/{}".format(epoch, num_epochs-1))
#         print("-"*10)
#
#         # 初始化参数
#         # 训练集损失函数
#         train_loss = 0.0
#         # 训练集准确度
#         train_corrects = 0
#         # 验证集损失函数
#         val_loss = 0.0
#         # 验证集准确度
#         val_corrects = 0
#         # 训练集样本数量
#         train_num = 0
#         # 验证集样本数量
#         val_num = 0
#
#         # 对每一个mini-batch训练和计算
#         for step, (b_x, b_y) in enumerate(train_dataloader):
#             # 将特征放入到训练设备中
#             b_x = b_x.to(device)
#             # 将标签放入到训练设备中
#             b_y = b_y.to(device)
#             # 设置模型为训练模式
#             model.train()
#
#             # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
#             output = model(b_x)
#             # 查找每一行中最大值对应的行标
#             pre_lab = torch.argmax(output, dim=1)
#             # 计算每一个batch的损失函数
#             loss = criterion(output, b_y)
#
#             # 将梯度初始化为0
#             optimizer.zero_grad()
#             # 反向传播计算
#             loss.backward()
#             # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
#             optimizer.step()
#             # 对损失函数进行累加
#             train_loss += loss.item() * b_x.size(0)
#             # 如果预测正确，则准确度train_corrects加1
#             train_corrects += torch.sum(pre_lab == b_y.data)
#             # 当前用于训练的样本数量
#             train_num += b_x.size(0)
#         for step, (b_x, b_y) in enumerate(val_dataloader):
#             # 将特征放入到验证设备中
#             b_x = b_x.to(device)
#             # 将标签放入到验证设备中
#             b_y = b_y.to(device)
#             # 设置模型为评估模式
#             model.eval()
#             # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
#             output = model(b_x)
#             # 查找每一行中最大值对应的行标
#             pre_lab = torch.argmax(output, dim=1)
#             # 计算每一个batch的损失函数
#             loss = criterion(output, b_y)
#
#
#             # 对损失函数进行累加
#             val_loss += loss.item() * b_x.size(0)
#             # 如果预测正确，则准确度train_corrects加1
#             val_corrects += torch.sum(pre_lab == b_y.data)
#             # 当前用于验证的样本数量
#             val_num += b_x.size(0)
#
#         # 计算并保存每一次迭代的loss值和准确率
#         # 计算并保存训练集的loss值
#         train_loss_all.append(train_loss / train_num)
#         # 计算并保存训练集的准确率
#         train_acc_all.append(train_corrects.double().item() / train_num)
#
#         # 计算并保存验证集的loss值
#         val_loss_all.append(val_loss / val_num)
#         # 计算并保存验证集的准确率
#         val_acc_all.append(val_corrects.double().item() / val_num)
#
#         print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
#         print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))
#
#         if val_acc_all[-1] > best_acc:
#             # 保存当前最高准确度
#             best_acc = val_acc_all[-1]
#             # 保存当前最高准确度的模型参数
#             best_model_wts = copy.deepcopy(model.state_dict())
#
#         # 计算训练和验证的耗时
#         time_use = time.time() - since
#         print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))
#
#     # 选择最优参数，保存最优参数的模型
#     model.load_state_dict(best_model_wts)
#     # torch.save(model.load_state_dict(best_model_wts), "C:/Users/86159/Desktop/LeNet/best_model.pth")
#     torch.save(best_model_wts, "/GooLeNet-2/best_model.pth")
#
#
#     train_process = pd.DataFrame(data={"epoch":range(num_epochs),
#                                        "train_loss_all":train_loss_all,
#                                        "val_loss_all":val_loss_all,
#                                        "train_acc_all":train_acc_all,
#                                        "val_acc_all":val_acc_all,})
#
#     return train_process
#
#
# def matplot_acc_loss(train_process):
#     # 显示每一次迭代后的训练集和验证集的损失函数和准确率
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
#     plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("Loss")
#     plt.subplot(1, 2, 2)
#     plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
#     plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
#     plt.xlabel("epoch")
#     plt.ylabel("acc")
#     plt.legend()
#     plt.show()
#
#
# if __name__ == '__main__':
#     # 加载需要的模型
#     GoogLeNet = GoogLeNet(Inception)
#     # 加载数据集
#     train_data, val_data = train_val_data_process()
#     # 利用现有的模型进行模型的训练
#     train_process = train_model_process(GoogLeNet, train_data, val_data, num_epochs=50)
#     matplot_acc_loss(train_process)
import torch
import torchvision
# 打印核心版本+GPU状态
print("="*60)
print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ TorchVision版本: {torchvision.__version__}")
print(f"✅ CUDA版本: {torch.version.cuda}")
print(f"✅ 是否调用GPU: {torch.cuda.is_available()}")
print(f"✅ 你的显卡型号: {torch.cuda.get_device_name(0)}")
print("="*60)
import copy
import time
import pandas as pd
import torch
#import transforms
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import GoogLeNet, Inception
from tqdm import tqdm  # 导入tqdm


# 数据预处理函数（保持不变）
def train_val_data_process():
	#定义数据集路径
	ROOT_TRAIN = r'data\train'
	#数据归一化
	normalize = transforms.Normalize(mean=[0.22890999, 0.1963964, 0.14335695],std=[0.09950233, 0.07996743, 0.06593084])
	#定义数据集初始处理方法的变量
	train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

	#加载数据集   #第一个参数是路径，第二个参数是处理方法
	train_data =ImageFolder(root=ROOT_TRAIN,transform=train_transform)

	# print(train_data.class_to_idx)


	train_data, val_data = Data.random_split(train_data, [round(len(train_data) * 0.8), round(len(train_data) * 0.2)])

	train_dataloader = Data.DataLoader(dataset=train_data,
	                                   batch_size=32,
	                                   shuffle=True,
	                                   num_workers=0)

	val_dataloader = Data.DataLoader(dataset=val_data,
	                                 batch_size=32,
	                                 shuffle=False,
	                                 num_workers=0)

	return train_dataloader, val_dataloader



train_dataloader, val_dataloader = train_val_data_process()


# 创建模型训练函数（添加进度条）
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
		print(f"检测到 {torch.cuda.device_count()} 个GPU，使用DataParallel")
		model = nn.DataParallel(model)
	print(f"使用设备: {device}")

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()
	model = model.to(device)
	best_model_wts = copy.deepcopy(model.state_dict())

	# 初始化记录变量
	best_acc = 0.0
	train_loss_all = []
	val_loss_all = []
	train_acc_all = []
	val_acc_all = []
	since = time.time()

	# 外层循环：遍历所有epoch
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		print('-' * 10)

		# 初始化参数
		train_loss = 0
		val_loss = 0
		train_acc = 0
		val_acc = 0
		train_num = 0
		val_num = 0

		# =============== 训练阶段 ===============
		model.train()

		# 创建训练进度条
		train_pbar = tqdm(train_dataloader,
		                  desc=f'Training Epoch {epoch + 1}/{num_epochs}',
		                  ncols=135,  # 进度条宽度
		                  leave=True)  # 完成后保留进度条

		for step, (b_x, b_y) in enumerate(train_pbar):
			b_x = b_x.to(device)
			b_y = b_y.to(device)

			# 前向传播
			output = model(b_x)
			pre_lab = torch.argmax(output, dim=1)
			loss = criterion(output, b_y)

			# 反向传播
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# 累计统计
			batch_loss = loss.item()
			batch_acc = torch.sum(pre_lab == b_y.data).item() / b_x.size(0)
			train_loss += batch_loss * b_x.size(0)
			train_acc += torch.sum(pre_lab == b_y.data)
			train_num += b_x.size(0)

			# 更新进度条描述
			train_pbar.set_postfix({
				'Loss': f'{batch_loss:.4f}',
				'Acc': f'{batch_acc:.4f}',
				'Avg_Loss': f'{train_loss / train_num:.4f}',
				'Avg_Acc': f'{(train_acc / train_num).item():.4f}' if train_num > 0 else '0.0000'
			})

		# 关闭训练进度条
		train_pbar.close()

		# =============== 验证阶段 ===============
		model.eval()

		# 创建验证进度条
		val_pbar = tqdm(val_dataloader,
		                desc=f'Validation Epoch {epoch + 1}/{num_epochs}',
		                ncols=135,
		                leave=True)

		with torch.no_grad():  # 验证阶段不需要计算梯度
			for step, (b_x, b_y) in enumerate(val_pbar):
				b_x = b_x.to(device)
				b_y = b_y.to(device)

				# 前向传播
				output = model(b_x)
				pre_lab = torch.argmax(output, dim=1)
				loss = criterion(output, b_y)

				# 累计统计
				batch_loss = loss.item()
				batch_acc = torch.sum(pre_lab == b_y.data).item() / b_x.size(0)
				val_loss += batch_loss * b_x.size(0)
				val_acc += torch.sum(pre_lab == b_y.data)
				val_num += b_x.size(0)

				# 更新进度条描述
				val_pbar.set_postfix({
					'Loss': f'{batch_loss:.4f}',
					'Acc': f'{batch_acc:.4f}',
					'Avg_Loss': f'{val_loss / val_num:.4f}',
					'Avg_Acc': f'{(val_acc / val_num).item():.4f}' if val_num > 0 else '0.0000'
				})

		# 关闭验证进度条
		val_pbar.close()

		# ================ 记录和输出当前epoch结果 ================
		# 计算并保存每一轮次迭代的loss值和准确率
		epoch_train_loss = train_loss / train_num
		epoch_train_acc = (train_acc.double() / train_num).cpu().item()
		epoch_val_loss = val_loss / val_num
		epoch_val_acc = (val_acc.double() / val_num).cpu().item()

		train_loss_all.append(epoch_train_loss)
		train_acc_all.append(epoch_train_acc)
		val_loss_all.append(epoch_val_loss)
		val_acc_all.append(epoch_val_acc)

		# 打印当前epoch的结果
		print('Epoch {} - Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(
			epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))

		# 寻找最高准确度权重参数
		if epoch_val_acc > best_acc:
			best_acc = epoch_val_acc
			best_model_wts = copy.deepcopy(model.state_dict())
			print(f'*** 新的最佳准确率: {best_acc:.4f} ***')

		# 训练耗时
		time_used = time.time() - since
		print('训练和验证耗费的时间: {:.0f}m {:.0f}s\n'.format(time_used // 60, time_used % 60))

	# ================= 训练结束后的处理 =================
	torch.save(best_model_wts, './best_model.pth')
	print(f'最佳模型已保存，验证准确率: {best_acc:.4f}')

	# 创建训练过程记录DataFrame
	train_process = pd.DataFrame({
		'epoch': range(num_epochs),
		'train_loss_all': train_loss_all,
		'val_loss_all': val_loss_all,
		'train_acc_all': train_acc_all,
		'val_acc_all': val_acc_all,
	})
	return train_process


# 可视化函数（保持不变）
def matplot_acc_loss(train_process):
	# 设置画布大小，和你示例图一致的宽屏样式
	plt.figure(figsize=(12, 4), dpi=100)
	# 设置全局字体，防止中文乱码（可选，不加也不影响曲线）
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False

	# ========== 左图：训练/验证 Loss 曲线 (和你示例图完全一致) ==========
	plt.subplot(1, 2, 1)
	# 训练loss：红色实线+圆点，线条加粗，和示例图一致
	plt.plot(train_process["epoch"], train_process.train_loss_all, 'r-o', label='train loss', linewidth=1.5,
			 markersize=3)
	# 验证loss：蓝色实线+方块，线条加粗，和示例图一致
	plt.plot(train_process["epoch"], train_process.val_loss_all, 'b-s', label='val loss', linewidth=1.5, markersize=3)
	# 添加网格线，和你示例图一致的浅色网格，必加！
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.legend(loc='upper right', fontsize=10)
	plt.xlabel('epoch', fontsize=11)
	plt.ylabel('Loss', fontsize=11)
	plt.title('训练集与验证集损失值变化', fontsize=12)

	# ========== 右图：训练/验证 Acc 曲线 (和你示例图完全一致) ==========
	plt.subplot(1, 2, 2)
	# 训练acc：红色实线+圆点，线条加粗
	plt.plot(train_process["epoch"], train_process.train_acc_all, 'r-o', label='train acc', linewidth=1.5, markersize=3)
	# 验证acc：蓝色实线+方块，线条加粗
	plt.plot(train_process["epoch"], train_process.val_acc_all, 'b-s', label='val acc', linewidth=1.5, markersize=3)
	# 添加网格线，和示例图一致
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.legend(loc='lower right', fontsize=10)
	plt.xlabel('epoch', fontsize=11)
	plt.ylabel('Accuracy', fontsize=11)
	plt.title('训练集与验证集准确率变化', fontsize=12)

	# ========== 核心：自动保存高清图片到项目根目录 ==========
	plt.tight_layout()  # 自动调整子图间距，防止重叠
	plt.savefig('train_curve.png', dpi=300, bbox_inches='tight')  # dpi=300 高清，打印无压力
	# 弹出曲线窗口查看
	plt.show()


# 主程序入口
if __name__ == '__main__':
	GooLeNet = GoogLeNet( Inception )
	train_process = train_model_process(GooLeNet, train_dataloader, val_dataloader, num_epochs=50)
	matplot_acc_loss(train_process)

