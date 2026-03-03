# import torch
# from torch import nn
# from torchsummary import summary
# import torch.nn.functional as F
#
# #结合Inception多尺度感知以及ResNet残差跳接改善梯度消失
# class InceptionRes(nn.Module):
# 	def __init__(self, in_channels, c1, c2, c3, c4):
# 		super(InceptionRes, self).__init__()
#
# 		#线路1:1*1卷积
# 		self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
#
# 		self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
# 		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
#
# 		self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
# 		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
#
# 		# 线路4: MaxPool -> 1x1
# 		self.p4_1 = nn.AvgPool2d(kernel_size=3, padding=1 ,stride=1)
# 		self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
#
# 		# --- 残差连接的准备工作 ---
# 		# 计算 Inception 输出的总通道数
# 		self.total_out_channels = c1 + c2[1] + c3[1] + c4
#
# 		# 如果输入和输出通道不一致，需要用1x1卷积来对齐，才能相加
# 		if in_channels != self.total_out_channels:
# 			self.shortcut = nn.Conv2d(in_channels, self.total_out_channels, kernel_size=1)
# 		else:
# 			self.shortcut = None
#
# 	def forward(self, x):
# 	# 正常经过 Inception 的四个分支
# 		p1 = F.relu(self.p1_1(x))
# 		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
# 		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
# 		p4 = F.relu(self.p4_2(self.p4_1(x)))
#
# 		# 拼接分支结果
# 		inception_output = torch.cat((p1, p2, p3, p4),1)
#
# 		#残差跳接
# 		residual = x
# 		if self.shortcut is not None:
# 			residual = self.shortcut(residual)
#
# 		#核心创新：输出 = Inception输出 + 原始输入x
# 		return F.relu(inception_output + residual)
#
#
# class InceptionResNet(nn.Module):
# 	def __init__(self, num_classes=10):
# 		super(InceptionResNet, self).__init__()
#
# 		# 第一部分：普通的卷积层处理输入
# 		self.b1 = nn.Sequential(
# 			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
# 			nn.Conv2d(64, 64, kernel_size=1),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 192, kernel_size=3, padding=1),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 		)
#
# 		# 第二部分：堆叠 InceptionRes 模块
# 		self.b2 = nn.Sequential(
# 			InceptionRes(192, 64, (96, 128), (16, 32), 32),
# 			InceptionRes(256, 128, (128, 192), (32, 96), 64),
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 		)
#
# 		self.b3 = nn.Sequential(
# 			InceptionRes(480, 192, (96, 208), (16, 48), 64),
# 			InceptionRes(512, 160, (112, 224), (24, 64), 64),
# 			InceptionRes(512, 128, (128, 256), (24, 64), 64),
# 			InceptionRes(512, 112, (144, 288), (32, 64), 64),
# 			InceptionRes(528, 256, (160, 320), (32, 128), 128),
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# 		)
#
# 		# 第三部分：分类层
# 		self.b4 = nn.Sequential(
# 			InceptionRes(832, 256, (160, 320), (32, 128), 128),
# 			InceptionRes(832, 384, (192, 384), (48, 128), 128),
# 			nn.AdaptiveAvgPool2d((1, 1)),
# 			nn.Flatten(),
# 			nn.Linear(1024, num_classes)  # 这里根据番茄数据集通常是10类
# 		)
#
# 	def forward(self, x):
# 		x = self.b1(x)
# 		x = self.b2(x)
# 		x = self.b3(x)
# 		x = self.b4(x)
# 		return x
#
# if __name__ == "__main__":
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 	# 实例化模型
# 	model = InceptionResNet(num_classes=10).to(device)
# 	# 输入是 (3, 224, 224) 因为是RGB彩色图片
# 	print(summary(model, (3, 224, 224)))


import torch
from torch import nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件：标准 Inception 模块 (用于原始 GoogLeNet)
# ==========================================
class Inception(nn.Module):
	def __init__(self, in_channels, c1, c2, c3, c4):
		super(Inception, self).__init__()
		# 线路1: 1x1卷积
		self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
		# 线路2: 1x1 -> 3x3
		self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
		# 线路3: 1x1 -> 5x5
		self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
		# 线路4: MaxPool -> 1x1
		self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
		self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
	
	def forward(self, x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))
		# 标准 Inception：直接拼接，没有残差相加
		return torch.cat((p1, p2, p3, p4), 1)


# ==========================================
# 2. 改进组件：带残差的 InceptionRes 模块 (用于你的创新模型)
# ==========================================
class InceptionRes(nn.Module):
	def __init__(self, in_channels, c1, c2, c3, c4):
		super(InceptionRes, self).__init__()
		# 四个分支保持不变
		self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
		self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
		self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
		self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
		self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
		
		# 残差处理：计算输出总通道数
		self.total_out_channels = c1 + c2[1] + c3[1] + c4
		self.shortcut = None
		# 如果输入输出维度不一致，用1x1卷积对齐
		if in_channels != self.total_out_channels:
			self.shortcut = nn.Conv2d(in_channels, self.total_out_channels, kernel_size=1)
	
	def forward(self, x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))
		inception_output = torch.cat((p1, p2, p3, p4), 1)
		
		# 残差连接
		residual = x
		if self.shortcut is not None:
			residual = self.shortcut(x)
		return F.relu(inception_output + residual)


# ==========================================
# 3. 网络骨架 (通用)
# ==========================================
class BaseGoogLeNet(nn.Module):
	def __init__(self, block_type, num_classes=10):
		super(BaseGoogLeNet, self).__init__()
		self.block1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)
		self.block2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=1),
			nn.ReLU(),
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		)
		self.block3 = nn.Sequential(
			block_type(192, 64, (96, 128), (16, 32), 32),
			block_type(256, 128, (128, 192), (32, 96), 64),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		)
		self.block4 = nn.Sequential(
			block_type(480, 192, (96, 208), (16, 48), 64),
			block_type(512, 160, (112, 224), (24, 64), 64),
			block_type(512, 128, (128, 256), (24, 64), 64),
			block_type(512, 112, (144, 288), (32, 64), 64),
			block_type(528, 256, (160, 320), (32, 128), 128),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		)
		self.block5 = nn.Sequential(
			block_type(832, 256, (160, 320), (32, 128), 128),
			block_type(832, 384, (192, 384), (48, 128), 128),
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(1024, num_classes)
		)
	
	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)
		return x


# --- 供外部调用的两个类 ---

# 原始 GoogLeNet (对照实验用这个)
class GoogLeNet(BaseGoogLeNet):
	def __init__(self, num_classes=10):
		super(GoogLeNet, self).__init__(Inception, num_classes)


# 改进版 InceptionResNet (你的创新模型用这个)
class InceptionResNet(BaseGoogLeNet):
	def __init__(self, num_classes=10):
		super(InceptionResNet, self).__init__(InceptionRes, num_classes)