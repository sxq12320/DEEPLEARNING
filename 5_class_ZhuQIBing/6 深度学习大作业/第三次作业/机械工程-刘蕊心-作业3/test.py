# import torch
# import torch.utils.data as Data
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from PIL import Image
# import os
#
# # 引入模型
# from model import InceptionResNet
#
# # ================= 配置区域 =================
# # 这里我们用 'val' 文件夹充当测试集
# ROOT_TEST = r'data/Tomato_Dataset/val'
# MODEL_PATH = 'best_model.pth'
# # ===========================================
#
# # 这是一个标准的 PlantVillage 番茄类别列表，用于打印结果
# CLASSES = [
# 	'Tomato_Bacterial_spot',
# 	'Tomato_Early_blight',
# 	'Tomato_Late_blight',
# 	'Tomato_Leaf_Mold',
# 	'Tomato_Septoria_leaf_spot',
# 	'Tomato_Spider_mites_Two_spotted_spider_mite',
# 	'Tomato_Target_Spot',
# 	'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
# 	'Tomato_Tomato_mosaic_virus',
# 	'Tomato_healthy'
# ]
#
#
# def test_data_process():
# 	# 预处理要和训练时保持一致
# 	test_transform = transforms.Compose([
# 		transforms.Resize((224, 224)),
# 		transforms.ToTensor(),
# 		# 你的训练代码
# 		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187],
# 		                     std=[0.1802, 0.1568, 0.1944])
# 		])
#
# 	if not os.path.exists(ROOT_TEST):
# 		print(f"❌ 错误: 找不到测试路径 {ROOT_TEST}")
# 		return None
#
# 	test_data = ImageFolder(ROOT_TEST, transform=test_transform)
#
# 	# 打印一下实际读到的类别，确保顺序和 CLASSES 对应（ImageFolder是按字母顺序排的）
# 	print(f"测试集类别: {test_data.classes}")
#
# 	test_dataloader = Data.DataLoader(dataset=test_data,
# 	                                  batch_size=1,  # 逐张测试
# 	                                  shuffle=False,
# 	                                  num_workers=0)
# 	return test_dataloader
#
#
# def test_model_accuracy(model, test_dataloader):
# 	device = "cuda" if torch.cuda.is_available() else 'cpu'
# 	model = model.to(device)
# 	model.eval()  # 开启评估模式
#
# 	test_corrects = 0.0
# 	test_num = 0
#
# 	print("正在计算整体准确率...")
#
# 	with torch.no_grad():
# 		for b_x, b_y in test_dataloader:
# 			b_x = b_x.to(device)
# 			b_y = b_y.to(device)
#
# 			output = model(b_x)
# 			pre_lab = torch.argmax(output, dim=1)
#
# 			test_corrects += torch.sum(pre_lab == b_y.data)
# 			test_num += b_x.size(0)
#
# 	test_acc = test_corrects.double().item() / test_num
# 	print(f"✅ 测试集样本总数: {test_num}")
# 	print(f"✅ 最终测试准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")
#
#
# def predict_single_image(model, image_path):
# 	device = "cuda" if torch.cuda.is_available() else 'cpu'
# 	model = model.to(device)
# 	model.eval()
#
# 	# 图片预处理
# 	transform = transforms.Compose([
# 		transforms.Resize((224, 224)),
# 		transforms.ToTensor(),
# 		transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 	])
#
# 	try:
# 		image = Image.open(image_path).convert('RGB')
# 		image = transform(image)
# 		image = image.unsqueeze(0)  # 增加 batch 维度: [1, 3, 224, 224]
# 		image = image.to(device)
#
# 		with torch.no_grad():
# 			output = model(image)
# 			pre_lab = torch.argmax(output, dim=1)
# 			class_index = pre_lab.item()
#
# 		print(f"\n🖼️ 图片: {image_path}")
# 		print(f"🏷️ 预测结果: {CLASSES[class_index]}")
#
# 	except Exception as e:
# 		print(f"❌ 读取图片失败: {e}")
#
#
# if __name__ == "__main__":
# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# 	# 1. 加载模型
# 	# 注意：这里的类别数必须和训练时一样（通常是10）
# 	model = InceptionResNet(num_classes=10)
#
# 	if os.path.exists(MODEL_PATH):
# 		model.load_state_dict(torch.load(MODEL_PATH))
# 		print("已加载 best_model.pth 权重")
# 	else:
# 		print("❌ 没找到 best_model.pth，请先运行 model_train.py")
# 		exit()
#
# 	# 2. 计算整体准确率
# 	dataloader = test_data_process()
# 	if dataloader:
# 		test_model_accuracy(model, dataloader)
#
# # 3. (可选) 单张图片测试
# # 你可以把一张图片路径放这里测试
# # predict_single_image(model, r'data/Tomato_Dataset/val/Tomato_healthy/some_image.jpg')


import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
# 导入两个模型
from model import InceptionResNet, GoogLeNet

# ================= 【🔴 重点修改区域】 =================
# 你要测试哪个文件夹里的模型？
RESULT_DIR = './results/GoogLeNet'

# 该文件夹对应的模型类型是？('GoogLeNet' 或 'InceptionResNet')
MODEL_TYPE = 'GoogLeNet'

# 验证集路径
ROOT_TEST = r'./data/Tomato_Dataset/val'


# ======================================================

def test_data_process():
	test_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		# 保持和训练一致的归一化
		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187], std=[0.1802, 0.1568, 0.1944])
	])
	
	test_data = ImageFolder(ROOT_TEST, transform=test_transform)
	test_dataloader = Data.DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=0)
	return test_dataloader


def test_model_accuracy(model, test_dataloader):
	device = "cuda" if torch.cuda.is_available() else 'cpu'
	model = model.to(device)
	model.eval()
	
	test_corrects = 0.0
	test_num = 0
	
	print(f"正在测试模型: {MODEL_TYPE} ...")
	
	with torch.no_grad():
		for b_x, b_y in test_dataloader:
			b_x = b_x.to(device)
			b_y = b_y.to(device)
			output = model(b_x)
			pre_lab = torch.argmax(output, dim=1)
			test_corrects += torch.sum(pre_lab == b_y.data)
			test_num += b_x.size(0)
	
	test_acc = test_corrects.double().item() / test_num
	print(f"✅ 模型路径: {RESULT_DIR}")
	print(f"✅ 最终测试准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")


if __name__ == "__main__":
	dataloader = test_data_process()
	
	# 1. 初始化对应结构的模型
	if MODEL_TYPE == 'GoogLeNet':
		model = GoogLeNet(num_classes=10)
	else:
		model = InceptionResNet(num_classes=10)
	
	# 2. 从指定文件夹加载权重
	model_path = os.path.join(RESULT_DIR, 'best_model.pth')
	
	if os.path.exists(model_path):
		model.load_state_dict(torch.load(model_path))
		print(f"已加载权重: {model_path}")
		test_model_accuracy(model, dataloader)
	else:
		print(f"❌ 错误: 在 {RESULT_DIR} 下没找到 best_model.pth，请先运行训练代码！")