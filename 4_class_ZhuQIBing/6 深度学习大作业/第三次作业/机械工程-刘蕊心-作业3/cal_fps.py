# import torch
# import time
# from model import InceptionResNet  # 导入你的模型
#
# # ================= 配置 =================
# # 这里的 num_classes 要和你训练时一样
# NUM_CLASSES = 10
# # 测试多少次取平均值
# TEST_TIMES = 500
# # 设备
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # =======================================
#
# def calculate_fps():
# 	print(f"🚀 正在准备环境，使用设备: {DEVICE}")
#
# 	# 1. 加载模型结构
# 	model = InceptionResNet(num_classes=NUM_CLASSES).to(DEVICE)
# 	model.eval()  # 开启验证模式 (不计算梯度，不启用Dropout)
#
# 	# 2. 生成一个虚拟的图片数据 (Batch Size=1, RGB 3通道, 224x224)
# 	# 模拟一张图片
# 	dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
#
# 	print("🔥 正在预热 GPU (Warm up)...")
# 	# 预热：先跑 50 次，让 GPU 显存准备好，这部分时间不计入
# 	with torch.no_grad():
# 		for _ in range(50):
# 			_ = model(dummy_input)
#
# 	print(f"⏱️ 开始测试 FPS (循环 {TEST_TIMES} 次)...")
#
# 	# 3. 正式计时
# 	start_time = time.time()
#
# 	with torch.no_grad():
# 		for _ in range(TEST_TIMES):
# 			_ = model(dummy_input)
#
# 			# 如果是用 GPU，需要加上这行同步代码，确保计算真的完成了
# 			if DEVICE.type == 'cuda':
# 				torch.cuda.synchronize()
#
# 	end_time = time.time()
#
# 	# 4. 计算结果
# 	total_time = end_time - start_time
# 	avg_time_per_image = total_time / TEST_TIMES
# 	fps = 1.0 / avg_time_per_image
#
# 	print("\n" + "=" * 30)
# 	print(f"📊 测试结果:")
# 	print(f"总耗时: {total_time:.4f} 秒")
# 	print(f"单张推理耗时 (Latency): {avg_time_per_image * 1000:.2f} ms")
# 	print(f"每秒帧数 (FPS): {fps:.2f}")
# 	print("=" * 30)
#
# 	return fps
#
#
# if __name__ == '__main__':
# 	fps = calculate_fps()
#
# 	# 这里给你一个简单的评价反馈，方便你写论文用
# 	if fps > 30:
# 		print("\n✅ 导师点评：太棒了！你的 FPS > 30，完全满足实时检测需求！论文里可以大胆吹！")
# 	else:
# 		print("\n⚠️ 导师点评：FPS 稍低（低于30），可能是模型比较大。在论文里可以解释为“牺牲了一定速度换取了高精度”。")


import torch
import time
from model import InceptionResNet, GoogLeNet  # 导入两个模型

# ================= 【🔴 修改这里】 =================
MODEL_TYPE = 'GoogLeNet'  # 'GoogLeNet' 或 'InceptionResNet'
# 这里的路径其实不影响FPS计算（因为FPS只看网络结构，不看权重），但为了严谨可以指向对应的文件
MODEL_PATH = './results/GoogLeNet/best_model.pth'
# ===============================================

NUM_CLASSES = 10
TEST_TIMES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_fps():
	print(f"🚀 正在测试 FPS | 模型: {MODEL_TYPE} | 设备: {DEVICE}")
	
	# 根据配置初始化模型
	if MODEL_TYPE == 'GoogLeNet':
		model = GoogLeNet(num_classes=NUM_CLASSES).to(DEVICE)
	else:
		model = InceptionResNet(num_classes=NUM_CLASSES).to(DEVICE)
	
	model.eval()
	
	# 模拟输入
	dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
	
	print("🔥 正在预热 GPU (Warm up)...")
	with torch.no_grad():
		for _ in range(50):
			_ = model(dummy_input)
	
	print(f"⏱️ 开始测试 (循环 {TEST_TIMES} 次)...")
	start_time = time.time()
	with torch.no_grad():
		for _ in range(TEST_TIMES):
			_ = model(dummy_input)
			if DEVICE.type == 'cuda':
				torch.cuda.synchronize()
	end_time = time.time()
	
	total_time = end_time - start_time
	avg_time = total_time / TEST_TIMES
	fps = 1.0 / avg_time
	
	print("\n" + "=" * 30)
	print(f"📊 模型: {MODEL_TYPE}")
	print(f"单张耗时: {avg_time * 1000:.2f} ms")
	print(f"FPS:      {fps:.2f}")
	print("=" * 30)


if __name__ == '__main__':
	calculate_fps()