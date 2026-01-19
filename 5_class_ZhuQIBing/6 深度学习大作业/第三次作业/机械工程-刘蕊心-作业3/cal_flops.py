# import torch
# from thop import profile
# from model import InceptionResNet
#
# # ================= 配置 =================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # =======================================
#
# def cal_params_flops():
# 	print("正在计算模型复杂度...")
# 	model = InceptionResNet(num_classes=10).to(DEVICE)
#
# 	# 创建一个虚拟输入 (1张图, 3通道, 224x224)
# 	input = torch.randn(1, 3, 224, 224).to(DEVICE)
#
# 	# 使用 thop 库自动计算
# 	flops, params = profile(model, inputs=(input,))
#
# 	print("\n" + "=" * 30)
# 	print(f"参数量 (Params): {params / 1e6:.2f} M (百万)")
# 	print(f"计算量 (FLOPs) : {flops / 1e9:.2f} G (十亿)")
# 	print("=" * 30)
#
#
# if __name__ == '__main__':
# 	cal_params_flops()


import torch
from thop import profile
from model import InceptionResNet, GoogLeNet

# ================= 【🔴 修改这里】 =================
MODEL_TYPE = 'GoogLeNet'  # 'GoogLeNet' 或 'InceptionResNet'
# ===============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_params_flops():
	print(f"正在计算复杂度 | 模型: {MODEL_TYPE}")
	
	if MODEL_TYPE == 'GoogLeNet':
		model = GoogLeNet(num_classes=10).to(DEVICE)
	else:
		model = InceptionResNet(num_classes=10).to(DEVICE)
	
	input = torch.randn(1, 3, 224, 224).to(DEVICE)
	flops, params = profile(model, inputs=(input,))
	
	print("\n" + "=" * 30)
	print(f"模型: {MODEL_TYPE}")
	print(f"参数量 (Params): {params / 1e6:.2f} M")
	print(f"计算量 (FLOPs) : {flops / 1e9:.2f} G")
	print("=" * 30)


if __name__ == '__main__':
	cal_params_flops()