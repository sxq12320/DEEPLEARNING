import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import InceptionResNet  # 导入你的模型

# ================= 配置 =================
# 这里的类别必须和你训练时的顺序一模一样！
# 根据你之前发的日志，这是正确的顺序：
CLASSES = [
	'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
	'Septoria_leaf_spot', 'Spider_Mite', 'Target_Spot',
	'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_model.pth'


# =======================================

def get_transform():
	# ⚠️ 必须和训练时一模一样的预处理
	return transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		# 用你算出来的那个均值方差
		transforms.Normalize(mean=[0.4526, 0.4632, 0.4187],
		                     std=[0.1802, 0.1568, 0.1944])
	])


def main():
	print("🚀 正在加载模型...")
	# 1. 加载模型
	model = InceptionResNet(num_classes=10).to(DEVICE)
	if torch.cuda.is_available():
		model.load_state_dict(torch.load(MODEL_PATH))
	else:
		model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
	model.eval()
	print("✅ 模型加载完毕！正在打开摄像头...")
	
	# 2. 打开摄像头 (0 通常是电脑自带，如果有外接USB可能是 1)
	cap = cv2.VideoCapture(0)
	
	# 设置预处理
	transform = get_transform()
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	while True:
		ret, frame = cap.read()
		if not ret:
			print("❌ 无法获取摄像头画面")
			break
		
		# 3. 图像预处理
		# OpenCV 读进来是 BGR，PyTorch 需要 RGB，转换一下
		img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pil_img = Image.fromarray(img_rgb)
		
		# 变成 Tensor 并增加 Batch 维度 [1, 3, 224, 224]
		input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
		
		# 4. 推理
		with torch.no_grad():
			output = model(input_tensor)
			# 用 Softmax 算出概率
			probabilities = torch.nn.functional.softmax(output, dim=1)
			prob, predicted_idx = torch.max(probabilities, 1)
		
		# 获取结果
		class_name = CLASSES[predicted_idx.item()]
		confidence = prob.item()
		
		# 5. 在画面上画字
		# 如果置信度 > 70% 显示绿色，否则显示红色（表示不确定）
		color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
		
		# 显示类别
		text = f"{class_name} ({confidence * 100:.1f}%)"
		cv2.putText(frame, text, (30, 50), font, 1, color, 2, cv2.LINE_AA)
		
		# 显示 FPS 提示
		cv2.putText(frame, "Press 'Q' to Exit", (30, 90), font, 0.7, (255, 255, 255), 1)
		
		# 6. 显示窗口
		cv2.imshow('Tomato Disease Detection (Real-Time)', frame)
		
		# 按 Q 退出
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()