from ultralytics import YOLO

# 1. 加载你训练好的最佳权重文件
model = YOLO('E:/mastercode/3_ultralytics-main/results/Amodal_Segment/Apple/yolo11_origin2/weights/best.pt')  # 请替换为你的 best.pt 实际路径

# 2. 对图片进行推理，并设置 save=True
results = model.predict(source='E:/mastercode/data/Apple_RGB_D_Amoal/yolo/test/images/_MG_2652_23.png', save=True, show=False)

print("推理完成，带有 Mask 叠加的图片已默认保存在 runs/segment/predict 目录下")