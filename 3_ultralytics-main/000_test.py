from ultralytics import YOLO
# 将模型尝试着看看是否实现了要求，看看效果的样子而已
# 加载模型
model = YOLO(r"E:/mastercode/3_ultralytics-main/results\segment/4_jeurk_test_mini/7-yolo11n_Dense_Skip/weights/best.pt")

# 预测
results = model.predict(
    source="E:/mastercode/data/jeruk_split/yolo/val/images",   # 图片或文件夹
    save=False,              # 保存结果
    conf=0.50,               # 置信度阈值
    project=r"E:/mastercode/3_ultralytics-main/results/segment/4_jeurk_test_mini/7-yolo11n_Dense_Skip/test_pic",   # ← 你想保存的根目录
    name="predict_val",
    batch = 4,
    imgsz = 640,
    stream=True
)

# 手动保存：将每个结果移到 CPU 再绘图
for i, r in enumerate(results):
    r_cpu = r.cpu()                     # 关键：mask 张量移到 CPU
    r_cpu.save(filename=f"result_{i}.jpg")   # 或者 r_cpu.plot() 再保存
# 查看结果
# for r in results:
#     print(r.masks)   # 分割结果