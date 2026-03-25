from ultralytics import YOLO
# 将模型尝试着看看是否实现了要求，看看效果的样子而已
# 加载模型
model = YOLO(r"E:\mastercode\3_ultralytics-main\results\segment\1_caomei_test\train4\weights\best.pt")

# 预测
results = model.predict(
    source="E:/mastercode/data/caomei/final/images/val",   # 图片或文件夹
    save=True,              # 保存结果
    conf=0.50,               # 置信度阈值
    project=r"E:\mastercode\3_ultralytics-main\results\segment\1_caomei_test\train4",   # ← 你想保存的根目录
    name="predict_val"
)

# 查看结果
# for r in results:
#     print(r.masks)   # 分割结果