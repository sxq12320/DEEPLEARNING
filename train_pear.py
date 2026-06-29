from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='E:/mastercode/data/pear_data2/data_detect.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    close_mosaic=0,       # 关闭Mosaic增强，解决坐标超界
    augment=True,         # 保留其他增强
    mosaic=0,             # 确保Mosaic关闭
    degrees=10,           # 允许小角度旋转
    flipud=0.5,           # 上下翻转
    fliplr=0.5,           # 左右翻转
    hsv_h=0.015,          # 色调变化
    hsv_s=0.7,            # 饱和度变化
    hsv_v=0.4,            # 亮度变化
    project='E:/mastercode/pear_runs',
    name='yolov8n_baseline'
)

print("训练完成！")
