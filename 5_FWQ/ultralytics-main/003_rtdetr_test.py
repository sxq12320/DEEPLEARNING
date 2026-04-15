from ultralytics import RTDETR

if __name__ == '__main__':
    # 推理（自动下载预训练权重）
    model = RTDETR(r"E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml")   # large 版本

    model.train(
        data=r'E:\mastercode\6.yolo\ultralytics-main\202_kouzhao_data.yaml',
        epochs=100,
        imgsz=640,
        batch=2,          # 4GB显存就用2，8GB可以用4
        device=0,
        amp=True,         # 开启混合精度，省显存+提速约30%
        cache=True,       # 数据缓存到内存，加快训练
    )