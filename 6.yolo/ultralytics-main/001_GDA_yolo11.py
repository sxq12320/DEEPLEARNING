from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\26\yolo26n-seg.yaml')
    yolo.train(
        data=r'E:\mastercode\6.yolo\ultralytics-main\201_caomei_data.yaml',
        project=r'E:/mastercode/6.yolo/runs/segment',
        epochs=150,
        imgsz=640,
        batch=16,
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='AdamW',
        device=0,
        copy_paste=0.3,  # 开启并提高复制粘贴的概率
        erasing=0.6,     # 提高随机擦除的概率（模拟大块树叶遮挡）
        mixup=0.1        # 适度开启图像混合
    )
