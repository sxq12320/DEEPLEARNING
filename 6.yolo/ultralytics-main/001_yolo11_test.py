from PIL.ImageOps import scale

from ultralytics import YOLO

if __name__ == '__main__':
    # yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg.yaml')
    # yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg-sxq_2.yaml')
    yolo = YOLO(r'E:\mastercode\6.yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg-sxq_3.yaml')
    yolo.train(
        data=r'E:\mastercode\6.yolo\ultralytics-main\201_caomei_data.yaml',
        project=r'E:/mastercode/6.yolo/runs/segment',
        epochs=100,
        imgsz=512,
        batch=8,
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='AdamW',
        device=0,
        copy_paste=0.3,  # 开启并提高复制粘贴的概率
        erasing=0.3,     # 提高随机擦除的概率（模拟大块树叶遮挡）
        mixup=0.1,        # 适度开启图像混合

        #增加色彩增强操作
        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,
        #几何增强
        scale = 0.5,
        degrees = 15,
        flipud = 0.3,
        translate = 0.15,
        mosaic = 0.8,

    )
