from ultralytics import YOLO

if __name__ == '__main__':
    # yolo = YOLO(r"E:\mastercode\3_ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg.yaml") # 原始的yolo11n架构，通道数量不变
    yolo = YOLO(r"E:\mastercode\3_ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg.yaml") # 将原始的通道数量降低一半
    yolo.train(
        data=r'E:\mastercode\3_ultralytics-main\205_jeurk_spilt_data.yaml',
        project=r'E:\mastercode\3_ultralytics-main\results\segment\4_jeurk_test_mini',
        name='1-yolo11n_HalfChannel',
        epochs=200,
        imgsz=640,
        batch=8,
        lr0=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        optimizer='SGD',
        amp = False,
        cache=True
    )