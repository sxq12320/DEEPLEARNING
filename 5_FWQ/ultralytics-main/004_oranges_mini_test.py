from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO(r'E:\mastercode\3_ultralytics-main\ultralytics\cfg\models\11\yolo11-seg-sxq-1-oranges-2.yaml')
    yolo.train(
        data=r'E:\mastercode\3_ultralytics-main\204_oranges_mini.yaml',
        project=r'E:\mastercode\3_ultralytics-main\results\segment\3_oranges_test_mini',
        name='yolo11_P2_add',
        epochs=200,
        imgsz=640,
        batch=4,
        lr0=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='AdamW',
        amp = False,
        cache=True
    )
