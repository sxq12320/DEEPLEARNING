from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO(r'E:\mastercode\3_ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg.yaml')
    yolo.train(
        data=r'E:\mastercode\3_ultralytics-main\205_jeurk_spilt_data.yaml',
        project=r'E:\mastercode\3_ultralytics-main\results\segment\4_jeurk_test_mini',
        name='yolo11_origin',
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
