from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO(r'E:\mastercode\3_ultralytics-main\ultralytics\cfg\models\11\yolo11n-seg-sxq_2.yaml')
    yolo.train(
        data=r'E:\mastercode\3_ultralytics-main\203_kvasir_data.yaml',
        project=r'E:\mastercode\3_ultralytics-main\results\segment\1_caomei_test',
        name='yolo11_improve_2',
        epochs=200,
        imgsz=640,
        batch=8,
        lr0=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        optimizer='AdamW',
        amp = False,
        cache = True
    )
