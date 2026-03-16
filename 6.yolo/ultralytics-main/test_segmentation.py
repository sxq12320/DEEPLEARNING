from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26n-seg.pt')  # 可换 s/m/l/x
    model.train(
        data='ultralytics-main/voc20007_seg_yolov8.yaml',
        project=r'E:/mastercode/6.yolo/runs/segment',
        epochs=300,
        imgsz=640,
        batch=8,
        # lr0 = 0.00001,
        # lrf = 0.01,
        # momentum = 0.73,
        # weight_decay = 0.0005,
        # warmup_epochs = 3,
        # optimizer='AdamW',
        device=0  # CPU 则改为 'cpu'
    )