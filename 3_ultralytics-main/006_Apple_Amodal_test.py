from ultralytics import YOLO

if __name__ == '__main__':
    # 正常传入你的 4 通道图纸
    yaml_path = 'E:\\mastercode\\3_ultralytics-main\\ultralytics\\cfg\\models\\11\\2_yolo11n-seg-DWCONV.yaml'
    yolo = YOLO(yaml_path)
    
    # 4. 开始从零训练
    yolo.train(
        data='E:/mastercode/3_ultralytics-main/206_Apple_Amodal.yaml',
        project=r'E:\mastercode\3_ultralytics-main\results\Amodal_Segment\Apple',
        name='2_yolo11n-seg-DWCONV',
        epochs=300,
        imgsz=640,
        batch=8,
        
        pretrained=False, # 坚持原则，从零训练
        
        # 严防死守色彩增强
        hsv_h=0.0, 
        hsv_s=0.0, 
        hsv_v=0.0,
        
        workers=4
    )