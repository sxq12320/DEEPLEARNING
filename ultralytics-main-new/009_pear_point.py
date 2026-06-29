import torch
import random
import numpy as np
from ultralytics import YOLO

if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # 检测模型
    # yolo = YOLO(r'E:\mastercode\ultralytics-main-new\ultralytics\cfg\models\11\yolo11n.yaml')
    # yolo.train(
    #         data=r"E:/mastercode/data/pear_data2/data_detect.yaml",
    #         project=r"E:/mastercode/ultralytics-main-new/results",
    #         mosaic=0,
    #         close_mosaic=1,
    #         name="pear_1",
    #         optimizer="AdamW",
    #         epochs=300,
    #         patience=300,
    #         imgsz=640,
    #         batch=8,
    #         lr0=0.001,
    #         workers=4,
    #         device=0,
    #         seed=SEED,
    #         amp=1,
    #         dropout=0.1,
    #     )
    # 关键点模型
    # model.train(data='E:/mastercode/data/pear_data2/data.yaml', epochs=100)
    # 关键点模型
    yolo = YOLO(r'E:\mastercode\ultralytics-main-new\ultralytics\cfg\models\11\yolo11n-pose.yaml')
    yolo.train(
            data=r"E:/mastercode/data/pear_data2/data_pose.yaml",
            project=r"E:/mastercode/ultralytics-main-new/results",
            mosaic=0,
            close_mosaic=1,
            name="pear_pose",
            optimizer="AdamW",
            epochs=300,
            patience=300,
            imgsz=640,
            batch=8,
            lr0=0.001,
            workers=4,
            device=0,
            seed=SEED,
            amp=1,
            dropout=0.1,
        )