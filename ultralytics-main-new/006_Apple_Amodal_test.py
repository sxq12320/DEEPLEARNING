from ultralytics import YOLO
import torch
import random
import numpy as np
from ultralytics import YOLO
from thop import profile

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    yolo = YOLO(r"E:\mastercode\ultralytics-main-new\ultralytics\cfg\models\11\yolo11-seg.yaml")
    yolo.train(
         data=r"E:\mastercode\ultralytics-main-new\206_Apple_Amodal.yaml",
         project=r"E:\mastercode\ultralytics-main-new\results",
         name="01_yolo11n-seg-base-rgbd_fix",
         optimizer="PIDAO",
         epochs=20,
         patience=50,
         imgsz=400,
         batch=4,
         lr0=0.01,
         workers=4,
         device=0,
         cache=False,  
         seed=SEED,
         amp = 0,
      )