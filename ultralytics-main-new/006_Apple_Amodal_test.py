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
    yolo = YOLO(r"E:\mastercode\ultralytics-main-new\mine_yaml\ablation1_4ch_input.yaml")
    yolo.train(
         data=r"E:/mastercode/ultralytics-main-new/206_Apple_Amodal.yaml",
         project=r"E:/mastercode/ultralytics-main-new/results",
         name="01_yolo11_doublebarnch_RGBD_shuffleNet_depth_AdamW",
         optimizer="AdamW",
         epochs=300,
         patience=50,
         imgsz=640,
         batch=2,
         lr0=0.01,
         workers=4,
         device=0,
         cache=False,  
         seed=SEED,
         amp = 0,
         dropout = 0.1,
      )