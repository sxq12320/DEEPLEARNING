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
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation1_4ch_input.yaml")
        # 01 基线模型使用四通道RGBD输入即可，优化器AdamW
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation2_rgb_conv_depth.yaml") 
        # 02 双分支主干网络，RGB使用yolo11主干结构，Depth仅仅使用Conv进行下采样操作，优化器AdamW
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation3_rgb_shufflenet_depth.yaml") 
        # 03 双分支主干网络，RGB使用yolo11主干结构，Depth使用shufflenet结构，优化器AdamW
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation4_optimizers.yaml") 
        # 04 双分支主干网络结构，RGB使用yolo11主干结构，Depth使用shufflenet结构，融合使用bypass，优化器SMC
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation4_optimizers.yaml") 
        # 05 双分支主干网络结构，RGB使用yolo11主干结构，Depth使用shufflenet结构，融合使用bypass，优化器PIDAO
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation4_optimizers.yaml") 
        # 06 双分支主干网络结构，RGB使用yolo11主干结构，Depth使用shufflenet结构，融合使用bypass，优化器AdamW
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation5_ct_fusion.yaml") 
        # 07 双分支主干网络结构，RGB使用yolo11主干结构，Depth使用shufflenet结构，融合使用CTModulesV1，优化器AdamW
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation5_ct_fusion.yaml") 
        # 09 双分支主干网络结构，RGB使用yolo11主干结构，Depth使用shufflenet结构，融合使用CTModulesV1，优化器SMC
    # yolo = YOLO(r"/data/sxq/code/ultralytics-main-new/mine_yaml/ablation5_ct_fusion.yaml") 
        # 010 双分支主干网络结构，RGB使用yolo11主干结构，Depth使用shufflenet结构，融合使用CTModulesV1，优化器PIDAO

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