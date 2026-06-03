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
    """
    Apple Amodal RGBD 分割实验 — 控制理论驱动的双流融合架构

    ═══════════════════════════════════════════════════════════
    数据管道修复说明：
    ═══════════════════════════════════════════════════════════
    1. 深度图加载精度修复 (base.py)：
       - 修复前：cv2.IMREAD_GRAYSCALE → 8位截断，丢失 99.6% 深度信息
       - 修复后：cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE → 保留 16位精度
       - 归一化：uint16 (mm) / 1000.0 → float32 (m)，范围 [0, ~10]

    2. 深度图插值修复 (base.py)：
       - 修复前：cv2.INTER_LINEAR (线性插值，会混合不同深度值)
       - 修复后：cv2.INTER_NEAREST (最近邻插值，保持深度边界清晰)

    ═══════════════════════════════════════════════════════════
    模型架构修复说明：
    ═══════════════════════════════════════════════════════════
    1. 添加 SplitChannels 模块：正确分离 4ch RGBD → RGB(3ch) + Depth(1ch)
    2. YAML 添加 channels: 4：模型初始化时使用 4 通道输入
    3. DetectionModel 修复：优先读取 YAML 中的 channels 值
    4. 融合模块 forward 修复：BypassModule/KalmanGatedFusion/ESOFusion/IDAPBCFusion
       接受列表输入 [tensor_rgb, tensor_depth]，与 _predict_once 传参方式一致
    5. BypassModule 空间对齐：YAML 中 Depth 流使用 Conv stride=2 逐级下采样，
       确保与 RGB 流在 P3/P4/P5 各阶段空间尺寸一致

    ═══════════════════════════════════════════════════════════
    实验配置（按顺序排列，取消注释即可切换）：
    ═══════════════════════════════════════════════════════════

    1. yolo11-base-rgbd.yaml
       对照组：Base 基础非对称双流模型。
       RGB 主干为标准 YOLO11 结构，Depth 流用轻量 Conv 逐级下采样对齐空间尺寸。
       融合方式为最简单的 Bypass 旁路相加，无任何控制理论计算。
       作为所有消融实验的基准对照线。

    2. yolo11-ct-A.yaml
       消融组 A：仅浅层卡尔曼融合。
       仅激活 P3 阶段的 KalmanGatedFusion（卡尔曼自适应滤波融合），
       P4/P5 依然使用无源 Bypass 对照。验证"卡尔曼增益机制"单独的贡献。

    3. yolo11-ct-AB.yaml
       消融组 A+B：卡尔曼 + ESO 扰动补偿。
       同时激活 P3 KalmanGatedFusion 与 P4 ESOFusion（自抗扰观测器融合），
       深层 P5 保持 Bypass 对照。验证"渐进叠加控制"的增益效果。

    4. yolo11-ct-ABC.yaml
       满血组 A+B+C：三阶段渐进控制融合。
       三个控制单元全激活：
         P3 → KalmanGatedFusion（卡尔曼自适应）
         P4 → ESOFusion（ESO扰动补偿）
         P5 → IDAPBCFusion（IDA-PBC能量成型）
       完整的控制理论驱动 RGBD 融合架构。
    """

    # ======================== 1. 对照组：Base 基础双流 ========================
    yolo = YOLO(r"ultralytics/cfg/models/11/yolo11-base-rgbd.yaml")
    yolo.train(
         data=r"ultralytics-main/206_Apple_Amodal.yaml",
         project=r"E:/mastercode/ultralytics-main/results",
         name="01_yolo11n-seg-base-rgbd_fix",
         optimizer="SMC",
         epochs=20,
         patience=50,
         imgsz=540,
         batch=4,
         lr0=0.01,
         workers=4,
         device=0,
         cache=False,  # 缓存预处理图片到磁盘，首次加载慢，后续 epoch 从缓存读取
         seed=SEED,

         # SMC 超参数（可选，不传用默认值）
         smc_plateau_threshold=0.15,   # 梯度停滞阈值
         smc_plateau_patience=5,       # plateau 耐心值
         smc_escape_push=0.10,         # 逃离位移量
         smc_escape_push_steps=20,     # 逃离步数
         smc_reconv_steps=40,          # 重收敛步数
         smc_reconv_lr_mult=3.0,       # 重收敛 LR 倍数
         smc_beta1_low=0.1,            # 逃离时 β₁
         smc_beta2_low=0.9,            # 逃离时 β₂

    )
   

    # ======================== 2. 消融组 A：卡尔曼融合 ========================
    # yolo = YOLO(r"ultralytics/cfg/models/11/yolo11-ct-A.yaml")
    # yolo.train(
    #     data=r"ultralytics-main/206_Apple_Amodal.yaml",
    #     project=r"ultralytics-main/results",
    #     name="02_yolo11n-seg-ct-A-kalman",
    #     epochs=300,
    #     imgsz=640,
    #     batch=4,
    #     lr0=0.0001,
    #     workers=4,
    #     device=0,
    #     seed=SEED,
    # )

    # ======================== 3. 消融组 A+B：卡尔曼+ESO ========================
    # yolo = YOLO(r"ultralytics/cfg/models/11/yolo11-ct-AB.yaml")
    # yolo.train(
    #     data=r"ultralytics-main/206_Apple_Amodal.yaml",
    #     project=r"ultralytics-main/results",
    #     name="03_yolo11n-seg-ct-AB-kalman-eso",
    #     epochs=300,
    #     imgsz=640,
    #     batch=4,
    #     lr0=0.0001,
    #     workers=4,
    #     device=0,
    #     seed=SEED,
    # )

    # ======================== 4. 满血组 A+B+C：三阶段控制 ========================
    # yolo = YOLO(r"ultralytics/cfg/models/11/yolo11-ct-ABC.yaml")
    # yolo.train(
    #     data=r"ultralytics-main/206_Apple_Amodal.yaml",
    #     project=r"ultralytics-main/results",
    #     name="04_yolo11n-seg-ct-ABC-full",
    #     epochs=300,
    #     imgsz=640,
    #     batch=4,
    #     lr0=0.0001,
    #     workers=4,
    #     device=0,
    #     seed=SEED,
    # )
