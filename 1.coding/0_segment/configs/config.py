import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_MAP = {
    "relu": nn.ReLU(inplace=True),
    "leakyrelu": nn.LeakyReLU(negative_slope=0.01, inplace=True),
    "prelu": nn.PReLU(),
    "relu6": nn.ReLU6(inplace=True),
    "silu": nn.SiLU(inplace=True),
    "gelu": nn.GELU(),
    "elu": nn.ELU(inplace=True),
    "selu": nn.SELU(inplace=True),
    "mish": nn.Mish(inplace=True),
    "hardswish": nn.Hardswish(inplace=True),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity(),
    "none": nn.Identity(),
}

# resnet18的主干网络生成
RESNET_18_CFG = [
    ["basic_conv_block", 3, 64, 7, 2, 3, 1, 1, "relu"],
    ["maxpool", 3, 2, 1, 1],
    # Stage 1: 两个 block，尺寸和通道均不变
    ["resnet_block_34", 64, 64, 1, "relu", "relu"],
    ["resnet_block_34", 64, 64, 1, "relu", "relu"],
    # Stage 2: 第一个 block 下采样 + 升通道，第二个保持
    ["resnet_block_34", 64, 128, 2, "relu", "relu"],
    ["resnet_block_34", 128, 128, 1, "relu", "relu"],
    # Stage 3
    ["resnet_block_34", 128, 256, 2, "relu", "relu"],
    ["resnet_block_34", 256, 256, 1, "relu", "relu"],
    # Stage 4
    ["resnet_block_34", 256, 512, 2, "relu", "relu"],
    ["resnet_block_34", 512, 512, 1, "relu", "relu"],
    ["adaptive_avg_pool", (1, 1)],
    ["flatten"],
    ["linear", 512, 1000, True],
]

# ResNet-18 骨干网络配置（不含分类头，用于分割等密集预测任务）
RESNET_18_BACKBONE_CFG = [
    ["basic_conv_block", 3, 64, 7, 2, 3, 1, 1, "relu"],
    ["maxpool", 3, 2, 1, 1],
    ["resnet_block_34", 64, 64, 1, "relu", "relu"],
    ["resnet_block_34", 64, 64, 1, "relu", "relu"],
    ["resnet_block_34", 64, 128, 2, "relu", "relu"],
    ["resnet_block_34", 128, 128, 1, "relu", "relu"],
    ["resnet_block_34", 128, 256, 2, "relu", "relu"],
    ["resnet_block_34", 256, 256, 1, "relu", "relu"],
    ["resnet_block_34", 256, 512, 2, "relu", "relu"],
    ["resnet_block_34", 512, 512, 1, "relu", "relu"],
]

RESNET_34_CFG = [
    ["basic_conv_block", 3, 64, 7, 2, 3, 1, 1, "relu"],
]

# ---------------------------------------------------------------------------
# ResNet-18 分阶段配置（用于多尺度特征提取）
# Splits RESNET_18_BACKBONE_CFG into stem + 4 stages，
# 每阶段输出作为 FPN 的一级输入
# 通道变化：3 → 64(1/4) → 64(1/4) → 128(1/8) → 256(1/16) → 512(1/32)
# ---------------------------------------------------------------------------
RESNET_18_STEM_CFG = [
    ["basic_conv_block", 3, 64, 7, 2, 3, 1, 1, "relu"],
    ["maxpool", 3, 2, 1, 1],
]
RESNET_18_STAGE1_CFG = [
    ["resnet_block_34", 64, 64, 1, "relu", "relu"],
    ["resnet_block_34", 64, 64, 1, "relu", "relu"],
]
RESNET_18_STAGE2_CFG = [
    ["resnet_block_34", 64, 128, 2, "relu", "relu"],
    ["resnet_block_34", 128, 128, 1, "relu", "relu"],
]
RESNET_18_STAGE3_CFG = [
    ["resnet_block_34", 128, 256, 2, "relu", "relu"],
    ["resnet_block_34", 256, 256, 1, "relu", "relu"],
]
RESNET_18_STAGE4_CFG = [
    ["resnet_block_34", 256, 512, 2, "relu", "relu"],
    ["resnet_block_34", 512, 512, 1, "relu", "relu"],
]


# ---------------------------------------------------------------------------
# YOLO11 模型缩放配置
# 宽度：[stem1, stem2, P3, P4, P5] 各阶段输出通道
# 深度缩放因子：控制 C3K2 内部 bottleneck 重复次数
# ---------------------------------------------------------------------------
YOLO11_CONFIGS = {
    # YOLO11 Nano：极小模型，适合移动端 / CPU 推理
    "nano": {
        "channels": [16, 32, 64, 128, 256],
        "depth_scale": 0.33,
        "description": "YOLO11 Nano — 移动端 / CPU 友好",
    },
    # YOLO11 Small：速度与精度的平衡点
    "small": {
        "channels": [32, 64, 128, 256, 512],
        "depth_scale": 0.67,
        "description": "YOLO11 Small — 速度/精度平衡",
    },
    # YOLO11 Medium：常规 GPU 训练首选
    "medium": {
        "channels": [64, 128, 256, 512, 512],
        "depth_scale": 1.0,
        "description": "YOLO11 Medium — 常规精度取向",
    },
}


# ---------------------------------------------------------------------------
# TS-Dual 架构默认配置
# ---------------------------------------------------------------------------
TS_DUAL_MODEL_CFG = {
    "backbone": {
        "name": "ts_dual_backbone",
        "args": {
            "in_ch_rgb": 4,
            "in_ch_depth": 1,
            "channels": [32, 64, 128],
            "activation": "silu",
            "exchange_reduction": 4,
        },
    },
    "neck": [
        {
            "name": "afpn_neck",
            "args": {
                "in_channels": [32, 64, 128],
                "out_channels": 128,
                "activation": "silu",
            },
        },
        {
            "name": "dyhead_neck",
            "args": {
                "channels": 128,
                "reduction": 4,
                "activation": "silu",
            },
        },
    ],
    "head": {
        "name": "decoupled_segdet_head",
        "args": {
            "in_channels": 128,
            "mask_out_ch": 1,
            "activation": "silu",
            "bbox_hidden": 128,
        },
    },
}

TS_DUAL_LOSS_CFG = {
    "mask_loss": "bce",
    "mask_weight": 1.0,
    "fourier_weight": 0.2,
    "bbox_weight": 1.0,
    "nwd_constant": 20.0,
}
