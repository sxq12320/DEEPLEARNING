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
    ["linear", 512, 1000, True]
]

