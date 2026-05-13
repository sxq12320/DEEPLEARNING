import torch.nn as nn

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
