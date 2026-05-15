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


def get_activation(act_name: str, activation_map: dict):
    """根据名称从映射表中获取激活函数模块。

    Args:
        act_name (str): 激活函数名称。
        activation_map (dict): 名称到模块的映射表。

    Returns:
        torch.nn.Module: 激活函数模块。

    Raises:
        ValueError: 当激活函数名称不支持时抛出。
    """
    act_name = act_name.strip().lower()
    if act_name not in activation_map:
        supported = ",".join(sorted(activation_map.keys()))
        raise ValueError(
            f"Unsupported activation: {act_name}. Supported activations: {supported}"
        )
    return activation_map[act_name]
