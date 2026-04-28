import torch.nn as nn


def get_activation(act_name: str, activation_map: dict):
    """根据名称从映射表中获取激活函数模块。"""
    act_name = act_name.strip().lower()
    if act_name not in activation_map:
        supported = ",".join(sorted(activation_map.keys()))
        raise ValueError(f"Unsupported activation: {act_name}. Supported activations: {supported}")
    return activation_map[act_name]


def autopad(k, p=None, d=1):
    """返回适当的填充大小，使卷积前后尺寸不变。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
