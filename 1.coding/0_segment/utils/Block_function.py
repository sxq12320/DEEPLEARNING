import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act_name:str , activation_map:dict):
    '''
    根据名称从映射表中获取激活函数模块。

    Args:
        act_name (str): 激活函数名称,函数内部会转为小写并去除首尾空格。
        activation_map (dict): 激活函数映射表,键为名称,值为激活模块实例。

    Returns:
        nn.Module: 对应的激活函数模块实例。

    Raises:
        ValueError: act_name 不在 activation_map 中时抛出。
    '''
    act_name = act_name.strip().lower()
    if act_name not in activation_map:
        supported = ",".join(sorted(activation_map.keys()))
        raise ValueError(f"Unsupported activation: {act_name}. Supported activations: {supported}")
    return activation_map[act_name]


def autopad(k, p=None, d=1):
    """
    返回p使得在当前的条件之下让卷积前后的图片尺寸不发生任何的变化

    Args:
        k (int): 卷积核大小。
        p (int, optional): 填充大小。默认为 None。
        d (int): 膨胀率。默认为 1。

    Returns:
        p (int): 适当的填充大小。
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p