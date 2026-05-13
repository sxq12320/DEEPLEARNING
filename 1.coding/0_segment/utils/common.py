import torch.nn as nn


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


def autopad(k, p=None, d=1):
    """计算合适的填充大小，使卷积前后尺寸不变。

    Args:
        k (int | List[int]): 卷积核大小。
        p (int | List[int] | None): 预设填充大小。
        d (int): 空洞率。

    Returns:
        int | List[int]: 计算后的填充大小。
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
