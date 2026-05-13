"""网络构建及映射模块。

提供依据配置参数实例化网路层次序列的基础方法。
"""

import torch.nn as nn
from .registry import BLOCK_REGISTRY


def make_layers(cfg):
    """根据配置列表构建网络层序列。

    Args:
        cfg (list): 网络结构配置列表，每个元素描述一个模块及其参数。

    Returns:
        nn.Sequential: 按配置构建得到的层序列。

    Raises:
        KeyError: 当 block 类型未注册时抛出。
    """
    layers = []
    for item in cfg:
        block_type = item[0]
        builder = BLOCK_REGISTRY[block_type]
        layers.append(builder(*item[1:]))
    return nn.Sequential(*layers)
