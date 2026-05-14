"""主干、颈部、头部模块注册表。

提供统一的注册入口，便于通过配置构建网络。
"""

from typing import Callable, Dict, Type

BACKBONE_REGISTRY: Dict[str, Type] = {}
NECK_REGISTRY: Dict[str, Type] = {}
HEAD_REGISTRY: Dict[str, Type] = {}


def register_backbone(name: str) -> Callable:
    """注册 backbone 类。

    Args:
        name (str): 注册名称，不区分大小写。

    Returns:
        Callable: 装饰器函数。
    """

    def decorator(cls):
        BACKBONE_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def register_neck(name: str) -> Callable:
    """注册 neck 类。

    Args:
        name (str): 注册名称，不区分大小写。

    Returns:
        Callable: 装饰器函数。
    """

    def decorator(cls):
        NECK_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def register_head(name: str) -> Callable:
    """注册 head 类。

    Args:
        name (str): 注册名称，不区分大小写。

    Returns:
        Callable: 装饰器函数。
    """

    def decorator(cls):
        HEAD_REGISTRY[name.lower()] = cls
        return cls

    return decorator
