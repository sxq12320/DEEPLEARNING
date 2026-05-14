"""模型注册表。

包含基础模块、backbone、neck、head 的统一注册入口。
"""

from typing import Callable, Dict, Type

BLOCK_REGISTRY: Dict[str, Type] = {}
BACKBONE_REGISTRY: Dict[str, Type] = {}
NECK_REGISTRY: Dict[str, Type] = {}
HEAD_REGISTRY: Dict[str, Type] = {}


def register_block(name):
    """装饰器工厂：将构建函数注册到 BLOCK_REGISTRY。

    Args:
        name (str): 注册名称，内部会转为小写。

    Returns:
        Callable: 装饰器函数。
    """

    def decorator(func):
        """将构建函数写入注册表。

        Args:
            func (Callable): 待注册的构建函数。

        Returns:
            Callable: 原函数本身。
        """
        BLOCK_REGISTRY[name.lower()] = func
        return func

    return decorator


def register_backbone(name: str) -> Callable:
    """注册 backbone 类。

    Args:
        name (str): 注册名称，不区分大小写。

    Returns:
        Callable: 装饰器函数。
    """

    def decorator(cls):
        """将 backbone 类写入注册表。

        Args:
            cls (Type): 待注册的 backbone 类。

        Returns:
            Type: 原类本身。
        """
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
        """将 neck 类写入注册表。

        Args:
            cls (Type): 待注册的 neck 类。

        Returns:
            Type: 原类本身。
        """
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
        """将 head 类写入注册表。

        Args:
            cls (Type): 待注册的 head 类。

        Returns:
            Type: 原类本身。
        """
        HEAD_REGISTRY[name.lower()] = cls
        return cls

    return decorator
