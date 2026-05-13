BLOCK_REGISTRY = {}


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
