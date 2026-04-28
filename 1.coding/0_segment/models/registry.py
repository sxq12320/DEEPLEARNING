BLOCK_REGISTRY = {}

def register_block(name):
    """装饰器：将构建函数注册到 BLOCK_REGISTRY"""
    def decorator(func):
        BLOCK_REGISTRY[name.lower()] = func
        return func
    return decorator
