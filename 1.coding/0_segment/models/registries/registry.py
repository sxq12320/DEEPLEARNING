BLOCK_REGISTRY = {}

def register_block(name):
    """装饰器：将构建函数注册到 BLOCK_REGISTRY"""
    def decorator(func):
        '''
        实际执行注册的装饰器函数。

        Args:
            func (callable): 待注册的模块构建函数或类。

        Returns:
            callable: 原始函数对象, 以保持其可调用属性不变。
        '''
        BLOCK_REGISTRY[name.lower()] = func
        return func
    return decorator