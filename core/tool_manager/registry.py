from typing import Callable, List, Tuple

_PENDING_FUNCTIONS: List[Tuple[Callable, str]] = []


def register(name: str = None):
    """
    具注册装饰器
    """

    def decorator(func: Callable):
        _PENDING_FUNCTIONS.append((func, name))
        return func

    return decorator


def get_pending_functions() -> List[Tuple[Callable, str]]:
    """获取所有已注册但未加载的函数"""
    return _PENDING_FUNCTIONS.copy()


def clear_pending():
    _PENDING_FUNCTIONS.clear()
