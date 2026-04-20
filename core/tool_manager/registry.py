# core/tool_manager/registry.py
from typing import Callable, List, Tuple, Dict

_REGISTERED_FUNCTIONS: Dict[str, Tuple[Callable, str]] = {}


def register(name: str = None):
    """
    工具注册装饰器
    """

    def decorator(func: Callable):
        mod = getattr(func, "__module__", "unknown")
        key = f"{mod}.{func.__name__}"
        _REGISTERED_FUNCTIONS[key] = (func, name)
        return func

    return decorator


def get_pending_functions() -> List[Tuple[Callable, str]]:
    """获取所有已注册的函数"""
    return list(_REGISTERED_FUNCTIONS.values())


def clear_pending():
    # 废弃清空操作，让字典始终保持全量状态
    pass
