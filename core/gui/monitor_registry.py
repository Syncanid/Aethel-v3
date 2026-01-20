# core/gui/monitor_registry.py
from typing import Callable, Dict, Any, List


class MonitorRegistry:
    """
    监控数据注册中心
    允许系统的各个组件注册回调函数，GUI 会定期调用这些回调获取最新数据。
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitorRegistry, cls).__new__(cls)
            cls._instance.data_sources = {}
            cls._instance.plot_sources = {}
        return cls._instance

    def register_text_source(self, category: str, label: str, callback: Callable[[], Any]):
        """
        注册文本/JSON 类型的数据源 (例如 Scratchpad, Last Prompt)
        :param category: 分类 (Tab 名称)
        :param label: 标签名
        :param callback: 返回数据的函数 (通常是 lambda)
        """
        if category not in self.data_sources:
            self.data_sources[category] = []
        self.data_sources[category].append((label, callback))

    def register_metric_source(self, label: str, callback: Callable[[], float], min_val=0.0, max_val=1.0):
        """
        注册数值类型的数据源
        用于显示进度条或仪表盘
        """
        self.plot_sources[label] = {
            "callback": callback,
            "min": min_val,
            "max": max_val
        }

    def get_text_data(self) -> Dict[str, List[tuple]]:
        return self.data_sources

    def get_metric_data(self) -> Dict[str, Any]:
        result = {}
        for label, meta in self.plot_sources.items():
            try:
                val = meta["callback"]()
                result[label] = {
                    "value": val,
                    "min": meta["min"],
                    "max": meta["max"]
                }
            except:
                result[label] = {"value": 0, "min": 0, "max": 1}
        return result


# 全局单例
monitor_registry = MonitorRegistry()
