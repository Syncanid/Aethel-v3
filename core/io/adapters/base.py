from abc import ABC, abstractmethod

from core.io.event_bus import EventBus
from core.io.event_schema import Action


class BaseAdapter(ABC):
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe_action(self.handle_action)

    @abstractmethod
    async def run(self):
        """启动适配器的监听循环"""
        pass

    @abstractmethod
    async def handle_action(self, action: Action):
        """处理系统发出的动作 (如 send_message)"""
        pass

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """返回平台名称"""
        pass
