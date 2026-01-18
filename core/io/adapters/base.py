# core/io/adapters/base.py
from abc import ABC, abstractmethod
from typing import Optional

from core.io.event_bus import EventBus
from core.io.event_schema import Action, ActionResponse


class BaseAdapter(ABC):
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe_action(self.handle_action)

    @abstractmethod
    async def run(self):
        """启动适配器的监听循环"""
        pass

    @abstractmethod
    async def handle_action(self, action: Action) -> Optional[ActionResponse]:
        """
        处理系统发出的动作
        Return:
            - ActionResponse: 处理成功或失败的结果
            - None: 此适配器不处理该 Action (忽略)
        """
        pass

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """返回平台名称"""
        pass
