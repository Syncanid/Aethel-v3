import asyncio
import logging
from typing import List, Callable, Awaitable, Set

from core.io.event_schema import OneBotEvent, Action

logger = logging.getLogger(__name__)

# 定义处理器类型
EventHandler = Callable[[OneBotEvent], Awaitable[None]]
ActionHandler = Callable[[Action], Awaitable[None]]


class EventBus:
    def __init__(self):
        self._event_handlers: List[EventHandler] = []
        self._action_handlers: List[ActionHandler] = []
        self._background_tasks: Set[asyncio.Task] = set()

    def subscribe_event(self, handler: EventHandler):
        """订阅输入事件 (Event)"""
        self._event_handlers.append(handler)
        logger.debug(f"Event handler subscribed: {handler.__name__}")

    def subscribe_action(self, handler: ActionHandler):
        """订阅输出动作 (Action)"""
        self._action_handlers.append(handler)
        logger.debug(f"Action handler subscribed: {handler.__name__}")

    def publish_event(self, event: OneBotEvent):
        """发布输入事件"""
        for handler in self._event_handlers:
            self._create_task(handler(event), f"evt-{event.id}")

    def publish_action(self, action: Action):
        """发布输出动作"""
        for handler in self._action_handlers:
            self._create_task(handler(action), f"act-{action.action}")

    def _create_task(self, coro: Awaitable, name: str):
        """创建并管理后台任务"""
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
