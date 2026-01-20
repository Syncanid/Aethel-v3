# core/io/event_bus.py
import asyncio
import logging
from typing import List, Callable, Awaitable, Set, Optional

from core.io.event_schema import OneBotEvent, Action, ActionResponse, ActionStatus

logger = logging.getLogger(__name__)

# 定义处理器类型
EventHandler = Callable[[OneBotEvent], Awaitable[None]]
# Action 处理器现在必须返回 Optional[ActionResponse]
ActionHandler = Callable[[Action], Awaitable[Optional[ActionResponse]]]


class EventBus:
    def __init__(self):
        self._event_subscribers: List[EventHandler] = []
        self._action_subscribers: List[ActionHandler] = []
        self._background_tasks: Set[asyncio.Task] = set()

    def subscribe_event(self, handler: EventHandler):
        """订阅外部事件 (Incoming)"""
        self._event_subscribers.append(handler)
        logger.debug(f"Event handler subscribed: {handler.__name__}")

    def subscribe_action(self, handler: ActionHandler):
        """订阅系统动作 (Outgoing)"""
        self._action_subscribers.append(handler)
        logger.debug(f"Action handler subscribed: {handler.__name__}")

    def publish_event(self, event: OneBotEvent):
        """
        发布事件 (异步，Fire-and-Forget)
        安全地创建后台任务，并防止 GC。
        """
        for handler in self._event_subscribers:
            # 创建任务
            task = asyncio.create_task(self._safe_execute_event(handler, event))

            # 1. 加入集合建立强引用
            self._background_tasks.add(task)

            # 2. 任务完成后自动从集合移除
            task.add_done_callback(self._background_tasks.discard)

    def publish_action(self, action: Action):
        """发布动作 (兼容旧版，异步不等待)"""
        for handler in self._action_subscribers:
            task = asyncio.create_task(self._safe_execute_action(handler, action))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def dispatch_action(self, action: Action, timeout: float = 20.0) -> ActionResponse:
        """
        [Request-Response] 分发动作并等待结果
        """
        if not self._action_subscribers:
            return ActionResponse(status=ActionStatus.FAILED, message="No action handlers registered")

        # 创建所有处理器的任务
        tasks = [handler(action) for handler in self._action_subscribers]

        try:
            # 等待所有处理器完成 (return_exceptions=True 防止单个崩溃影响整体)
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)

            final_response = None

            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"Action handler failed: {res}")
                    if not final_response:
                        final_response = ActionResponse(status=ActionStatus.FAILED,
                                                        message=f"Handler error: {str(res)}")

                elif isinstance(res, ActionResponse):
                    # 如果处理器返回了有效响应
                    if res.status == ActionStatus.OK:
                        return res  # 只要有一个成功，立即返回

                    # 记录失败响应，继续检查其他处理器是否成功
                    final_response = res

                # 如果返回 None，说明该处理器忽略了此 Action (路由不匹配)，继续检查

            # 如果遍历完没有成功，返回最后的失败或默认失败
            return final_response or ActionResponse(
                status=ActionStatus.FAILED,
                message="No handler processed the action (Route not matched)"
            )

        except asyncio.TimeoutError:
            logger.error(f"Action dispatch timed out: {action.action}")
            return ActionResponse(status=ActionStatus.FAILED, message="Action execution timed out")
        except Exception as e:
            logger.error(f"EventBus dispatch error: {e}", exc_info=True)
            return ActionResponse(status=ActionStatus.FAILED, message=f"Bus error: {str(e)}")

    async def _safe_execute_event(self, handler, event):
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Event handler error: {e}", exc_info=True)

    async def _safe_execute_action(self, handler, action):
        try:
            await handler(action)
        except Exception as e:
            logger.error(f"Action handler error: {e}", exc_info=True)

    async def shutdown(self):
        """等待所有后台任务完成 (用于系统关闭)"""
        if self._background_tasks:
            logger.info(f"等待 {len(self._background_tasks)} 个后台事件任务完成...")
            # 取消所有任务或等待它们完成，这里选择取消以加快退出
            for task in self._background_tasks:
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
