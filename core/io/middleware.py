import logging
from typing import Callable, Awaitable, List

from core.io.event_schema import OneBotEvent

logger = logging.getLogger(__name__)

# 中间件函数签名：(event, next_call) -> None
# next_call 是一个无参的 awaitable，调用它表示放行
MiddlewareFunc = Callable[[OneBotEvent, Callable[[], Awaitable[None]]], Awaitable[None]]


class MiddlewareManager:
    def __init__(self):
        self._middlewares: List[MiddlewareFunc] = []

    def register(self, middleware: MiddlewareFunc):
        """注册中间件 (顺序敏感)"""
        self._middlewares.append(middleware)
        logger.info(f"中间件已注册: {middleware.__name__}")

    async def process_event(self, event: OneBotEvent, final_handler: Callable[[OneBotEvent], Awaitable[None]]):
        """
        执行中间件链
        :param event: 待处理事件
        :param final_handler: 链条末端的核心处理函数 (通常是 Agent 的处理逻辑)
        """
        index = 0

        async def next_step():
            nonlocal index
            if index < len(self._middlewares):
                current_middleware = self._middlewares[index]
                index += 1
                try:
                    # 调用中间件，传入 event 和 next_step 函数
                    await current_middleware(event, next_step)
                except Exception as e:
                    logger.error(f"中间件 {current_middleware.__name__} 执行异常: {e}", exc_info=True)
            else:
                # 所有中间件通过，执行最终处理器
                await final_handler(event)

        # 启动链条
        await next_step()
