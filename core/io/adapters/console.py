import asyncio
import logging
import sys
from typing import Optional

from core.io.adapters.base import BaseAdapter
from core.io.event_schema import OneBotEvent, EventType, DetailType, EventSource, Action, ActionResponse, ActionStatus

logger = logging.getLogger(__name__)


class ConsoleAdapter(BaseAdapter):
    @property
    def platform_name(self) -> str:
        return "console"

    async def run(self):
        """
        使用 run_in_executor 在后台线程监听 stdin，避免阻塞 asyncio 循环。
        """
        print(f"\n--- {self.platform_name} 适配器已启动。请输入消息并回车 ---\n")
        loop = asyncio.get_running_loop()

        while True:
            try:
                # 在执行器中阻塞读取，不影响主循环
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                content = line.strip()
                if not content:
                    continue

                # 构造标准事件
                event = OneBotEvent(
                    type=EventType.MESSAGE,
                    detail_type=DetailType.PRIVATE,
                    sub_type="console",
                    source=EventSource(
                        platform=self.platform_name,
                        user_id="admin_console"
                    ),
                    message=content,
                    alt_message=content,
                    raw_data={"raw": line}
                )

                # 发布到总线
                self.event_bus.publish_event(event)

            except (UnicodeDecodeError, KeyboardInterrupt):
                logger.info(f"\n[{self.platform_name}] 接收到控制台输入中断信号，已停止监听。")
                break
            except Exception as e:
                logger.error(f"控制台输入错误: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def handle_action(self, action: Action) -> Optional[ActionResponse]:
        """
        处理输出动作
        """
        # 简单的路由判断：如果是广播或者目标是 console，则显示
        if action.target_platform and action.target_platform != self.platform_name:
            return None

        if action.action == "send_message":
            params = action.params
            msg = params.get("message", "")
            logger.info(f"\n[Aethel] >> {msg}\n")
            return ActionResponse(status=ActionStatus.OK, message="Printed to console")

        elif action.action == "broadcast_log":
            content = action.params.get("content", "")
            logger.info(f"[状态] {content}")
            return ActionResponse(status=ActionStatus.OK)

        return None
