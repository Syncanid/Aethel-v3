# main.py
import asyncio
import logging
import signal
import sys
from typing import List

# --- 基础设施层 ---
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.infrastructure.logger import setup_logger
from core.io.adapters.console import ConsoleAdapter
from core.io.adapters.onebot import OneBotAdapter
# --- 神经系统层 ---
from core.io.event_bus import EventBus
# --- 认知内核层 ---
from core.kernel.agent import AutonomousAgent

# from core.io.adapters.websocket import OneBotAdapter # (预留：后续对接 OneBot)

# --- 装饰 ---
BANNER = r"""
    ___    ______   ______  __  __   ______   __
   /   |  / ____/  /_  __/ / / / /  / ____/  / /
  / /| | / __/      / /   / /_/ /  / __/    / / 
 / ___ |/ /___     / /   / __  /  / /___   / /___
/_/  |_/_____/    /_/   /_/ /_/  /_____/  /_____/
                                       v3.0 (Trinity)
"""

logger = logging.getLogger("System")


class AethelSystem:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []

        # 组件引用
        self.config = None
        self.db = None
        self.bus = None
        self.agent = None
        self.adapters = []
        self.web_server = None

    async def bootstrap(self):
        """系统引导程序"""
        print(BANNER)

        # 1. 加载配置
        self.config = Config()
        self.config.load()

        # 2. 初始化日志
        setup_logger(self.config)
        logger.info("正在初始化系统核心组件...")

        # 3. 连接数据库
        self.db = Database(self.config)
        await self.db.init()

        # 4. 启动神经总线
        self.bus = EventBus()

        # 5. 唤醒 Agent (大脑)
        # Agent 内部会自动初始化 ToolManager, Hippocampus, Scheduler
        self.agent = AutonomousAgent(self.config, self.bus, self.db)

        # 6. 加载适配器 (感官)
        # 控制台适配器 (始终启用)
        console_adapter = ConsoleAdapter(self.bus)
        self.adapters.append(console_adapter)

        # OneBot 适配器
        if self.config.get("onebot_enabled", True):  # 或者检查 URL 是否存在
            onebot_adapter = OneBotAdapter(self.bus, self.config)
            self.adapters.append(onebot_adapter)

            # [CRITICAL] 依赖注入
            # 将适配器实例注入到 Agent 的 ToolManager 中
            # 这样所有定义了 parameter `bot_client` 的工具都会自动获得这个实例
            self.agent.tool_manager.add_dependency("bot_client", onebot_adapter)

            logger.info("OneBot 适配器已挂载并注入工具层。")

        logger.info("系统组件初始化完成。")

    async def start(self):
        """启动所有服务"""
        logger.info(">>> Aethel v3 正在觉醒 <<<")

        # 1. 启动 Agent 主循环
        agent_task = asyncio.create_task(self.agent.run_autonomous_loop(), name="Agent-Core")
        self.tasks.append(agent_task)

        # 2. 启动适配器
        for adapter in self.adapters:
            t = asyncio.create_task(adapter.run(), name=f"Adapter-{adapter.platform_name}")
            self.tasks.append(t)

        logger.info(f"已启动 {len(self.tasks)} 个核心进程。")

        # 等待停止信号
        await self.shutdown_event.wait()

    async def shutdown(self):
        """优雅退出流程"""
        if self.shutdown_event.is_set():
            return  # 避免重复调用

        logger.warning("正在启动优雅退出流程...")
        self.shutdown_event.set()

        # 1. 取消所有任务
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # 2. 等待任务结束
        if self.tasks:
            logger.info("正在等待后台任务终止...")
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # 3. 关闭数据库连接
        if self.db:
            # 如果需要显示关闭
            pass

        # 4. 关闭 API Client
        if self.agent and self.agent.api_client:
            await self.agent.api_client.close()

        logger.info("系统已完全关闭。再见。")


async def main():
    system = AethelSystem()

    # 注册信号处理 (仅在非 Windows 平台)
    if sys.platform != 'win32':
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(system.shutdown()))
    else:
        logger.debug("Windows 环境检测：使用 KeyboardInterrupt 进行退出处理")

    try:
        await system.bootstrap()
        await system.start()
    except asyncio.CancelledError:
        # 正常退出信号
        pass
    except Exception as e:
        logger.critical(f"系统发生致命错误: {e}", exc_info=True)
    finally:
        # 无论如何（包括 Windows Ctrl+C），都确保执行关闭逻辑
        await system.shutdown()


if __name__ == "__main__":
    # Windows 兼容性设置
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Windows 下 Ctrl+C 会抛出此异常，但在 main() 的 finally 中已经处理了 shutdown
        pass
