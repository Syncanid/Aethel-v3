# main.py
import argparse
import asyncio
import logging
import platform
import sys
import threading
from typing import List, Optional

from core.evolution.skill_fabricator import SkillFabricator
from core.gui.monitor_registry import monitor_registry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.daemon_manager import DaemonManager
from core.infrastructure.database import Database
from core.infrastructure.logger import setup_logger
from core.io.adapters.console import ConsoleAdapter
from core.io.adapters.onebot_v11 import OneBotV11Adapter
from core.io.event_bus import EventBus
from core.kernel.agent import AutonomousAgent
from core.kernel.task_engine import TaskEngine

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
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.shutdown_event: Optional[asyncio.Event] = None
        self.tasks: List[asyncio.Task] = []

        # 组件引用
        self.config = None
        self.db = None
        self.bus = None
        self.api_client_global = None
        self.daemon_manager = None
        self.skill_fabricator = None
        self.agent = None
        self.task_engine = None
        self.recorder = None
        self.adapters = []

    async def bootstrap(self):
        """系统引导程序"""
        print(BANNER)

        self.loop = asyncio.get_running_loop()
        self.shutdown_event = asyncio.Event()

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

        self.api_client_global = GenericAPIClient(self.config)
        dependency_map = {
            "config": self.config,
            "database": self.db,
            "event_bus": self.bus,
            "api_client": self.api_client_global
        }
        self.daemon_manager = DaemonManager(self.bus, self.db, dependency_map)
        await self.daemon_manager.initialize()

        self.skill_fabricator = SkillFabricator(self.bus, self.api_client_global)

        # 5. 唤醒 Agent
        # Agent 内部会自动初始化 ToolManager, Hippocampus, Scheduler
        self.agent = AutonomousAgent(self.config, self.bus, self.db)

        # 初始化 TaskEngine
        self.task_engine = TaskEngine(self.config, self.bus, self.db)

        # 6. 加载适配器 (感官)
        console_adapter = ConsoleAdapter(self.bus)
        self.adapters.append(console_adapter)

        # OneBot v11 适配器
        ob_adapter = OneBotV11Adapter(self.bus, self.config)
        self.adapters.append(ob_adapter)

        # 依赖注入
        self.agent.tool_manager.add_dependency("ob_adapter", ob_adapter)
        self.agent.tool_manager.add_dependency("adapters", self.adapters)
        self.agent.tool_manager.add_dependency("daemon_manager", self.daemon_manager)

        self.task_engine.daemon_manager = self.daemon_manager

        # === 注册 GUI 监控点 ===
        self._register_monitors()

        logger.info("系统组件初始化完成。")

    def _register_monitors(self):
        """注册 GUI 监控源"""
        # 注册配置信息
        monitor_registry.register_text_source(
            "系统", "配置",
            lambda: self.config.all
        )
        monitor_registry.register_text_source(
            "系统", "后台进程",
            lambda: "\n".join(
                [f"- {name}" for name in self.daemon_manager.running_tasks.keys()]) if self.daemon_manager else "无"
        )

    async def start(self):
        """启动所有服务"""
        logger.info(">>> Aethel v3 正在觉醒 <<<")

        # 1. 启动 Agent 主循环
        agent_task = asyncio.create_task(self.agent.run_autonomous_loop(), name="Agent-Core")
        self.tasks.append(agent_task)

        # 2. 启动 Task Engine 主循环
        engine_task = asyncio.create_task(self.task_engine.run_engine_loop(), name="Task-Engine")
        self.tasks.append(engine_task)

        # 3. 启动适配器
        for adapter in self.adapters:
            t = asyncio.create_task(adapter.run(), name=f"Adapter-{adapter.platform_name}")
            self.tasks.append(t)

        logger.info(f"已启动 {len(self.tasks)} 个核心进程。")

        # 等待停止信号
        await self.shutdown_event.wait()

    async def shutdown(self):
        """优雅退出流程"""
        if not self.shutdown_event or self.shutdown_event.is_set():
            return

        logger.warning("正在启动退出流程...")
        self.shutdown_event.set()

        # 强制停止所有后台脚本
        if self.daemon_manager:
            for d_name in list(self.daemon_manager.running_tasks.keys()):
                self.daemon_manager.stop_daemon(d_name)

        # 1. 捕获当前 loop 中所有的任务
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task(loop)
        all_tasks = [t for t in asyncio.all_tasks(loop) if t is not current_task]

        if all_tasks:
            logger.info(f"正在向 {len(all_tasks)} 个后台异步任务发送取消信号...")
            for task in all_tasks:
                task.cancel()

            # 2. 关键：等待所有任务处理完 CancelledError 的后事
            logger.info("正在等待后台任务完成清理归档...")
            await asyncio.gather(*all_tasks, return_exceptions=True)

        # 3. 关闭所有 API Client
        if self.agent and self.agent.api_client:
            await self.agent.api_client.close()
        if self.task_engine and self.task_engine.api_client:
            await self.task_engine.api_client.close()
        if self.api_client_global:
            await self.api_client_global.close()

        logger.info("系统已完全关闭。再见。")


def start_backend_thread(system):
    """在子线程中运行 asyncio 事件循环"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(system.bootstrap())
        loop.run_until_complete(system.start())
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.critical(f"后端线程异常: {e}", exc_info=True)
    finally:
        # 确保 shutdown 被调用
        try:
            loop.run_until_complete(system.shutdown())
        except Exception as e:
            logger.error(f"关闭清理过程异常: {e}", exc_info=True)
        finally:
            loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aethel-v3 AI Agent System")
    parser.add_argument("--nogui", action="store_true", help="以无头模式启动 (不显示 GUI)")
    args = parser.parse_args()

    # Windows 兼容性
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 1. 创建 System 实例
    system = AethelSystem()

    if args.nogui:
        # === NoGUI 模式 ===
        print(">>> 正在以无头模式 (NoGUI) 启动 <<<")
        print(">>> 按 Ctrl+C 停止系统 <<<")

        # 直接在主线程创建循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 引导并启动
            loop.run_until_complete(system.bootstrap())
            loop.run_until_complete(system.start())
        except KeyboardInterrupt:
            logger.info("接收到键盘中断信号 (Ctrl+C)，准备退出...")
            if system.shutdown_event:
                system.shutdown_event.set()
        except Exception as e:
            logger.critical(f"系统运行异常: {e}", exc_info=True)
        finally:
            # 优雅退出
            try:
                loop.run_until_complete(system.shutdown())
            except Exception as e:
                logger.error(f"关闭过程中发生错误: {e}", exc_info=True)
            loop.close()
            sys.exit(0)

    else:
        # === GUI 模式 (默认) ===
        # 延迟导入，避免 NoGUI 模式下缺少 PyQt6 导致报错
        try:
            from core.gui.dashboard import run_gui
        except ImportError as e:
            print(f"错误: 无法导入 GUI 模块 ({e})。")
            print("提示: 请安装 PyQt6 或使用 'python main.py --nogui' 启动无头模式。")
            sys.exit(1)

        # 2. 在子线程启动后端 (不再传递外部创建的 loop)
        t = threading.Thread(target=start_backend_thread, args=(system,), daemon=True)
        t.start()

        print(">>> 正在启动 GUI 监控终端 (关闭窗口以退出系统) <<<")

        # 3. 在主线程运行 GUI (阻塞直到窗口关闭)
        try:
            exit_code = run_gui()
        except KeyboardInterrupt:
            exit_code = 0
        except Exception as e:
            print(f"GUI Error: {e}")
            exit_code = 1

        # 4. 退出处理：物理切断并等待子线程安全回收
        print("正在停止后台服务...")
        if system.loop and system.loop.is_running() and system.shutdown_event:
            system.loop.call_soon_threadsafe(system.shutdown_event.set)

        t.join(timeout=5.0)  # 给予后台足够的归档和断开时间
        sys.exit(exit_code)
