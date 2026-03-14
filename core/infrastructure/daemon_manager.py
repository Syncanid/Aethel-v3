# core/infrastructure/daemon_manager.py
import asyncio
import datetime
import logging
import time
from collections import deque
from typing import Dict, Any

from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, EventSource, DetailType

logger = logging.getLogger(__name__)


class DaemonManager:
    def __init__(self, event_bus: EventBus, database: Database, dependency_map: Dict[str, Any]):
        self.event_bus = event_bus
        self.database = database
        self.dependency_map = dependency_map
        self.running_tasks: Dict[str, asyncio.Task] = {}
        # 为每个脚本维护一个内存级双端队列，最多保留 100 行最近日志
        self.daemon_logs: Dict[str, deque] = {}

    async def initialize(self):
        """系统启动时，自动拉起标记为 'running' 的守护进程"""
        logger.info("正在恢复后台守护进程...")
        async with self.database.get_connection() as conn:
            cursor = await conn.execute("SELECT name, code FROM daemon_scripts WHERE status='running'")
            rows = await cursor.fetchall()
            for name, code in rows:
                await self._mount_and_run(name, code)

    async def _update_db_status(self, name: str, status: str):
        async with self.database.get_connection() as conn:
            await conn.execute("UPDATE daemon_scripts SET status=?, updated_at=? WHERE name=?",
                               (status, time.time(), name))
            await conn.commit()

    async def _mount_and_run(self, name: str, code: str) -> bool:
        if name in self.running_tasks:
            self.stop_daemon(name)

        # 初始化/重置该进程的日志队列
        self.daemon_logs[name] = deque(maxlen=100)

        # --- 沙盒函数定义 ---
        def custom_print(*args, sep=' ', end='\n'):
            """拦截脚本中的 print，将其打入该进程专属的日志环形队列中"""
            message = sep.join(str(a) for a in args)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for line in message.splitlines():
                log_entry = f"[{timestamp}] {line}"
                self.daemon_logs[name].append(log_entry)
            # 可选：也输出到主控制台方便调试
            logger.debug(f"[Daemon:{name}] {message}")

        def publish_event(message: str, level: str = "info"):
            """允许脚本主动向系统抛出事件"""
            event = OneBotEvent(
                type=EventType.NOTICE,
                detail_type=DetailType.INTERNAL_DRIVE,
                source=EventSource(platform="internal_daemon"),
                message=message,
                extra={"level": level, "daemon_name": name}
            )
            self.event_bus.publish_event(event)

        # 构造沙盒全局变量
        sandbox_globals = {
            "__builtins__": __builtins__,
            "__name__": f"daemon_{name}",
            "asyncio": asyncio,
            "logger": logging.getLogger(f"daemon.{name}"),
            "print": custom_print,
            "publish": publish_event
        }
        # 注入 Aethel 的底层依赖 (database, api_client 等)
        sandbox_globals.update(self.dependency_map)

        try:
            # 动态执行代码
            exec(code, sandbox_globals)
            daemon_main = sandbox_globals.get("daemon_main")

            if not daemon_main or not asyncio.iscoroutinefunction(daemon_main):
                self.daemon_logs[name].append("[ERROR] 启动失败：缺少 `async def daemon_main():` 入口")
                return False

            # 挂载到后台事件循环
            async def _task_wrapper():
                try:
                    self.daemon_logs[name].append("[SYSTEM] Daemon started.")
                    await daemon_main()
                except asyncio.CancelledError:
                    self.daemon_logs[name].append("[SYSTEM] Daemon stopped by manual cancellation.")
                except Exception as e:
                    import traceback
                    err_msg = traceback.format_exc()
                    self.daemon_logs[name].append(f"[CRASH] {err_msg}")
                    await self._update_db_status(name, "stopped")

            task = asyncio.create_task(_task_wrapper())
            self.running_tasks[name] = task
            logger.info(f"✅ 后台守护进程 [{name}] 启动成功。")
            return True

        except Exception as e:
            self.daemon_logs[name].append(f"[COMPILE ERROR] {e}")
            logger.error(f"Daemon [{name}] 代码编译失败: {e}")
            return False

    # --- 供工具调用的 CRUD 接口 ---
    def get_recent_logs(self, name: str, lines: int = 20) -> str:
        if name not in self.daemon_logs:
            return f"没有找到守护进程 {name} 的运行日志记录。"
        log_list = list(self.daemon_logs[name])
        recent_logs = log_list[-lines:] if lines > 0 else log_list
        return "\n".join(recent_logs) if recent_logs else f"守护进程 {name} 的日志为空。"

    async def create_or_edit(self, name: str, code: str):
        async with self.database.get_connection() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO daemon_scripts (name, code, status, created_at, updated_at) VALUES (?, ?, 'stopped', ?, ?)",
                (name, code, time.time(), time.time())
            )
            await conn.commit()

    async def start_daemon(self, name: str) -> str:
        async with self.database.get_connection() as conn:
            cursor = await conn.execute("SELECT code FROM daemon_scripts WHERE name=?", (name,))
            row = await cursor.fetchone()
            if not row: return f"找不到脚本 {name}"

        success = await self._mount_and_run(name, row[0])
        if success:
            await self._update_db_status(name, "running")
            return f"守护进程 {name} 已启动并在后台运行。"
        return f"守护进程 {name} 启动失败，请使用 read_log 查看编译错误。"

    def stop_daemon(self, name: str) -> str:
        if name in self.running_tasks:
            self.running_tasks[name].cancel()
            self.running_tasks.pop(name)
            asyncio.create_task(self._update_db_status(name, "stopped"))
            return f"守护进程 {name} 已停止。"
        return f"守护进程 {name} 当前未运行。"

    async def delete_daemon(self, name: str) -> str:
        self.stop_daemon(name)
        async with self.database.get_connection() as conn:
            await conn.execute("DELETE FROM daemon_scripts WHERE name=?", (name,))
            await conn.commit()
        if name in self.daemon_logs:
            del self.daemon_logs[name]
        return f"守护进程 {name} 已彻底删除。"
