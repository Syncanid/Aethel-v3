# tools/System2/daemon_ops.py
from typing import Optional

from core.tool_manager.registry import register


@register()
async def manage_daemon_script(
        action: str,
        name: str,
        code: Optional[str] = None,
        daemon_manager=None
) -> str:
    """
    [高阶能力] 后台守护进程(Daemon)的管理器。
    允许通过此工具创建和管理长驻后台的异步 Python 任务（如定时轮询、端口监听等）。

    【输出与通讯规范】：
    1. 使用 `print()` 静默记录：记录中间状态、心跳、抓取数据等。
    2. 使用 `publish()` 主动通知：仅当发现关键事件、异常、需系统或用户介入时使用。
       ⚠️ 切勿在高频循环中使用 `publish()`！
    3. 若脚本为死循环，必须在循环中 `await asyncio.sleep(N)` 以释放协程控制权。

    :param action: 要执行的操作。
        可选值：
            "create_or_edit", "start", "stop", "restart", "delete",
            "list", "view_code", "read_log"
    :param name: 守护进程唯一标识符（英文，如 "news_crawler"）
    :param code: 当 action 为 "create_or_edit" 时必填，完整的 Python 脚本代码。
                 必须包含 async def daemon_main(): 入口。
    :param daemon_manager: 后台守护进程管理器实例，需实现对应的接口。
    """

    if daemon_manager is None:
        return "❌ Error: daemon_manager 依赖未注入。"

    db = daemon_manager.database

    try:
        if action == "list":
            async with db.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT name, status, updated_at FROM daemon_scripts"
                )
                rows = await cursor.fetchall()
                if not rows:
                    return "当前系统中没有注册任何后台脚本。"

                res = ["**系统中的守护进程列表**："]
                res.extend(
                    f"- **{r[0]}** [状态: {r[1]}]" for r in rows
                )
                return "\n".join(res)

        elif action == "view_code":
            async with db.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT code, status FROM daemon_scripts WHERE name=?",
                    (name,),
                )
                row = await cursor.fetchone()

            if not row:
                return f"❌ 找不到脚本 {name}"
            code_text, status = row
            return f"🧩 脚本 `{name}` (当前状态: {status}) 的代码如下：\n```python\n{code_text}\n```"

        elif action == "read_log":
            logs = daemon_manager.get_recent_logs(name, lines=20)
            status = "运行中" if name in daemon_manager.running_tasks else "已停止"
            return f"📊 守护进程 [{name}] (当前状态: {status}) 的最近日志：\n```text\n{logs}\n```"

        elif action == "create_or_edit":
            if not code:
                return "❌ Error: 创建或编辑脚本必须提供 code 参数。"

            # 去除 Markdown 代码块标记
            for prefix in ("```python", "```"):
                if code.startswith(prefix):
                    code = code[len(prefix):]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

            await daemon_manager.create_or_edit(name, code)
            return f"✅ 脚本 `{name}` 已保存，当前状态为 stopped。可使用 action='start' 启动。"

        elif action == "start":
            return await daemon_manager.start_daemon(name)

        elif action == "stop":
            return daemon_manager.stop_daemon(name)

        elif action == "restart":
            daemon_manager.stop_daemon(name)
            return await daemon_manager.start_daemon(name)

        elif action == "delete":
            return await daemon_manager.delete_daemon(name)

        else:
            return f"❌ Error: 未知的 action '{action}'。"

    except Exception as e:
        return f"⚠️ 发生异常：{type(e).__name__} - {e}"
