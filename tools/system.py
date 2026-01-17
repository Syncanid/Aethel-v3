# tools/system.py
import asyncio
import logging
import platform
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


@register()
async def execute_shell_command(command: str, timeout: int = 60) -> str:
    """
    [System] 在宿主机终端执行 Shell 命令。

    警告: 这是一个高风险工具。
    1. 仅在确实需要操作系统级操作时使用。
    2. 尽量避免交互式命令 (如 vim, top)，可能会导致挂起。
    3. 命令执行有超时限制。

    Args:
        command: 要执行的命令字符串 (例如 'ls -la', 'ping google.com').
        timeout: 超时时间(秒)，默认 60。
    """
    # 简单的黑名单过滤 (防止极度危险操作，虽然对有心人防不住)
    blacklist = ["rm -rf /", ":(){ :|:& };:"]
    for bad in blacklist:
        if bad in command:
            return f"命令被拒绝: 包含危险模式 '{bad}'"

    logger.warning(f"正在执行 Shell 命令: {command}")

    try:
        # 创建子进程
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # 等待结果或超时
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return f"执行超时 ({timeout}s)。进程已终止。"

        # 编解码输出
        def decode(b):
            try:
                return b.decode('utf-8')
            except:
                return b.decode('gbk', errors='ignore')  # 兼容 Windows 中文

        out_str = decode(stdout).strip()
        err_str = decode(stderr).strip()

        result = f"--- Command: {command} ---\n"
        if out_str:
            result += f"[STDOUT]\n{out_str}\n"
        if err_str:
            result += f"[STDERR]\n{err_str}\n"

        if not out_str and not err_str:
            result += "(命令执行完成，无输出)"

        return result

    except Exception as e:
        return f"命令执行异常: {e}"
