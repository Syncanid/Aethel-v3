# tools/System2/terminal_ops.py
import asyncio
import logging
import os
import traceback
import uuid
from collections import deque
from typing import Dict, Any, Optional

import asyncssh

from core.infrastructure.config_loader import get_config
from core.kernel.task_engine import TaskEngine
from core.tool_manager.output_cache import ToolOutputCache
from core.tool_manager.registry import register
from core.utilities import calculate_tokens

logger = logging.getLogger(__name__)

TERMINAL_SESSIONS: Dict[str, Dict[str, Any]] = {}
DELIMITER_PREFIX = "[===AETHEL_CMD_DONE:"
DELIMITER_SUFFIX = "===]"

# 为模型设定安全极限
MAX_SAFE_TOKENS = get_config().get("llm.model_context", 16384) - 1000

IS_WINDOWS = os.name == 'nt'


async def _read_until_delimiter(stdout_stream, stderr_stream, timeout: int = 15) -> str:
    """基于 MPSC 队列与双向链表的全局滑动窗口异步读取器"""

    queue = asyncio.Queue()

    async def _stream_producer(stream):
        """流生产者：将标准输出或错误流按行推入共享队列"""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                await queue.put(line.decode('utf-8', errors='replace'))
        except Exception:
            pass
        finally:
            await queue.put(None)  # 注入 EOF 信号

    # 拉起并发生产者
    task_out = asyncio.create_task(_stream_producer(stdout_stream))
    task_err = asyncio.create_task(_stream_producer(stderr_stream))

    # 消费者状态维护
    rolling_window = deque()
    current_tokens = 0
    has_evicted = False
    exit_code = "UNKNOWN"

    chunk_lines = []
    chunk_char_length = 0
    eof_count = 0

    try:
        # 使用绝对时钟计算剩余超时，防止消费者被永远阻塞
        loop = asyncio.get_event_loop()
        end_time = loop.time() + timeout

        while eof_count < 2:
            time_left = end_time - loop.time()
            if time_left <= 0:
                raise asyncio.TimeoutError()

            try:
                line_str = await asyncio.wait_for(queue.get(), timeout=time_left)
            except asyncio.TimeoutError:
                raise

            if line_str is None:
                eof_count += 1
                continue

            # 捕获定界符：提取退出码并直接跳出，定界符本身不进入历史记录
            if DELIMITER_PREFIX in line_str:
                try:
                    exit_code = line_str.split(DELIMITER_PREFIX)[1].split(DELIMITER_SUFFIX)[0]
                except Exception:
                    pass
                break

            chunk_lines.append(line_str)
            chunk_char_length += len(line_str)

            # 【水位线触发】执行块级 Token 结算与滑动窗口维护
            if chunk_char_length > 2000:
                chunk_text = "".join(chunk_lines)
                tokens = calculate_tokens(chunk_text)

                # 极端防御：单行或单块的 Token 量直接大于阈值（如打印了巨大的 base64）
                if tokens > MAX_SAFE_TOKENS:
                    # 粗略切片截取尾部，确保能塞进队列
                    chunk_text = chunk_text[-MAX_SAFE_TOKENS * 2:]
                    tokens = calculate_tokens(chunk_text)

                # FIFO 驱逐逻辑：从头部剔除旧块，直到为新块腾出足够空间
                while current_tokens + tokens > MAX_SAFE_TOKENS and rolling_window:
                    _, evicted_tokens = rolling_window.popleft()
                    current_tokens -= evicted_tokens
                    has_evicted = True

                rolling_window.append((chunk_text, tokens))
                current_tokens += tokens

                # 重置累加器
                chunk_lines.clear()
                chunk_char_length = 0

        # 循环结束，处理残存在累加器中的最后一部分数据
        if chunk_lines:
            chunk_text = "".join(chunk_lines)
            tokens = calculate_tokens(chunk_text)
            while current_tokens + tokens > MAX_SAFE_TOKENS and rolling_window:
                _, evicted_tokens = rolling_window.popleft()
                current_tokens -= evicted_tokens
                has_evicted = True
            if tokens <= MAX_SAFE_TOKENS:
                rolling_window.append((chunk_text, tokens))

    except asyncio.TimeoutError:
        return f"【Terminal Error】命令执行超时（>{timeout}秒）。可能进入了交互模式或执行被阻塞挂起。"
    finally:
        # 清理可能悬挂的生产者协程
        task_out.cancel()
        task_err.cancel()

    # 组装最终结果
    result_texts = [text for text, _ in rolling_window]

    # 向大模型注入结构化的视觉提示，声明前置数据已丢失
    if has_evicted:
        result_texts.insert(0,
                            f"\n... [日志过长，早期输出已被滑动窗口丢弃，当前仅保留终端最后 {MAX_SAFE_TOKENS} Tokens 的核心数据] ...\n\n")

    final_text = "".join(result_texts).strip()
    return f"Exit Code: {exit_code}\nOutput:\n{final_text if final_text else '<Empty Output>'}"


@register()
async def init_ssh_session(
        host: str,
        username: str,
        port: int = 22,
        password: Optional[str] = None,
        private_key_path: Optional[str] = None,
        task_engine: TaskEngine = None
) -> str:
    """
    建立一个持久化的 SSH 会话。调用成功后，后续的所有 run_terminal_command 将在这个 SSH 环境中执行，并保持目录等上下文。
    """
    task_id = getattr(task_engine, "_current_task_id", str(uuid.uuid4()))

    if task_id in TERMINAL_SESSIONS:
        return "【系统提示】当前任务已存在一个活跃的终端会话。请先调用 close_terminal_session 结束旧会话。"

    try:
        connect_kwargs = {"host": host, "port": port, "username": username, "known_hosts": None}
        if private_key_path:
            connect_kwargs["client_keys"] = [private_key_path]
        elif password:
            connect_kwargs["password"] = password
        else:
            return "【Semantic Error】必须提供 password 或 private_key_path。"

        conn = await asyncssh.connect(**connect_kwargs)
        # 启动一个 bash 进程
        process = await conn.create_process('/bin/bash')

        TERMINAL_SESSIONS[task_id] = {
            "type": "ssh",
            "conn": conn,
            "process": process,
            "stdin": process.stdin,
            "stdout": process.stdout,
            "stderr": process.stderr
        }

        # 初始化环境，防止乱码和提示符干扰
        process.stdin.write("export TERM=dumb\n")

        return f"SSH 会话已成功建立 ({username}@{host}:{port})。环境就绪，你可以开始调用 run_terminal_command。"
    except Exception as e:
        return f"【Terminal Error】SSH 连接失败: {str(e)}"


@register()
async def run_terminal_command(
        command: str,
        purpose: str,
        timeout: int = 15,
        task_engine: TaskEngine = None
) -> str:
    """
    在当前终端会话中执行命令。

    【极大危险警告】: 绝对禁止执行 top, tail -f, less, vi 等交互式或无限阻塞命令！

    :param command: 要执行的 Shell 命令。
    :param purpose: 【必填】你执行这条命令的明确目的！例如“查找端口占用进程”或“查看前20行配置”。
                    如果输出结果很长，底层系统将启动提炼模型，【严格按照此目的】为你过滤废话。
                    如果你填得含糊不清，提炼模型会把所有数据当废话删除！
    """
    task_id = getattr(task_engine, "_current_task_id", str(uuid.uuid4()))

    # 动态异构环境初始化
    if task_id not in TERMINAL_SESSIONS:
        try:
            if IS_WINDOWS:
                # Windows 环境：拉起 PowerShell 并绕过执行策略
                proc = await asyncio.create_subprocess_shell(
                    'powershell.exe -NoProfile -ExecutionPolicy Bypass',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                # Unix 环境
                proc = await asyncio.create_subprocess_shell(
                    '/bin/bash',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            TERMINAL_SESSIONS[task_id] = {
                "type": "local",
                "conn": None,
                "process": proc,
                "stdin": proc.stdin,
                "stdout": proc.stdout,
                "stderr": proc.stderr
            }

            if not IS_WINDOWS:
                proc.stdin.write(b"export TERM=dumb\n")
            await proc.stdin.drain()
            logger.info(f"🐚 [System 2] 启动了本地持久化终端 (平台: {'Windows' if IS_WINDOWS else 'Unix'})")
        except Exception as e:
            logger.debug(f"Current Loop Type: {type(asyncio.get_running_loop())}")
            logger.error(f"子进程拉起失败详情: {traceback.format_exc()}")
            return f"【SYSTEM CRASH】无法拉起本地进程: {str(e)}"

    session = TERMINAL_SESSIONS[task_id]

    # 跨平台定界符注入策略
    if session["type"] == "local" and IS_WINDOWS:
        # PowerShell 语法注入。注意转义双引号和换行符 `n
        # 即使报错也要保证定界符能输出
        injected_command = f"Try {{ Invoke-Expression -Command '{command.replace(chr(39), chr(39) + chr(39))}' }} Finally {{ Write-Output \"`n{DELIMITER_PREFIX}$LASTEXITCODE{DELIMITER_SUFFIX}\" }}\n"
    else:
        # Bash / SSH 语法注入
        injected_command = f"({command}) ; echo -e \"\\n{DELIMITER_PREFIX}$?{DELIMITER_SUFFIX}\"\n"

    try:
        if session["type"] == "ssh":
            session["stdin"].write(injected_command)
        else:
            # 解决 Windows 平台的编码问题 (GBK / UTF-8 混杂)
            encode_type = 'gbk' if IS_WINDOWS else 'utf-8'
            session["stdin"].write(injected_command.encode(encode_type, errors='replace'))
            await session["stdin"].drain()

        # 1. 获得最多 MAX_SAFE_CHARS 的终端回显
        raw_terminal_result = await _read_until_delimiter(session["stdout"], session["stderr"], timeout=timeout)

        # 2. 如果发生了 Terminal Error (超时等)，直接返回，不走提炼
        if "【Terminal Error】" in raw_terminal_result:
            return raw_terminal_result

        # 3. 接入缓存与小模型提炼层
        # 设定阈值 500 字：少于 500 字直接原样返回；多于 500 字才启动 LLM 提炼，节省 Token。
        receipt_id, refined, final_response = await ToolOutputCache.process_tool_output(
            raw_content=raw_terminal_result,
            purpose=purpose,
            threshold=500
        )

        # 4. 额外包装：确保最终回传的文本保留退出码信息，这对系统决策至关重要
        # 将原始输出的首行（Exit Code: X）强行拼接到提炼结果前面
        exit_code_line = raw_terminal_result.split('\n')[0]
        return f"{exit_code_line}\n{final_response}"

    except Exception as e:
        return f"【Semantic Error】命令写入流失败: {str(e)}"


@register()
async def close_terminal_session(task_engine: TaskEngine = None) -> str:
    """
    清理并销毁当前任务的终端会话。当任务完成或不再需要控制台时调用。
    """
    task_id = getattr(task_engine, "_current_task_id", "UNKNOWN")

    if task_id in TERMINAL_SESSIONS:
        session = TERMINAL_SESSIONS.pop(task_id)
        try:
            if session["type"] == "ssh":
                session["process"].terminate()
                session["conn"].close()
            else:
                session["process"].terminate()
        except Exception:
            pass
        return "终端会话已销毁，上下文已清除。"

    return "当前没有活跃的终端会话。"
