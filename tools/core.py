# tools/core.py
import json
from typing import Dict, Any

from core.io.event_bus import EventBus
from core.tool_manager.registry import register


@register()
async def update_scratchpad(
        goal: str,
        progress: str,
        next_action: str,
        agent_state: Dict[str, Any],  # 依赖注入
        event_bus: EventBus  # 依赖注入
) -> str:
    """
    更新你的内部状态（记事本）。
    当你的目标改变、取得进展或决定采取新计划时，请务必调用此工具。

    Args:
        goal: 你当前正在努力实现的最高级目标。
        progress: 到目前为止已完成工作的总结。
        next_action: 你计划立即采取的下一步行动。
    """
    # 直接修改注入的引用
    agent_state["goal"] = goal
    agent_state["progress"] = progress
    agent_state["next_action"] = next_action

    # 广播日志
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"📝 状态已更新: {json.dumps(agent_state, ensure_ascii=False)}"}
    ))
    return "记事本更新成功。"


@register()
async def send_message(
        message: str,
        event_bus: EventBus  # 依赖注入
) -> str:
    """
    向用户或控制台发送文本消息。
    使用此工具回复用户或陈述你的发现。

    Args:
        message: 消息的具体内容。
    """
    event_bus.publish_action(Action(
        action="send_message",
        params={"message": message}
    ))
    return "消息已外发。"


import asyncio
import logging
import sys
import io
import datetime
import traceback
from dateutil import parser
from typing import Dict, Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.tool_manager.registry import register
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, EventSource, Action

logger = logging.getLogger(__name__)


# --- 内部辅助函数：发送唤醒事件 ---
async def _dispatch_wake_up(event_bus: EventBus, reason: str):
    """调度器回调：发送唤醒事件"""
    logger.info(f"⏰ 等待结束，触发唤醒: {reason}")
    event = OneBotEvent(
        type=EventType.NOTICE,  # 使用 Notice 类型
        detail_type="wake_up",
        sub_type="timer",
        source=EventSource(platform="system", user_id="timer"),
        message=f"【系统唤醒】: {reason}",
        alt_message=f"【系统唤醒】: {reason}",
        extra={"status": "wake_up"}
    )
    event_bus.publish_event(event)


@register()
async def wait(
        duration: Optional[float] = None,
        until: Optional[str] = None,
        reason: str = "Timer expired",
        scheduler: AsyncIOScheduler = None,  # 注入
        event_bus: EventBus = None  # 注入
) -> str:
    """
    让系统进入等待/挂起状态。
    系统会挂起当前任务，直到指定时间到达后通过事件被唤醒。
    注意：在等待期间，如果有新的用户消息，系统依然会被打断并处理。

    Args:
        duration: 等待的秒数 (相对时间)。例如 30.5。
        until: 等待直到具体的日期时间 (绝对时间)。支持 ISO 格式 (如 '2026-01-20 08:00:00')。
        reason: 唤醒时的提示信息。
    """
    run_date = None

    # 1. 计算触发时间
    if until:
        try:
            # 使用 dateutil 解析自然语言或 ISO 时间
            run_date = parser.parse(until)
            # 如果解析出的时间没有时区，且当前是 Awareness 的，需处理 (这里简化，假设本地时间)
            if run_date < datetime.datetime.now():
                return f"错误: 目标时间 {until} 已经是过去式了。"
        except Exception as e:
            return f"错误: 无法解析时间字符串 '{until}'。请使用 ISO 格式 (YYYY-MM-DD HH:MM:SS)。"

    elif duration is not None:
        if duration <= 0:
            return "错误: 等待时长必须大于 0。"
        run_date = datetime.datetime.now() + datetime.timedelta(seconds=duration)

    else:
        return "错误: 必须提供 'duration' (秒) 或 'until' (日期字符串) 其中之一。"

    # 2. 添加调度任务
    if scheduler and run_date:
        scheduler.add_job(
            _dispatch_wake_up,
            'date',
            run_date=run_date,
            args=[event_bus, reason]
        )
        timestamp_str = run_date.strftime("%Y-%m-%d %H:%M:%S")
        return f"已进入休眠模式。系统将在 {timestamp_str} 唤醒，原因: {reason}。"
    else:
        return "系统错误: 调度器未初始化。"


# --- 工具 2: 接入 EventBus 的 Python 解释器 ---

class AethelInterface:
    """注入到 Python 脚本中的 API 对象"""

    def __init__(self, event_bus: EventBus):
        self._bus = event_bus

    def print(self, *args):
        """模拟 print"""
        print(*args)  # 会被重定向捕获

    def send_event(self, type: str, detail_type: str, message: str, **kwargs):
        """
        [高级] 直接向系统总线注入一个 Event。
        这让脚本可以模拟用户说话，或者触发系统通知。
        """
        event = OneBotEvent(
            type=type,  # e.g., "message", "notice"
            detail_type=detail_type,
            source=EventSource(platform="script", user_id="python_interpreter"),
            message=message,
            alt_message=message,
            extra=kwargs
        )
        self._bus.publish_event(event)

    def dispatch_action(self, action_name: str, params: Dict[str, Any] = None):
        """
        [高级] 直接触发系统 Action。
        例如: api.dispatch_action("send_message", {"message": "Hello"})
        """
        if params is None: params = {}
        action = Action(
            action=action_name,
            params=params,
            target_platform="script"
        )
        self._bus.publish_action(action)


@register()
async def python_interpreter(
        code: str,
        event_bus: EventBus  # 注入
) -> str:
    """
    [Code] 执行一段 Python 脚本。
    该环境具有高权限，可以通过 `api` 对象直接与系统事件总线交互。

    警告：
    1. 不要使用此工具来单纯打印文本或列表，如果要回复用户，请直接使用 `send_message`。
    2. 仅在需要计算、逻辑处理或操作 `api` 时使用。

    可用对象:
    - api: AethelInterface 实例

    Args:
        code: 要执行的 Python 代码字符串。
    """

    # 1. 准备沙箱环境
    output_capture = io.StringIO()
    interface = AethelInterface(event_bus)

    sandbox_globals = {
        "__builtins__": __builtins__,  # 允许基础内置函数
        "api": interface,
        "datetime": datetime,
        "asyncio": asyncio,
        "json": __import__("json"),
        "math": __import__("math")
    }

    # 2. 捕获 stdout
    original_stdout = sys.stdout
    sys.stdout = output_capture

    error_msg = None
    try:
        # 执行代码
        exec(code, sandbox_globals)
    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = original_stdout

    # 3. 整理结果
    output = output_capture.getvalue()

    result_msg = f"--- 脚本执行输出 ---\n{output}"
    if error_msg:
        result_msg += f"\n--- 运行时错误 ---\n{error_msg}"

    # 如果没有输出也没有错误，提示成功
    if not output and not error_msg:
        result_msg += "\n(脚本执行完毕，无文本输出)"

    return result_msg
