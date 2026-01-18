# tools/core.py
import asyncio
import datetime
import io
import json
import logging
import sys
import traceback
from typing import Dict, Any, Optional, List, Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dateutil import parser

from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, EventSource, Action, ActionStatus
from core.kernel.agent import AutonomousAgent
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


@register()
async def send_message(
        message: str,
        target_id: str,
        platform: str,
        target_type: Literal["private", "group", "channel"],
        event_bus: EventBus = None,
) -> str:
    """
    发送消息。支持指定发送目标（私聊/群组）。

    Args:
        message: 消息内容。
        target_id: 目标用户ID 或 群组ID。
        platform: 目标平台 (如 'qq', 'telegram')。
        target_type: 消息类型 ('private', 'group', 'channel')。
    """

    if not target_id:
        return "错误: 无法确定发送目标 (target_id 为空)。"

    # 2. 构造 Action 参数
    params = {
        "message": message,
        "user_id": target_id if target_type == "private" else None,
        "group_id": target_id if target_type == "group" else None,
        "detail_type": target_type
    }

    # 3. 发布动作
    action = Action(
        action="send_message",
        params=params,
        target_platform=platform  # 指定平台适配器处理
    )

    # 2. 同步分发并等待结果
    try:
        # 设置 10 秒超时，避免 Agent 永久卡死
        response = await event_bus.dispatch_action(action, timeout=10.0)

        # 3. 处理不同的结果状态
        if response.status == ActionStatus.OK:
            # 成功
            return f"消息已发送 [{platform}] {target_type}: {target_id}"

        else:
            # 失败处理
            error_msg = response.message

            # 特殊错误：平台不存在 (Route not matched)
            if "Route not matched" in error_msg:
                return (
                    f"发送失败: 找不到平台适配器 '{platform}'。\n"
                    f"可能原因：\n"
                    f"1. 平台名称拼写错误\n"
                    f"2. 该平台的适配器未在 main.py 中加载"
                )

            # 其他错误 (如网络超时、被禁言)
            return f"发送失败: {error_msg}"

    except asyncio.TimeoutError:
        return f"发送超时: 平台 '{platform}' 在 10 秒内没有响应。"
    except Exception as e:
        return f"系统异常: 发送过程中发生错误 - {str(e)}"


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
        scheduler: AsyncIOScheduler = None,
        event_bus: EventBus = None,
        agent: AutonomousAgent = None,
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
        # 获取 job 对象
        job = scheduler.add_job(
            _dispatch_wake_up,
            'date',
            run_date=run_date,
            args=[event_bus, reason]
        )

        # 将 Job ID 绑定到 Agent 实例，用于后续取消
        if agent:
            agent.wakeup_job_id = job.id
            agent.is_sleeping = True

        timestamp_str = run_date.strftime("%Y-%m-%d %H:%M:%S")
        return f"已进入休眠模式。系统将在 {timestamp_str} 唤醒，原因: {reason}。"
    else:
        return "系统错误: 调度器未初始化。"


@register()
async def wait_forever(
        reason: str = "Standby",
        agent: Any = None,
        event_bus: EventBus = None
) -> str:
    """
    [Control] 进入无限期休眠状态，直到收到外部事件（如用户消息）唤醒。
    这比 wait(duration) 更适合空闲状态。

    Args:
        reason: 休眠的具体原因（例如："等待用户指令", "任务已完成"）。
    """
    if agent:
        # 如果存在之前的定时唤醒任务（例如之前的 wait 设置的），则取消它
        # 避免定时器在休眠期间意外触发唤醒
        if hasattr(agent, "wakeup_job_id") and agent.wakeup_job_id:
            try:
                # 假设 agent.scheduler 是 AsyncIOScheduler 实例
                agent.scheduler.remove_job(agent.wakeup_job_id)
                logger.info(f"[WaitForever] 已移除原有的定时唤醒任务: {agent.wakeup_job_id}")
            except Exception as e:
                # 忽略任务不存在的错误
                logger.debug(f"[WaitForever] 移除定时任务失败 (可能已不存在): {e}")
            agent.wakeup_job_id = None

        # 强制设置休眠标志
        agent.is_sleeping = True
        logger.info(f"Agent entering infinite sleep: {reason}")

    # 广播系统状态变更
    if event_bus:
        event_bus.publish_action(Action(
            action="broadcast_log",
            params={"content": f"💤 系统进入无限期待命状态: {reason}"}
        ))

    return f"系统已进入无限期待命状态 ({reason})。停止思考循环，等待外部事件唤醒。"


# --- 工具 2: 接入 EventBus 的 Python 解释器 ---

class AethelInterface:
    """注入到 Python 脚本中的 API 对象"""

    def __init__(self, event_bus: EventBus, tool_manager: Any = None, agent_state: Dict[str, Any] = None):
        self._bus = event_bus
        self._tool_manager = tool_manager
        self._agent_state = agent_state

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

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取当前系统已加载的真实工具列表"""
        if self._tool_manager:
            # 返回 schema 列表（通常包含 name, description）
            return self._tool_manager.get_tool_schemas()
        return [{"error": "ToolManager not available"}]

    def get_system_status(self) -> Dict[str, Any]:
        """获取真实的系统状态快照"""
        status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent_state": self._agent_state if self._agent_state else "Unknown",
            "components": {
                "event_bus": "active",
                "tool_manager": "active" if self._tool_manager else "inactive"
            }
        }
        return status


@register()
async def python_interpreter(
        code: str,
        event_bus: EventBus,
        tool_manager: Any = None,
        agent_state: Dict[str, Any] = None
) -> str:
    """
    [Code] 执行一段 Python 脚本，返回stdout和stderr。
    该环境具有高权限，预置了全局对象 `api` 用于与系统交互。

    使用说明：
    1. `api` 是直接可用的全局对象，**严禁**使用 `import api` 或 `from api import ...`。
    2. `api` 对象仅支持以下方法：
       - `api.get_available_tools() -> List[Dict]`: 获取可用工具列表。
       - `api.get_system_status() -> Dict`: 获取系统状态快照(含时间)。
       - `api.send_event(type, detail_type, message, **kwargs)`: 注入事件。
       - `api.dispatch_action(action_name, params)`: 触发动作。
       - `api.print(*args)`: 打印日志。

    警告：
    1. 【严禁模拟】绝对禁止编写代码来"手动定义"工具列表或系统状态。
    2. 不要使用此工具来单纯打印文本，如果要回复用户，请使用 `send_message`。

    Args:
        code: 要执行的 Python 代码字符串。
    """

    # 1. 准备沙箱环境
    output_capture = io.StringIO()
    # 将依赖传入接口
    interface = AethelInterface(event_bus, tool_manager, agent_state)

    available_tools = []
    if tool_manager:
        available_tools = tool_manager.get_tool_schemas()

    sandbox_globals = {
        "__builtins__": __builtins__,  # 允许基础内置函数
        "api": interface,
        "tools": available_tools,
        "agent_state": agent_state,
        "datetime": datetime,
        "asyncio": asyncio,
        "json": json,
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


@register()
async def get_active_adapters(adapters: List[Any]) -> str:
    """
    [System] 获取当前系统已加载的所有 IO 适配器列表。
    用于检查系统是否正确连接到了各个平台（如 Console, OneBot 等）。
    """
    lines = ["当前活跃的适配器接口:"]
    for i, adapter in enumerate(adapters, 1):
        # 获取平台名称 (BaseAdapter 属性)
        name = getattr(adapter, "platform_name", "Unknown")
        # 获取类名作为辅助信息
        class_name = adapter.__class__.__name__

        # 尝试获取运行状态 (如果适配器有 is_running 属性)
        status_suffix = ""
        if hasattr(adapter, "_running"):
            status = "运行中" if getattr(adapter, "_running") else "已停止"
            status_suffix = f" - {status}"

        lines.append(f"{i}. {name.upper()} ({class_name}){status_suffix}")

    return "\n".join(lines)
