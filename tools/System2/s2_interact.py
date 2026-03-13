# System2Tools/s2_interact.py
from typing import Optional

from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, DetailType, TaskPayload, EventSource
from core.tool_manager.registry import register


@register()
async def report_task_progress(
        progress_message: str,
        event_bus: EventBus = None,
        agent_state: Optional[dict] = None
) -> str:
    """
    向 System 1 主动汇报当前任务的关键进度。
    当任务取得阶段性重大进展，或者即将执行耗时较长的操作时，调用此工具主动向系统1（前台）汇报进度。不要滥用，只有关键节点才汇报。
    :param progress_message: 要汇报的具体进度内容。
    """
    task_id = agent_state.get("current_task_id", "unknown")

    # 构造 TaskPayload
    payload = TaskPayload(
        task_id=task_id,
        progress_msg=progress_message
    )

    source = EventSource(
        platform="internal",
    )

    # 发送 TASK_PROGRESS 事件，标记 requires_user_input 为 False
    event = OneBotEvent(
        type=EventType.TASK,
        detail_type=DetailType.TASK_PROGRESS,
        source=source,
        extra={
            "task_payload": payload.model_dump(),
            "requires_user_input": False  # 仅作阶段性汇报，不需要用户回复
        }
    )

    event_bus.publish_event(event)

    # 返回给 S2 LLM 的执行结果
    return "进度已成功汇报给 System 1。"


@register()
async def ask_system1_for_help(
        question: str,
        event_bus: EventBus = None,
        agent_state: Optional[dict] = None
) -> str:
    """
    当执行任务时遇到缺失的关键信息（如需要验证码、密码、确认选项），使用此工具暂停当前思考，向系统1求助。
    :param question: 你需要 System 1 去问用户的具体问题。
    """
    task_id = agent_state.get("current_task_id", "unknown")

    # 构造 TaskPayload
    payload = TaskPayload(
        task_id=task_id,
        progress_msg=f"【等待输入】{question}",
        description=question  # 将问题内容放在这里
    )

    source = EventSource(
        platform="internal",
    )

    # 发送一个特殊的 TASK_PROGRESS 事件，带有 requires_user_input 标记
    event = OneBotEvent(
        type=EventType.TASK,
        detail_type=DetailType.TASK_PROGRESS,
        source=source,
        extra={
            "task_payload": payload.model_dump(),
            "requires_user_input": True
        }
    )

    event_bus.publish_event(event)

    # 返回给 S2 LLM 的观测结果，引导它挂起
    return "已将问题发送给 System 1，请调用 `wait` 工具挂起自己，等待 System 1 通过 TASK_UPDATE 将用户的答案传回给你。"
