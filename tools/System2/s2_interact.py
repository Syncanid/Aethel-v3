# System2Tools/s2_interact.py
import logging
from typing import Optional

from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, DetailType, TaskPayload, EventSource
from core.kernel.task_engine import TaskEngine
from core.kernel.task_registry import global_task_registry
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


def _resolve_session_to_source(session_id: str) -> EventSource:
    """反向解析引擎"""
    # 切割格式：'group' 和 'onebot:12345'
    ctx_type, puid = session_id.split('_', 1)
    # 进一步切割出平台与纯数字 ID
    plat, ctx_id = puid.split(':', 1)

    if ctx_type in ["group", "private"]:
        return EventSource(platform=plat, group_id=ctx_id)
    return EventSource(platform="internal")


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
    payload = TaskPayload(task_id=task_id, progress_msg=progress_message)

    # 提取订阅者矩阵
    subscribers = await global_task_registry.get_subscribers(task_id)
    if not subscribers:
        subscribers = ["internal_default"]

    # 遍历拓扑散射
    for session_id in subscribers:
        source = _resolve_session_to_source(session_id)
        event = OneBotEvent(
            type=EventType.TASK,
            detail_type=DetailType.TASK_PROGRESS,
            source=source,
            extra={
                "task_payload": payload.model_dump(),
                "requires_user_input": False
            }
        )
        event_bus.publish_event(event)

    # 返回给 S2 LLM 的执行结果
    return "进度已成功汇报。"


@register()
async def revert_to_checkpoint(
        reason: str,
        task_engine: TaskEngine = None
) -> str:
    """
    当任务陷入死胡同、遇到无法解决的逻辑错误，或者你意识到当前的探索路径完全错误时，调用此工具。
    它将清除当前所有被污染的记忆和状态，将系统时间线倒退回上一个安全、稳定的检查点。

    :param reason: 必须详细说明为什么这条路走不通（例如：“目标文件夹不存在，且尝试创建失败，该方案不可行”）。系统将记录这个教训。
    """
    # 将当前的失败原因透传给引擎，引擎负责合并所有的失败记忆并执行物理回滚
    injection_prompt = await task_engine.rollback_to_last_stable(new_reason=reason)

    # 这一长串带有所有失败历史的 Prompt 将作为本工具的 output 挂载到刚恢复的纯净 history 末尾
    return injection_prompt


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

    subscribers = await global_task_registry.get_subscribers(task_id)
    if not subscribers:
        subscribers = ["internal_default"]

    for session_id in subscribers:
        source = _resolve_session_to_source(session_id)
        event = OneBotEvent(
            type=EventType.TASK,
            detail_type=DetailType.TASK_PROGRESS,
            source=source,
            extra={
                "task_payload": payload.model_dump(),
                "requires_user_input": True  # 触发中断
            }
        )
        event_bus.publish_event(event)

    return f"强制中断请求已分发至 {len(subscribers)} 个监控通道。请调用 `wait` 工具将当前线程挂起，监听回传信号。"


@register()
async def conclude_task(
        result: str,
        status: str,
        generate_skill: bool = False,
        task_engine: TaskEngine = None
) -> str:
    """
    当任务已经得出最终结论，或者确认彻底失败无法继续时，必须调用此工具来结束任务进程。

    :param result: 任务的最终执行结果、结论或失败原因汇总。此内容将直接作为最终报告。
    :param status: 任务最终定性状态，必须是 "success" 或 "failure"。
    :param generate_skill: 如果本次任务成功，并且你认为这次探索出了一套高价值、可复用的工作流或代码，将其设为 True，系统会在后台自动总结并固化为一个标准 Skill。
    """
    task_engine.task_status = {
        "finished": True,
        "final_result": result,
        "status": status,
        "skill": generate_skill
    }

    logger.info(f"✅ [System 2] 任务主动宣布结束。结论: {result[:50]}...")

    return f"任务结束信号已发送。结论：{result}。是否触发技能演化：{generate_skill}。"
