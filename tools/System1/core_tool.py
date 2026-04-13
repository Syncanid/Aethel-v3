# tools/core_tool.py
import asyncio
import datetime
import logging
from typing import Any, Optional, List, Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dateutil import parser

from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, EventSource, Action, ActionStatus
from core.kernel.agent import AutonomousAgent
from core.limbic.manager import LimbicManager
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


# --- 内部辅助函数：发送唤醒事件 ---
async def _dispatch_wake_up(event_bus: EventBus, reason: str):
    """调度器回调：发送唤醒事件"""
    logger.info(f"⏰ 等待结束，触发唤醒: {reason}")
    event = OneBotEvent(
        type=EventType.NOTICE,  # 使用 Notice 类型
        detail_type="wake_up",
        sub_type="timer",
        source=EventSource(platform="system"),
        message=f"【系统唤醒】: {reason}",
        alt_message=f"【系统唤醒】: {reason}",
        extra={"status": "wake_up"}
    )
    event_bus.publish_event(event)


@register()
async def send_message(
        message: str,
        platform: str,
        target_id: str,
        target_type: Literal["private", "group", "channel"],
        wait_minutes: Optional[float] = None,
        event_bus: EventBus = None,
        agent: AutonomousAgent = None,
        scheduler: AsyncIOScheduler = None,
        fragmenter: Any = None,
) -> str:
    """
    发送消息。支持指定发送目标（私聊/群组）。
    发送消息之前必须获取基本信息（平台和用户ID/群组ID）。
    如果你希望在发送完这条消息后立刻进入等待/休眠状态，请填写 wait_minutes 参数。

    Args:
        message: 消息内容。
        platform: 目标平台（PUID的前半部分）。
        target_id: 目标user_id或group_id（PUID的后半部分）。
        target_type: 消息类型 ('private', 'group', 'channel')。
        wait_minutes: (可选) 发送后等待的分钟数，小于等于0或为空则不等待。
    """

    if not target_id:
        return "错误: 无法确定发送目标 (target_id 为空)。"

    # 1. 情绪碎片化处理
    try:
        # 延迟导入我们上一阶段编写的 OutputFragmenter
        from core.io.fragmentation import OutputFragmenter

        # 尝试从 agent 实例中获取 limbic 状态
        if agent and hasattr(agent, "limbic"):
            state = await agent.limbic.get_state()
            # 将 LLM 生成的完整文本切碎，并附带计算好的延迟时间
            fragments = fragmenter.fragment(message, state)
        else:
            # 如果获取不到状态，降级为整段发送，无延迟
            fragments = [(message, 0.0)]
    except ImportError:
        logger.warning("未找到 OutputFragmenter 模块，降级为普通发送模式。")
        fragments = [(message, 0.0)]
    except Exception as e:
        logger.error(f"消息碎片化处理异常: {e}", exc_info=True)
        fragments = [(message, 0.0)]

    if not fragments:
        return "消息已被过滤或为空，未发送任何内容。"

    # 2. 带有真实感停顿的循环发送
    final_status = ""

    for index, (frag_text, delay) in enumerate(fragments):
        # 模拟人类打字停顿
        if delay > 0:
            logger.debug(f"模拟情绪打字停顿: {delay:.2f} 秒...")
            await asyncio.sleep(delay)

        # 构造 Action 参数
        params = {
            "message": frag_text,
            "user_id": target_id if target_type == "private" else None,
            "group_id": target_id if target_type == "group" else None,
            "detail_type": target_type
        }

        # 发布动作
        action = Action(
            action="send_message",
            params=params,
            target_platform=platform  # 指定平台适配器处理
        )

        try:
            # 设置 10 秒超时，避免 Agent 永久卡死
            response = await event_bus.dispatch_action(action, timeout=10.0)

            # 处理不同的结果状态
            if response.status == ActionStatus.OK:
                final_status = f"消息已完成发送 [{platform}] {target_type}: {target_id}"
            else:
                # 失败处理
                error_msg = response.message

                # 特殊错误：平台不存在
                if "Route not matched" in error_msg:
                    return (
                        f"发送失败: 找不到平台适配器 '{platform}'。\n"
                        f"可能原因：\n"
                        f"1. 平台名称拼写错误\n"
                        f"2. 该平台的适配器未在 main.py 中加载"
                    )
                # 如果某一个碎片发送失败，直接向 LLM 返回报错，打断后续连发
                return f"发送失败 (片段 {index + 1}/{len(fragments)}): {error_msg}"

        except asyncio.TimeoutError:
            return f"发送超时: 平台 '{platform}' 在 10 秒内没有响应 (片段 {index + 1})。"
        except Exception as e:
            return f"系统异常: 发送过程中发生错误 - {str(e)}"

    # 3. 发送成功后的等待逻辑处理
    if wait_minutes is not None and wait_minutes > 0:
        if not scheduler:
            return f"{final_status}。但警告：调度器未初始化，系统未能进入等待状态。"

        # 添加调度任务
        job = scheduler.add_job(
            _dispatch_wake_up,
            'date',
            run_date=datetime.datetime.now() + datetime.timedelta(minutes=wait_minutes),
            args=[event_bus, "等待超时"]
        )

        if agent:
            agent.wakeup_job_id = job.id
            agent.is_sleeping = True

        logger.info(f"💤 消息发送完毕，已进入等待状态。")

        return "已结束等待 | " + final_status

    # 如果模型没有传入等待参数，正常返回发送成功的文本
    return final_status


@register()
async def wait(
        scheduler: AsyncIOScheduler,
        event_bus: EventBus,
        agent: AutonomousAgent,
        duration: Optional[float] = None,
        until: Optional[str] = None,
        reason: Optional[str] = None,
) -> Optional[str]:
    """
    让系统进入等待/挂起状态。
    系统会挂起当前任务，直到指定时间到达后通过事件被唤醒。
    注意：在等待期间，如果有新的用户消息，系统依然会被打断并处理。

    Args:
        duration: 等待的分钟 (相对时间)。
        until: 等待直到具体的日期时间 (绝对时间)，ISO 格式。
        reason: 启动等待的原因。
    """

    # 1. 计算触发时间
    if until:
        try:
            # 使用 dateutil 解析自然语言或 ISO 时间
            run_date = parser.parse(until)
            # 如果解析出的时间没有时区，且当前是 Awareness 的，需处理 (这里简化，假设本地时间)
            if run_date < datetime.datetime.now():
                return f"错误: 目标时间 {until} 已经是过去式了。"
        except Exception:
            return f"错误: 无法解析时间字符串 '{until}'。请使用 ISO 格式 (YYYY-MM-DD HH:MM:SS)。"

    elif duration is not None:
        if duration <= 0:
            return "错误: 等待时长必须大于 0。"
        run_date = datetime.datetime.now() + datetime.timedelta(minutes=duration)

    else:
        return "错误: 必须提供 'duration' (分) 或 'until' (日期字符串) 其中之一。"

    # 2. 添加调度任务
    if scheduler and run_date:
        # 获取 job 对象
        job = scheduler.add_job(
            _dispatch_wake_up,
            'date',
            run_date=run_date,
            args=[event_bus, "等待超时"]
        )

        # 将 Job ID 绑定到 Agent 实例，用于后续取消
        if agent:
            agent.wakeup_job_id = job.id
            agent.is_sleeping = True

        return None
    else:
        return "系统错误: 调度器未初始化。"


@register()
async def wait_forever(
        agent: AutonomousAgent = None,
        event_bus: EventBus = None,
        reason: Optional[str] = None,
) -> Optional[str]:
    """
    [Control] 进入无限期休眠状态，直到收到外部事件唤醒。
    优先使用wait工具，尽量不使用wait_forever工具。

    Args:
        reason: 启动休眠的原因。
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

    # 广播系统状态变更
    if event_bus:
        event_bus.publish_action(Action(
            action="broadcast_log",
            params={"content": f"💤 系统进入无限期待命状态: {reason}"}
        ))

    return None


@register()
async def suppress_urge(
        reason: str,
        duration_minutes: int,
        limbic: LimbicManager = None,
        event_bus: EventBus = None
) -> str:
    """
    当你内心产生了强烈的潜意识冲动（如社交渴望、极度好奇），但你评估当前的物理现实绝对不允许你发声，决定强行忍耐时调用此工具。
    高危警告：调用此工具不会让需求消失，而是会严重消耗你的认知能量（引发精神内耗），并可能在未来导致情绪反弹或失控。

    :param reason: 你决定压抑冲动的具体心理活动和客观原因。
    :param duration_minutes: 你打算强行让自己冷静和自闭的时间（分钟）。
    """
    if not limbic:
        return "工具执行失败：无法连接到底层边缘系统。"

    # 1. 触发边缘系统物理内耗 (调用我们在第一步写的接口)
    await limbic.suppress_drive(drive_type="social", fatigue_increase=0.3)

    # 2. 向系统控制台广播心流日志，增强观测性
    if event_bus:
        event_bus.publish_action(Action(
            action="broadcast_log",
            params={
                "content": f"🛡️ [心理压抑] 强行忍耐 {duration_minutes} 分钟。原因: {reason}"}
        ))

    # 3. 执行时间层面的物理挂起
    sleep_seconds = duration_minutes * 60
    await asyncio.sleep(sleep_seconds)

    return f"已完成 {duration_minutes} 分钟的自我压制。当前状态：冲动未完全消退，且感到明显的精神疲惫和一定的焦躁感。"


@register()
async def get_active_adapters(adapters: List[Any]) -> str:
    """
    [System] 获取当前系统已加载的所有 IO 适配器列表。
    用于检查系统是否正确连接到了各个平台（如 Console, OneBot 等）。
    """
    lines = []
    for i, adapter in enumerate(adapters, 1):
        # 获取平台名称 (BaseAdapter 属性)
        name = getattr(adapter, "platform_name", "Unknown")
        # 获取类名作为辅助信息
        class_name = adapter.__class__.__name__

        lines.append(f"{name} ({class_name})")

    return "\n".join(lines)
