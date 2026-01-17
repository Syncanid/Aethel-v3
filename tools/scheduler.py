# tools/scheduler.py
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.tool_manager.registry import register
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, DetailType, EventSource

logger = logging.getLogger(__name__)

async def _send_reminder(event_bus: EventBus, user_id: str, content: str):
    """调度器回调函数：发送提醒事件"""
    logger.info(f"⏰ 触发提醒 [{user_id}]: {content}")
    event = OneBotEvent(
        type=EventType.MESSAGE,
        detail_type=DetailType.PRIVATE,
        sub_type="scheduler",
        source=EventSource(platform="scheduler", user_id="system"),
        message=f"【提醒】{content}",
        alt_message=f"【提醒】{content}",
        # 可以把目标用户放在 extra 里，或者通过 EventSource.group_id 区分，
        # 但这里简单的逻辑是 Agent 收到消息后，看到内容决定发给谁。
        # 更严谨的做法是 Action 直接发送消息，但为了让 Agent "感知"到提醒，我们发 Event。
        extra={"target_user_id": user_id}
    )
    event_bus.publish_event(event)

@register()
async def add_reminder(
    seconds: int,
    content: str,
    scheduler: AsyncIOScheduler, # 注入
    event_bus: EventBus,         # 注入
    user_id: str                 # 注入 (当前用户)
) -> str:
    """
    [Schedule] 设置一个一次性的倒计时提醒。

    Args:
        seconds: 多少秒后触发。
        content: 提醒内容。
    """
    scheduler.add_job(
        _send_reminder,
        'date',
        run_date=None, # 立即计算
        args=[event_bus, user_id, content],
        kwargs=None,
        coalesce=True,
        misfire_grace_time=60,
        # 使用 timezone 敏感的 date 触发，或者直接用 seconds 延迟
        # APScheduler 的 'date' trigger 不直接支持 'seconds from now' 参数，
        # 但我们可以利用 timezone.now() + timedelta，或者用 'interval' 只运行一次（复杂）。
        # 最简单是使用 loop.call_later，但为了统一管理，我们在 args 里计算 datetime。
    )
    # 修正：APScheduler add_job 如果不传 run_date，默认是 immediate。
    # 我们需要计算 run_date。
    import datetime
    run_date = datetime.datetime.now() + datetime.timedelta(seconds=seconds)

    scheduler.add_job(
        _send_reminder,
        'date',
        run_date=run_date,
        args=[event_bus, user_id, content]
    )

    return f"已设定提醒，将在 {seconds} 秒后通知你。"

@register()
async def add_cron_job(
    cron_expression: str,
    content: str,
    scheduler: AsyncIOScheduler, # 注入
    event_bus: EventBus,         # 注入
    user_id: str                 # 注入
) -> str:
    """
    [Schedule] 设置一个周期性的 Cron 任务。
    格式遵循标准 cron (分 时 日 月 周)。

    Args:
        cron_expression: 5段式 Cron 表达式 (例如 "30 8 * * *" 每天8:30)。
        content: 任务内容/提醒。
    """
    try:
        # 解析 cron 字符串
        parts = cron_expression.split()
        if len(parts) != 5:
            return "Cron 表达式格式错误，需要5个字段 (分 时 日 月 周)。"

        minute, hour, day, month, day_of_week = parts

        scheduler.add_job(
            _send_reminder,
            'cron',
            minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week,
            args=[event_bus, user_id, content]
        )
        return f"Cron 任务已设定: {cron_expression} -> {content}"
    except Exception as e:
        return f"设定失败: {str(e)}"