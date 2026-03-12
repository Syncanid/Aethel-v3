# tools/System1/memory.py
import logging

from core.infrastructure.config_loader import get_config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import Action
from core.kernel.agent import AutonomousAgent
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)

# 轻量级懒加载，S1 只需要直接读 SQLite，不需要加载沉重的向量库
_db = None


def _get_db() -> Database:
    global _db
    if not _db:
        cfg = get_config()
        _db = Database(cfg)
    return _db


@register()
async def peek_core_memory(puid: str) -> str:
    """
    [读取] 极速瞥一眼用户的核心档案 (Core Memory)。
    当你突然忘记对话者的基本信息、称呼、特殊偏好或忌讳时，调用此工具。
    耗时极短，只返回结构化的基础设定，不会召回复杂的历史事件。

    Args:
        puid: 用户唯一标识 (platform:user_id)
    """
    db = _get_db()
    core_mems = await db.get_core_memory(puid)

    if not core_mems:
        return "该用户目前没有核心档案记录。"

    formatted = "\n".join([f"- {k}: {v}" for k, v in core_mems.items()])
    return f"[{puid} 的核心档案]\n{formatted}"


@register()
async def shift_attention(
        new_interest: str,
        reason: str,
        agent: AutonomousAgent
) -> str:
    """
    [自我调节] 迅速转移你的注意力焦点。
    当你在聊天中感知到话题发生了显著改变（例如从“编程”转到了“游戏”），
    调用此工具可以让底层的感知过滤网开始放行新话题的信息。

    Args:
        new_interest: 新的兴趣焦点（简短自然语言，如“网络安全”、“闲聊”）。
        reason: 转移注意力的原因。
    """
    if not new_interest or not new_interest.strip():
        return "错误: 注意力焦点不能为空。"

    try:
        if hasattr(agent, "attention") and hasattr(agent.attention, "update_interest"):
            await agent.attention.update_interest(new_interest)
            logger.info(f"⚡ [S1 Attention Shift] 焦点已转移: {new_interest} (原因: {reason})")
            return f"系统底层的注意力焦点已切换至 '{new_interest}'。"
        else:
            return "错误: Agent 核心未挂载注意力模块。"
    except Exception as e:
        logger.error(f"转移注意力失败: {e}", exc_info=True)
        return f"系统错误: 无法转移注意力。"


@register()
async def mark_important_moment(
        reason: str,
        event_bus: EventBus
) -> str:
    """
    [标记] 模拟大脑杏仁核，给当前的对话片段打上“高亮标记”。
    当你觉得用户刚刚说的话极其重要（比如许下承诺、表露强烈情感、给出重要密码或地址）时调用。
    这会通知后台的海马体在夜间睡眠时，优先将这段对话转化为永久的核心记忆。

    Args:
        reason: 为什么觉得这句话很重要？
    """
    # 通过事件总线向海马体发送“情绪高亮”信号
    event_bus.publish_action(Action(
        action="memory_highlight",
        params={"reason": reason, "timestamp": "now"}
    ))

    # 也向控制台/日志输出一下
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"✨ [S1 潜意识] 已标记当前瞬间为重要时刻 (原因: {reason})"}
    ))

    return "已通知海马体重点关注当前对话。"
