# tools/memory.py
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import get_config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import Action
from core.memory.schema import SemanticMemory
from core.memory.vector_store import VectorStore
from core.tool_manager.registry import register

# 懒加载单例，避免循环导入问题
_store = None


def _get_store():
    global _store
    if not _store:
        cfg = get_config()
        # 注意：这里新建实例用于工具调用，生产环境最好通过依赖注入传进来
        db = Database(cfg)
        client = GenericAPIClient(cfg)
        _store = VectorStore(db, client)
    return _store


@register()
async def remember_core_info(
    key: str,
    value: str,
    event_bus: EventBus,
    user_id: str
) -> str:
    """
    [写入] 记住关于用户的核心信息 (Core Memory)。
    用于记录用户的长期属性、偏好、关系等结构化信息。

    Args:
        key: 信息的键名，建议格式 '类别:名称' (如 'basic:name', 'pref:food')。
        value: 具体内容 (如 '张三', '喜欢吃辣')。
    """
    store = _get_store()
    await store.save_core_memory(user_id, key, value)

    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"💾 已写入核心记忆 [{user_id}]: {key} = {value}"}
    ))
    return f"已记住: {key} 是 {value}"


@register()
async def remember_knowledge(content: str, event_bus: EventBus) -> str:
    """
    [写入] 记住一条通用的知识或事实 (Semantic Memory)。
    适用于不随时间改变的知识。

    Args:
        content: 知识的具体内容。
    """
    store = _get_store()
    await store.save_vector_memory(SemanticMemory(content=content), "admin_console")
    return "已保存到知识库。"


@register()
async def update_knowledge_status(
        content_query: str,
        status: str,
        event_bus: EventBus
) -> str:
    """
    [更新] 更新某条知识或记忆的状态。
    用于标记信息已过期、失效或已解决。
    例如：用户说“电脑修好了”，你可以将“电脑坏了”的记忆标记为 'inactive'。

    Args:
        content_query: 用于定位记忆的内容关键词 (例如 "电脑坏了")。
        status: 新状态，可选值: 'active' (有效), 'inactive' (失效/已解决)。
    """
    store = _get_store()
    # 假设用户是 admin_console
    await store.update_memory_status(content_query, "admin_console", status)

    return f"已将关于 '{content_query}' 的记忆状态更新为 {status}。"


@register()
async def recall_memory(
    query: str,
    event_bus: EventBus,
    user_id: str
) -> str:
    """
    [读取] 主动搜索记忆库。
    当你觉得之前聊过某事但不确定细节时使用。

    Args:
        query: 搜索关键词或问题。
    """
    store = _get_store()
    results = await store.search_memory(query, user_id)

    if not results:
        return "未找到相关记忆。"

    found = "\n".join(results)
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"🔍 检索结果:\n{found}"}
    ))
    return f"找到以下记忆:\n{found}"
