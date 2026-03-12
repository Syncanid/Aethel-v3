# tools/System2/memory.py
import logging

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import get_config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import Action
from core.memory.schema import SemanticMemory
from core.memory.vector_store import VectorStore
from core.tool_manager.output_cache import ToolOutputCache
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)

# 懒加载完整的记忆重型引擎
_store = None


def _get_store():
    global _store
    if not _store:
        cfg = get_config()
        db = Database(cfg)
        client = GenericAPIClient(cfg)
        _store = VectorStore(db, client)
    return _store


@register()
async def deep_hybrid_search(
        query: str,
        puid: str,
        purpose: str,
        event_bus: EventBus
) -> str:
    """
    [读取] 启动三路深度混合检索 (Vector + FTS5 + Graph 1-hop)。
    当你需要追溯很久以前的具体事件、报错信息、或需要详细的上下文来回答复杂问题时调用。
    资源消耗较高，但能找回极其精准且带逻辑关系的历史记忆。

    Args:
        query: 详细的搜索关键词或完整的自然语言问题。
        puid: 用户唯一标识 (platform:user_id)
        purpose: 你为什么要搜索这个？你想从结果中得出什么结论？
    """
    store = _get_store()
    results = await store.search_memory(query, puid)

    if not results:
        return f"深度检索完毕：未找到与 '{query}' 相关的历史记忆。"

    found = "\n".join(results)
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"🧠 S2 深度检索结果:\n{found}"}
    ))

    receipt_id, refined, final_response = await ToolOutputCache.process_tool_output(
        raw_content=found,
        purpose=purpose
    )

    return final_response


@register()
async def explore_entity_graph(
        entity_name: str,
        puid: str,
        purpose: str,
        event_bus: EventBus
) -> str:
    """
    [探索] 主动展开实体关系图谱 (GraphRAG 2-hop)。
    当你遇到一个眼熟但想不起来具体背景的专有名词、人名或项目名时，
    调用此工具可以将其“祖宗十八代”的上下游逻辑关系网全部拉出。

    Args:
        entity_name: 要探索的核心实体名称 (必须简短，如 "Aethel", "Ubuntu", "GhostFrame")。
        puid: 用户唯一标识
        purpose: 你为什么要搜索这个？你想从结果中得出什么结论？
    """
    store = _get_store()
    # 强制进行图谱多跳检索
    results = await store._search_graph_edges(query=entity_name, user_id=puid, limit=15)

    if not results:
        return f"知识图谱中未发现关于实体 '{entity_name}' 的连接网络。"

    graph_lines = [item['content'] for item in results]
    found = "\n".join(graph_lines)

    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"🕸️ 展开 [{entity_name}] 关系网:\n{found}"}
    ))

    receipt_id, refined, final_response = await ToolOutputCache.process_tool_output(
        raw_content=f"关于 '{entity_name}' 的逻辑关联如下:\n{found}",
        purpose=purpose
    )

    return final_response


@register()
async def memorize_absolute_fact(
        content: str,
        puid: str,
        event_bus: EventBus
) -> str:
    """
    [写入] 强制刻印客观事实 (Semantic Memory)。
    跳过夜间的睡眠提取，由你(S2)直接向知识库写入不可篡改的客观真理或重要规则。
    适用于你通过代码执行、文件读取后得出的关键结论（如服务器IP、重要密码指引）。

    Args:
        content: 事实的具体内容 (必须是自包含的完整陈述)。
        puid: 用户唯一标识 (如果是通用知识，可传入 'global')
    """
    store = _get_store()
    await store.save_vector_memory(SemanticMemory(content=content), puid)
    return "✅ 客观事实已成功刻印至深层知识库。"


@register()
async def correct_cognitive_error(
        conflict_query: str,
        correct_fact: str,
        puid: str,
        event_bus: EventBus
) -> str:
    """
    [纠错] 主动修正认知矛盾与错误记忆。
    当你在推理时发现历史记忆互相矛盾（如以前记录“没修好”，现在确信“已修好”），
    调用此工具可以废弃旧有的错误记忆，并覆写正确的新事实。

    Args:
        conflict_query: 用于定位那条错误记忆的模糊关键词 (例如 "网络没修好")。
        correct_fact: 应当被记住的正确事实 (例如 "网络问题已在3月7日修好")。
        puid: 用户唯一标识
    """
    store = _get_store()

    # 1. 使旧记忆失效
    await store.update_memory_status(conflict_query, puid, "inactive")

    # 2. 写入新记忆
    await store.save_vector_memory(SemanticMemory(content=correct_fact), puid)

    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"🛠️ 认知纠偏: 废弃了关于 '{conflict_query}' 的旧知，更新为: {correct_fact}"}
    ))
    return f"认知纠偏完成。关于 '{conflict_query}' 的旧记忆已被废弃，新事实已建立。"
