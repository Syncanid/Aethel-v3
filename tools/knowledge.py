# tools/knowledge.py
import logging
import os
from typing import Dict, Any
from core.tool_manager.registry import register
from core.infrastructure.config_loader import get_config
from core.infrastructure.database import Database
from core.infrastructure.api_client import GenericAPIClient
from core.memory.vector_store import VectorStore
from core.memory.ingestor import KnowledgeIngestor
from core.io.event_bus import EventBus

# 单例辅助
_ingestor = None


def _get_components():
    global _ingestor
    if not _ingestor:
        cfg = get_config()
        db = Database(cfg)
        client = GenericAPIClient(cfg)
        store = VectorStore(db, client)
        _ingestor = KnowledgeIngestor(client, store)
    return _ingestor, _ingestor.vector_store


@register()
async def import_document(
        file_path: str,
        event_bus: EventBus
) -> str:
    """
    [Knowledge] 从本地文本文件导入知识。
    Agent 可以读取指定的文档（如 txt, md），将其自动整理并存入语义记忆库。

    Args:
        file_path: 文件的本地绝对路径或相对路径。
    """
    ingestor, _ = _get_components()

    if not os.path.exists(file_path):
        return f"错误: 文件路径不存在 {file_path}"

    try:
        count = await ingestor.ingest_file(file_path, "admin_console")  # 默认归属
        return f"导入成功: 已从 {os.path.basename(file_path)} 中提取并保存了 {count} 条知识片段。"
    except Exception as e:
        return f"导入失败: {str(e)}"


@register()
async def delete_knowledge(
        memory_id: str,
        memory_type: str = "semantic"
) -> str:
    """
    [Admin] 删除指定的记忆或知识条目。
    需要先通过查询工具获取到 memory_id。

    Args:
        memory_id: 记忆的唯一 ID。
        memory_type: 记忆类型 ('core', 'episodic', 'semantic')，默认为 semantic。
    """
    _, store = _get_components()
    success = await store.delete_memory(memory_type, memory_id, "admin_console")
    if success:
        return f"记忆条目 {memory_id} 已删除。"
    return "删除失败，ID 可能不存在。"


@register()
async def edit_knowledge(
        memory_id: str,
        new_content: str,
        memory_type: str = "semantic"
) -> str:
    """
    [Admin] 修正或更新现有的知识条目。

    Args:
        memory_id: 记忆 ID。
        new_content: 新的文本内容。
    """
    _, store = _get_components()
    success = await store.update_memory_content(memory_type, memory_id, new_content, "admin_console")
    if success:
        return "知识库已更新。"
    return "更新失败。"