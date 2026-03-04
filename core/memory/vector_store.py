# core/memory/vector_store.py
import datetime
import difflib
import logging
import time
import uuid
from typing import List, Dict, Union, Any

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.database import Database
from core.memory.schema import EpisodicMemory, SemanticMemory

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, database: Database, api_client: GenericAPIClient):
        self.db = database
        self.api_client = api_client

        # 直接使用 database 中初始化的 collection
        self.episodic_coll = database.episodic_collection
        self.semantic_coll = database.semantic_collection

    def _format_relative_time(self, timestamp: float) -> str:
        """计算相对时间描述 (中文)"""
        if not timestamp: return "未知时间"
        diff = time.time() - timestamp
        if diff < 60:
            return "刚刚"
        elif diff < 3600:
            return f"{int(diff // 60)}分钟前"
        elif diff < 86400:
            return f"{int(diff // 3600)}小时前"
        elif diff < 604800:
            return f"{int(diff // 86400)}天前"
        else:
            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d")

    def _format_memory_content(self, doc: str, meta: Dict[str, Any]) -> str:
        """
        格式化记忆内容，附加时间锚点和状态
        格式: [状态 | 相对时间] 内容
        """
        created_at = meta.get("created_at", time.time())
        time_str = self._format_relative_time(created_at)
        status = meta.get("status", "active")
        valid_until = meta.get("valid_until")

        # 检查是否过期
        prefix_tags = []

        # 1. 过期检查
        if valid_until is not None and time.time() > valid_until:
            status = "expired"
            prefix_tags.append("已过期")

        # 2. 状态检查
        if status == "inactive":
            prefix_tags.append("已失效")

        # 3. 时间标签
        prefix_tags.append(time_str)

        tag_str = " | ".join(prefix_tags)
        return f"[{tag_str}] {doc}"

    async def save_core_memory(self, user_id: str, key: str, content: str):
        """保存核心记忆 (SQLite KV)"""
        async with self.db.get_connection() as conn:
            await conn.execute("""
                INSERT OR REPLACE INTO core_memory (user_id, key, content, last_updated, source)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, key, content, time.time(), "assistant"))
            await conn.commit()
        logger.info(f"核心记忆已更新: {key} -> {content}")

    async def save_vector_memory(self, memory: Union[EpisodicMemory, SemanticMemory], user_id: str):
        """
        保存向量记忆 (双写机制: Chroma + SQLite)
        """
        try:
            # 1. 生成向量
            embedding = await self.api_client.create_embedding(memory.content)
            if not embedding:
                logger.warning("向量生成失败，跳过存储")
                return

            collection = self.episodic_coll if isinstance(memory, EpisodicMemory) else self.semantic_coll

            # 2. [去重检测] 查询最近似的 1 条记忆
            # 我们假设如果语义距离极近 (< 0.15) 且 文本相似度高，则是重复
            existing = collection.query(
                query_embeddings=[embedding],
                n_results=1,
                where={"user_id": user_id}
            )

            is_duplicate = False
            duplicate_id = None
            existing_doc = None

            # 检查是否有返回结果
            if existing['ids'] and existing['ids'][0]:
                existing_id = existing['ids'][0][0]
                existing_doc = existing['documents'][0][0]
                existing_dist = existing['distances'][0][0]  # Chroma 默认 L2 距离

                # 判定标准：向量距离很近 (根据模型调整，一般 < 0.2 表示非常相似)
                # 且 文本重合度高 (避免“我喜欢猫”和“我不喜欢猫”向量很近但意思相反的情况)
                if existing_dist < 0.2:
                    # 计算文本相似度 (SequenceMatcher)
                    similarity = difflib.SequenceMatcher(None, memory.content, existing_doc).ratio()
                    if similarity > 0.85:
                        is_duplicate = True
                        duplicate_id = existing_id
                        logger.info(f"检测到重复记忆 (相似度 {similarity:.2f})，触发合并策略。")

            target_id = None

            # 3. 分支处理
            if is_duplicate:
                target_id = duplicate_id
                # 策略 A: 语义记忆 -> 合并关键词，更新时间
                if isinstance(memory, SemanticMemory):
                    # 获取旧的 metadata
                    old_meta = existing['metadatas'][0][0]

                    # 合并关键词 (去重)
                    old_keywords = old_meta.get("keywords", "").split(",")
                    new_keywords = memory.keywords
                    merged_keywords = list(set([k for k in old_keywords if k] + new_keywords))

                    # 更新 metadata
                    old_meta["keywords"] = ",".join(merged_keywords)
                    old_meta["last_accessed"] = time.time()  # 刷新活跃时间

                    # 如果新内容更长/更详细，替换旧内容；否则保留旧内容
                    final_content = existing_doc
                    if len(memory.content) > len(existing_doc) + 10:  # 显著更长
                        final_content = memory.content
                        # 需要重新更新向量吗？如果内容变了最好更新，但为了节省资源，
                        # 如果是 update 操作，Chroma 需要传入 embedding。
                        # 这里我们复用本次生成的 embedding
                        collection.update(
                            ids=[duplicate_id],
                            embeddings=[embedding],
                            documents=[final_content],
                            metadatas=[old_meta]
                        )
                        # [同步更新 SQLite]
                        async with self.db.get_connection() as conn:
                            await conn.execute("UPDATE text_search_index SET content=? WHERE doc_id=?",
                                               (final_content, target_id))
                            await conn.commit()
                        logger.info(f"语义记忆已更新 (内容增强): {final_content[:20]}...")
                    else:
                        # 仅更新 metadata
                        collection.update(
                            ids=[duplicate_id],
                            metadatas=[old_meta]
                        )
                        logger.info(f"语义记忆已更新 (仅刷新热度): {final_content[:20]}...")

                # 策略 B: 情景记忆 -> 仅刷新时间，忽略重复
                # (同一个事件不需要存两次)
                else:
                    logger.info("重复的情景记忆，已跳过存储。")
                    return

            else:
                # 4. 无重复 -> 正常写入
                target_id = str(uuid.uuid4())
                metadata = memory.to_metadata()
                metadata["user_id"] = user_id

                collection.add(
                    embeddings=[embedding],
                    documents=[memory.content],
                    metadatas=[metadata],
                    ids=[target_id]
                )

                # [同步写入 SQLite]
                mem_type = "semantic" if isinstance(memory, SemanticMemory) else "episodic"
                async with self.db.get_connection() as conn:
                    await conn.execute(
                        "INSERT INTO text_search_index (doc_id, content, puid, type) VALUES (?, ?, ?, ?)",
                        (target_id, memory.content, user_id, mem_type)
                    )
                    await conn.commit()

                logger.debug(f"向量记忆已保存: {memory.content}")

        except Exception as e:
            logger.error(f"向量存储失败: {e}", exc_info=True)

    def _rrf_merge(self, list_a: List[Dict], list_b: List[Dict], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) 算法
        将两路检索结果合并，分数公式: score = 1 / (k + rank)
        """
        scores = {}
        content_map = {}

        # 处理列表 A (向量结果 - 优先，因为它带有 Metadata)
        for rank, item in enumerate(list_a):
            doc_id = item['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            content_map[doc_id] = item

        # 处理列表 B (文本结果)
        for rank, item in enumerate(list_b):
            doc_id = item['id']
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            # 只有当该 ID 不在 map 中时才添加 (优先保留向量检索返回的完整对象)
            if doc_id not in content_map:
                content_map[doc_id] = item

        # 按分数降序排列
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [content_map[doc_id] for doc_id in sorted_ids]

    async def search_memory(self, query: str, user_id: str, limit: int = 5) -> List[str]:
        """
        混合检索 (Hybrid Search): 向量检索 + 关键词检索(SQL LIKE) -> RRF 融合
        """

        # 1. 向量检索 (Vector Search) - 擅长语意模糊匹配
        vector_results = []
        query_vec = await self.api_client.create_embedding(query)
        if query_vec:
            for coll in [self.episodic_coll, self.semantic_coll]:
                res = coll.query(query_embeddings=[query_vec], n_results=limit * 2, where={"user_id": user_id})
                if res['ids']:
                    for i, doc_id in enumerate(res['ids'][0]):
                        vector_results.append({
                            "id": doc_id,
                            "content": res['documents'][0][i],
                            "metadata": res['metadatas'][0][i],
                            "type": "vector"
                        })

        # 2. 文本检索 (Text Search) - 擅长精确匹配 (ID, 错误码, 专有名词)
        text_results = []
        async with self.db.get_connection() as conn:
            # 使用 LIKE 进行包含匹配
            sql = "SELECT doc_id, content, type FROM text_search_index WHERE puid=? AND content LIKE ? LIMIT ?"
            cursor = await conn.execute(sql, (user_id, f"%{query}%", limit * 2))
            rows = await cursor.fetchall()
            for r in rows:
                text_results.append({
                    "id": r[0],
                    "content": r[1],
                    "metadata": {},  # 文本检索可能丢失 metadata，RRF 阶段会尝试互补
                    "type": "text"
                })

        # 3. 结果融合 (Fusion)
        merged_results = self._rrf_merge(vector_results, text_results)

        # 4. 格式化输出
        final_output = []

        # 优先添加 Core Memory (置顶)
        core_mems = await self.db.get_core_memory(user_id)
        for k, v in core_mems.items():
            if query in k or query in v:
                final_output.append(f"[核心档案] {k}: {v}")

        for item in merged_results[:limit]:
            # 如果有 metadata 则使用高级格式化 (显示时间)
            if item.get('metadata'):
                formatted = self._format_memory_content(item['content'], item['metadata'])
                tag = "情景" if "episodic" in str(item.get('metadata', '')) else "知识"  # 简易判断
                # 覆盖 tag
                formatted = formatted.replace("] ", f" | {tag}] ", 1)  # 插入类型标签
            else:
                # 纯文本回退格式
                formatted = f"[精确匹配] {item['content']}"

            final_output.append(formatted)

        return final_output

    async def update_memory_status(self, content_query: str, user_id: str, new_status: str):
        """
        根据内容模糊匹配更新记忆状态 (用于标记失效)
        注意：ChromaDB 的 update 需要 ID，这里简化为先 query 再 update，生产环境应存储 ID
        """
        query_vec = await self.api_client.create_embedding(content_query)
        if not query_vec: return

        # 仅搜索语义记忆进行状态更新
        results = self.semantic_coll.query(
            query_embeddings=[query_vec], n_results=1, where={"user_id": user_id}
        )

        if results['ids'] and results['ids'][0]:
            target_id = results['ids'][0][0]
            current_meta = results['metadatas'][0][0]

            # 更新 metadata
            current_meta["status"] = new_status

            self.semantic_coll.update(
                ids=[target_id],
                metadatas=[current_meta]
            )
            logger.info(f"记忆状态已更新为 {new_status}: {content_query}")

    async def delete_memory(self, memory_type: str, memory_id: str, user_id: str) -> bool:
        """删除指定记忆"""
        try:
            if memory_type == "core":
                async with self.db.get_connection() as conn:
                    await conn.execute("DELETE FROM core_memory WHERE user_id=? AND key=?", (user_id, memory_id))
                    await conn.commit()
            elif memory_type in ["episodic", "semantic"]:
                collection = self.episodic_coll if memory_type == "episodic" else self.semantic_coll
                collection.delete(ids=[memory_id], where={"user_id": user_id})
                # [同步删除 SQLite]
                async with self.db.get_connection() as conn:
                    await conn.execute("DELETE FROM text_search_index WHERE doc_id=?", (memory_id,))
                    await conn.commit()
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}", exc_info=True)
            return False

    async def update_memory_content(self, memory_type: str, memory_id: str, new_content: str, user_id: str) -> bool:
        """更新记忆内容"""
        try:
            if memory_type == "core":
                # core memory id is the key
                await self.save_core_memory(user_id, memory_id, new_content)
            elif memory_type in ["episodic", "semantic"]:
                collection = self.episodic_coll if memory_type == "episodic" else self.semantic_coll
                # 更新 Document，Chroma 需要重新 embedding
                new_embedding = await self.api_client.create_embedding(new_content)
                if new_embedding:
                    collection.update(
                        ids=[memory_id],
                        embeddings=[new_embedding],
                        documents=[new_content]
                    )
                    # [同步更新 SQLite]
                    async with self.db.get_connection() as conn:
                        await conn.execute("UPDATE text_search_index SET content=? WHERE doc_id=?",
                                           (new_content, memory_id))
                        await conn.commit()
            return True
        except Exception as e:
            logger.error(f"更新记忆失败: {e}", exc_info=True)
            return False

    async def list_memories(self, memory_type: str, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """分页列出记忆 (用于 WebUI 管理)"""
        results = []
        if memory_type == "core":
            async with self.db.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT key, content, last_updated FROM core_memory WHERE user_id=? LIMIT ? OFFSET ?",
                    (user_id, limit, offset)
                )
                rows = await cursor.fetchall()
                for r in rows:
                    results.append({"id": r[0], "content": r[1], "timestamp": r[2], "type": "core"})

        elif memory_type in ["episodic", "semantic"]:
            collection = self.episodic_coll if memory_type == "episodic" else self.semantic_coll
            # Chroma 的 get 方法支持 limit/offset
            data = collection.get(
                where={"user_id": user_id},
                limit=limit,
                offset=offset,
                include=["metadatas", "documents"]
            )
            for i, doc in enumerate(data["documents"]):
                meta = data["metadatas"][i]
                results.append({
                    "id": data["ids"][i],
                    "content": doc,
                    "timestamp": meta.get("created_at"),
                    "status": meta.get("status", "active"),
                    "type": memory_type
                })
        return results
