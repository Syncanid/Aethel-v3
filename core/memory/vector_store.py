# core/memory/vector_store.py
import asyncio
import datetime
import difflib
import json
import logging
import math
import re
import time
import uuid
from typing import List, Dict, Union, Any

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.database import Database
from core.limbic.arch import NeuroState
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

    def _rrf_merge(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        多路 RRF 融合算法
        """
        scores = {}
        content_map = {}

        for res_list in result_lists:
            for rank, item in enumerate(res_list):
                doc_id = item['id']
                # 计算 RRF 分数
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

                # 合并内容与元数据 (保留信息最全的那一份)
                if doc_id not in content_map or item.get('type') == 'vector':
                    content_map[doc_id] = item

                # 如果是图谱拉出的关联数据，给予额外的权重加成
                if item.get('type') == 'graph':
                    scores[doc_id] *= 1.2

        # 按分数降序排列
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        final_results = []
        for doc_id in sorted_ids:
            item = content_map[doc_id]
            item["rrf_score"] = scores[doc_id]
            final_results.append(item)

        return final_results

    async def _extract_entities_with_llm(self, query: str) -> List[str]:
        """
        使用 LLM 精准提取查询实体。
        """
        system_prompt = "你是一个实体提取引擎。请提取用户查询中的核心专有名词、项目名或关键事物。"

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "核心实体列表"
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }

        try:
            # 优先使用配置的 small_model (如果未配置则回退到主模型)，加快提取速度
            message = await self.api_client.create_chat_completion_once(
                messages=f"提取实体：{query}",
                system_prompt=system_prompt,
                model=self.api_client.model,
                schema=schema
            )

            content = message.get("content", "").strip()
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()

            # 解析 JSON
            data = json.loads(content)
            return data.get("entities", [])

        except Exception as e:
            logger.warning(f"LLM 实体提取失败，回退到正则切词: {e}")
            # 降级方案：保留现有的正则逻辑作为兜底
            return re.findall(r'[a-zA-Z0-9_]+|[\u4e00-\u9fa5]{2,}', query)

    async def search_graph_edges(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        """
        2-Hop 知识图谱子图检索引擎 (GraphRAG)
        提取实体 -> 命中种子节点 (Hop 1) -> 扩展邻居节点 (Hop 2) -> 距离衰减打分
        """
        graph_results = []

        # 1. 智能实体提取
        keywords = await self._extract_entities_with_llm(query)
        if not keywords: return []

        async with self.db.get_connection() as conn:
            # ==========================================
            # Hop 1: 寻找种子边 (直接命中关键词的核心实体)
            # ==========================================
            hop1_edges = {}
            seed_entities = set()

            # 动态构造 LIKE 条件
            conditions = []
            params_hop1 = [user_id]
            for kw in keywords:
                conditions.append("(source LIKE ? OR target LIKE ?)")
                params_hop1.extend([f"%{kw}%", f"%{kw}%"])

            where_clause = " OR ".join(conditions)
            sql_hop1 = f"""
                SELECT id, source, target, relation, context, weight 
                FROM graph_edges 
                WHERE puid = ? AND ({where_clause})
                ORDER BY weight DESC LIMIT ?
            """
            params_hop1.append(limit)  # 追加 LIMIT 参数

            cursor = await conn.execute(sql_hop1, params_hop1)
            rows = await cursor.fetchall()

            for r in rows:
                edge_id, source, target, relation, context, weight = r
                hop1_edges[edge_id] = {
                    "source": source, "target": target, "relation": relation,
                    "context": context, "weight": weight, "hop": 1
                }

                # 找出到底是哪个实体被命中了，将其作为 Hop 2 的扩散种子
                for kw in keywords:
                    if kw.lower() in source.lower(): seed_entities.add(source)
                    if kw.lower() in target.lower(): seed_entities.add(target)

            # ==========================================
            # Hop 2: 关系延展 (寻找种子实体的关联邻居)
            # ==========================================
            hop2_edges = {}
            if seed_entities:
                for seed in seed_entities:
                    # 查询该实体的“节点度数”（有多少条边连着它）
                    cursor = await conn.execute(
                        "SELECT COUNT(*) FROM graph_edges WHERE puid=? AND (source=? OR target=?)",
                        (user_id, seed, seed)
                    )
                    degree = (await cursor.fetchone())[0]

                    # 动态衰减：如果一个节点连着上百条边，说明它是废话节点（比如"我"），惩罚它
                    decay_factor = 0.5 * (1.0 / (1.0 + max(0, degree - 10) * 0.1))

                    # 如果惩罚太高，直接抛弃，防止爆炸
                    if decay_factor < 0.05: continue

                    sql_hop2 = """
                               SELECT id, source, target, relation, context, weight
                               FROM graph_edges
                               WHERE puid = ?
                                 AND (source = ? OR target = ?)
                               ORDER BY weight DESC LIMIT ? \
                               """
                    cursor = await conn.execute(sql_hop2, (user_id, seed, seed, 5))
                    rows2 = await cursor.fetchall()

                    for r in rows2:
                        edge_id, src, tgt, rel, ctx, w = r
                        if edge_id not in hop1_edges:
                            hop2_edges[edge_id] = {
                                "source": src, "target": tgt, "relation": rel,
                                "context": ctx, "weight": w * decay_factor, "hop": 2
                            }

            # ==========================================
            # 整合与格式化输出
            # ==========================================
            all_edges = {**hop1_edges, **hop2_edges}

            # 按照衰减后的权重倒序排列，截取前 limit 条
            sorted_edges = sorted(all_edges.items(), key=lambda x: x[1]["weight"], reverse=True)[:limit]

            for edge_id, data in sorted_edges:
                # 构建高度结构化的语义字符串供 LLM 吸收
                content = f"[{data['source']}] --({data['relation']})--> [{data['target']}]"
                if data['context']:
                    content += f" (补充事实: {data['context']})"

                graph_results.append({
                    "id": f"graph_{edge_id}",
                    "content": content,
                    "metadata": {
                        "type": "graph",
                        "source": data['source'],
                        "target": data['target'],
                        "hop": data['hop']
                    },
                    "type": "graph"
                })

        return graph_results

    async def search_memory(self, query: str, user_id: str, current_state: NeuroState = None, limit: int = 5) -> List[
        str]:
        """
        情绪依存的混合检索: Vector + FTS5 + Graph -> RRF -> Emotion Rerank
        """
        # 路一：向量检索 (ChromaDB - 擅长语意模糊匹配)
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

        # 路二：全文检索 (SQLite FTS5 BM25 - 擅长精准关键词匹配)
        text_results = []
        async with self.db.get_connection() as conn:
            sql = """
                  SELECT doc_id, content, type
                  FROM memory_fts
                  WHERE memory_fts MATCH ?
                    AND puid = ?
                  ORDER BY bm25(memory_fts) LIMIT ? \
                  """
            try:
                # FTS5 的 MATCH 语法需要处理特殊字符，这里做简单转义
                safe_query = query.replace('"', '""').replace("'", "''")
                cursor = await conn.execute(sql, (f'"{safe_query}"', user_id, limit * 2))
                rows = await cursor.fetchall()
                for r in rows:
                    text_results.append({
                        "id": r[0], "content": r[1], "metadata": {"type": r[2]}, "type": "text"
                    })
            except Exception as e:
                logger.warning(f"FTS5 检索解析跳过 (可能是查询不含明确词汇): {e}")

        # 路三：图谱检索 (1-hop 逻辑关系扩展)
        graph_results = await self.search_graph_edges(query, user_id, limit)

        # 融合：三路 RRF
        merged_results = self._rrf_merge([vector_results, text_results, graph_results])

        # 情绪维度重排 (保留原有优秀逻辑)
        if current_state and merged_results:
            def calculate_emotion_distance(item):
                meta = item.get("metadata", {})
                if not meta or item.get("type") == "graph":
                    return 0.5  # 图谱客观事实给中等距离，不严厉惩罚

                # 计算欧几里得距离: 当前情绪与记忆编码时情绪的距离
                dop_diff = current_state.dopamine - meta.get("emotion_dopamine", 0.5)
                cor_diff = current_state.cortisol - meta.get("emotion_cortisol", 0.5)
                ser_diff = current_state.serotonin - meta.get("emotion_serotonin", 0.5)

                return math.sqrt(dop_diff ** 2 + cor_diff ** 2 + ser_diff ** 2)

            # 综合得分 = 原RRF排名分数 - 情绪距离惩罚
            # 情绪状态越接近，惩罚越小，排名越靠前
            for item in merged_results:
                emo_dist = calculate_emotion_distance(item)
                item["final_score"] = item.get("rrf_score", 1.0) - (emo_dist * 0.3)

            merged_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # 格式化输出
        final_output = []

        # 优先添加 Core Memory (置顶)
        core_mems = await self.db.get_core_memory(user_id)
        for k, v in core_mems.items():
            if query in k or query in v:
                final_output.append(f"[核心档案] {k}: {v}")

        for item in merged_results[:limit]:
            if str(item.get("id")).startswith("graph_"):
                final_output.append(f"[逻辑图谱] {item['content']}")
            elif item.get('metadata') and 'created_at' in item['metadata']:
                formatted = self._format_memory_content(item['content'], item['metadata'])
                tag = "情景" if "episodic" in str(item.get('metadata', '')) else "知识"
                formatted = formatted.replace("] ", f" | {tag}] ", 1)
                final_output.append(formatted)

                # 触发异步巩固
                asyncio.create_task(self.mark_memory_accessed(item["id"], item["metadata"].get("type", "episodic")))
            else:
                final_output.append(f"[精确匹配] {item['content']}")

        return final_output

    async def mark_memory_accessed(self, memory_id: str, mem_type: str):
        """增加记忆访问计数，用于触发再巩固"""
        try:
            coll = self.episodic_coll if mem_type == "episodic" else self.semantic_coll
            res = coll.get(ids=[memory_id])
            if res['ids']:
                meta = res['metadatas'][0]
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_accessed"] = time.time()
                coll.update(ids=[memory_id], metadatas=[meta])

                # 若回忆次数超过阈值，向 EventBus 广播 RECONSOLIDATION 事件（可由 Agent 拦截）
                if meta["access_count"] % 3 == 0:
                    logger.info(f"记忆 [{memory_id}] 被频繁回忆，准备触发再巩固。")
        except Exception as e:
            logger.error(f"标记记忆访问失败: {e}")

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
