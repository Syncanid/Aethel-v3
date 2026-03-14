# core/memory/hippocampus.py
import asyncio
import json
import logging
import time
from typing import List, Dict, Set

from core.evolution.mimicry import SocialMimicry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.limbic.manager import LimbicManager
from core.memory.schema import EpisodicMemory, SemanticMemory
from core.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Hippocampus:
    def __init__(self, config: Config, limbic: LimbicManager, api_client: GenericAPIClient, database: Database,
                 agent_history: List[Dict]):
        """
        :param agent_history: 对 AutonomousAgent.history 的直接引用
        """
        self.config = config
        self.limbic = limbic
        self.api_client = api_client
        self.database = database
        self.history_ref = agent_history  # 直接持有引用
        self.vector_store = VectorStore(database, api_client)
        self.is_running = False

        # 状态追踪
        self.processed_ids: Set[int] = set()
        self.slow_lane_queue = asyncio.Queue()  # 待处理队列
        self.mimicry = SocialMimicry(config, api_client)

        # 归档配置
        self.BUFFER_LIMIT = 5  # 积攒多少条触发
        self.SILENCE_TIMEOUT = 60  # 静默多少秒触发

    async def start(self):
        """启动海马体循环"""
        self.is_running = True
        logger.info("海马体已启动...")

        # 崩溃恢复：从 WAL 加载未处理的消息
        try:
            async with self.database.get_connection() as conn:
                # 检查 wal_buffer 是否有遗留数据
                cursor = await conn.execute(
                    "SELECT event_id, content, role, metadata_json FROM wal_buffer ORDER BY created_at ASC"
                )
                rows = await cursor.fetchall()
                if rows:
                    logger.warning(f"⚡ 检测到非正常关闭，正在恢复 {len(rows)} 条未归档记忆...")
                    for row in rows:
                        event_id, content, role, meta_json = row
                        recovered_msg = {
                            "role": role,
                            "content": content,
                            "metadata": json.loads(meta_json)
                        }
                        # 注入特定的标记，避免 ID 冲突逻辑
                        recovered_msg["metadata"]["_wal_id"] = event_id
                        await self.slow_lane_queue.put(recovered_msg)
        except Exception as e:
            logger.error(f"WAL 恢复失败: {e}", exc_info=True)

        # 标记当前历史为已处理
        for msg in self.history_ref:
            self.processed_ids.add(id(msg))

        # 并发启动两个独立循环
        await asyncio.gather(
            self._ingest_loop(),
            self._dream_loop(),
            self._sleep_cycle_loop()
        )

    async def _ingest_loop(self):
        """
        [Fast Lane] 极速感知循环
        只负责从 history 中识别新消息并推入队列，不进行重型计算。
        """
        while self.is_running:
            try:
                new_msgs = self._scan_delta()
                for msg in new_msgs:
                    # 生成唯一 ID
                    wal_id = str(id(msg))
                    msg.setdefault("metadata", {})
                    msg["metadata"]["_wal_id"] = wal_id  # 注入 ID 以便后续删除

                    # WAL 落盘
                    # 必须在放入内存队列前完成，保证可靠性
                    async with self.database.get_connection() as conn:
                        await conn.execute(
                            "INSERT OR IGNORE INTO wal_buffer (event_id, content, role, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                            (
                                wal_id,
                                msg.get("content", ""),
                                msg.get("role", "unknown"),
                                json.dumps(msg.get("metadata", {})),
                                time.time()
                            )
                        )
                        await conn.commit()

                    await self.slow_lane_queue.put(msg)
            except Exception as e:
                logger.error(f"Ingest loop error: {e}", exc_info=True)

            await asyncio.sleep(2)  # 高频检查 (2s)

    async def _dream_loop(self):
        """
        [Slow Lane] 造梦循环
        负责记忆整理 和 社会化进化。
        """
        buffer = []
        last_dream_time = asyncio.get_event_loop().time()

        while self.is_running:
            try:
                # 1. 尝试从队列获取消息 (带超时，保证即使没有新消息也能检查超时归档)
                try:
                    msg = await asyncio.wait_for(self.slow_lane_queue.get(), timeout=5)
                    buffer.append(msg)
                except asyncio.TimeoutError:
                    pass

                # 2. 检查触发条件
                current_time = asyncio.get_event_loop().time()
                is_buffer_full = len(buffer) >= self.BUFFER_LIMIT
                is_timeout = (current_time - last_dream_time > self.SILENCE_TIMEOUT) and len(buffer) > 0

                if is_buffer_full or is_timeout:
                    # 触发深层处理
                    logger.info(f"💤 进入梦境处理 (Items: {len(buffer)})")

                    # 复制 buffer 避免处理中途被修改
                    processing_batch = list(buffer)

                    # 任务 A: 记忆归档
                    await self._consolidate_memory(processing_batch)

                    # 任务 B: 社会化进化
                    await self.mimicry.evolve(processing_batch)

                    # 事务完成：清理 WAL
                    # 只有当 LLM 处理完成后才删除，确保“至少一次”语义
                    wal_ids = [m["metadata"].get("_wal_id") for m in processing_batch if "_wal_id" in m["metadata"]]
                    if wal_ids:
                        async with self.database.get_connection() as conn:
                            placeholders = ",".join("?" * len(wal_ids))
                            await conn.execute(f"DELETE FROM wal_buffer WHERE event_id IN ({placeholders})", wal_ids)
                            await conn.commit()

                    buffer.clear()
                    last_dream_time = current_time

            except Exception as e:
                logger.error(f"Dream loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _sleep_cycle_loop(self):
        """
        睡眠周期：夜间执行 Episodic -> Semantic 压缩，以及图谱修剪
        """
        while self.is_running:
            # 每天凌晨 3 点执行压缩 (这里为了演示，可使用定时器或固定检测)
            now = time.localtime()
            if now.tm_hour == 3 and now.tm_min == 0:
                logger.info("🌙 进入深度睡眠周期：开始记忆维护...")

                # 1. 压缩情景记忆
                await self._compress_episodic_to_semantic()

                # 2. 自动修剪与清理知识图谱
                await self._prune_graph_edges()

                await asyncio.sleep(60)  # 避免同一分钟内重复执行

            await asyncio.sleep(30)  # 每半分钟检查一次时间

    async def _compress_episodic_to_semantic(self):
        """执行记忆压缩算法"""
        # 1. 获取所有老旧的 Episodic 记忆 (例如7天前的)
        seven_days_ago = time.time() - (7 * 86400)

        # 通过 Chroma 获取这些数据 (需遍历所有 user_id，此处简化)
        users_res = self.database.episodic_collection.get(include=["metadatas"])
        user_ids = set(m.get("user_id") for m in users_res["metadatas"] if m)

        for uid in user_ids:
            old_memories = self.database.episodic_collection.get(
                where={"user_id": uid, "created_at": {"$lt": seven_days_ago}},
                include=["documents", "ids"]
            )

            if len(old_memories["ids"]) < 5:
                continue  # 太少不值得压缩

            content_list = old_memories["documents"]
            prompt = f"""
            你是一个睡眠中的大脑。请将以下零散的短期对话情景记忆，压缩提取为1-2条关于用户的【长期的、概括性的事实或习惯】。
            原始片段：{json.dumps(content_list, ensure_ascii=False)}
            """

            resp = await self.api_client.create_chat_completion([{"role": "user", "content": prompt}])
            compressed_fact = resp["choices"][0]["message"]["content"].strip()

            # 存入 Semantic
            await self.vector_store.save_vector_memory(SemanticMemory(content=compressed_fact), uid)

            # 删除旧的 Episodic (遗忘细节)
            self.database.episodic_collection.delete(ids=old_memories["ids"])
            logger.info(f"🧠 [睡眠压缩] 提取事实：{compressed_fact}，并删除了 {len(old_memories['ids'])} 条旧细节。")

    async def _prune_graph_edges(self):
        """
        [Sleep Task] 知识图谱边缘修剪。
        清理低权重、重复或极其老旧的图谱关系，防止超级节点爆炸。
        """
        try:
            async with self.database.get_connection() as conn:
                # 1. 清理完全重复的边 (Subject-Predicate-Object 完全一致，只保留最新的一条)
                cleanup_duplicates_sql = """
                                         DELETE \
                                         FROM graph_edges
                                         WHERE id NOT IN (SELECT MAX(id) \
                                                          FROM graph_edges \
                                                          GROUP BY source, target, relation, puid) \
                                         """
                cursor = await conn.execute(cleanup_duplicates_sql)
                dup_deleted = cursor.rowcount

                # 2. 衰减所有边的权重 (模拟记忆遗忘曲线)
                # 每天衰减 5% 的权重
                await conn.execute("UPDATE graph_edges SET weight = weight * 0.95")

                # 3. 删除权重过低 (低于 0.1) 且时间超过 30 天的无效关系
                thirty_days_ago = time.time() - (30 * 86400)
                cleanup_weak_sql = """
                                   DELETE \
                                   FROM graph_edges
                                   WHERE weight < 0.1 AND timestamp < ? \
                                   """
                cursor = await conn.execute(cleanup_weak_sql, (thirty_days_ago,))
                weak_deleted = cursor.rowcount

                await conn.commit()

            logger.info(f"🕸️ [睡眠清理] 图谱维护完成: 删除了 {dup_deleted} 条重复边，{weak_deleted} 条过期弱连接边。")

        except Exception as e:
            logger.error(f"图谱修剪失败: {e}", exc_info=True)

    def _scan_delta(self) -> List[Dict]:
        """扫描增量消息 (仅在内存中操作，极快)"""
        new_msgs = []

        # 1. 扫描增量消息
        # 注意：history 是动态变化的（被修剪），但对象 ID 在内存中是唯一的
        # 我们遍历当前的 history，找出未见过的对象
        current_history_ids = set()

        # 遍历当前历史快照
        for msg in list(self.history_ref):
            msg_id = id(msg)
            current_history_ids.add(msg_id)

            if msg_id not in self.processed_ids:
                # 过滤逻辑
                metadata = msg.get("metadata", {})
                if metadata.get("ephemeral", False):
                    self.processed_ids.add(msg_id)
                    continue

                if msg.get("role") in ["user", "assistant"]:
                    new_msgs.append(msg)
                    self.processed_ids.add(msg_id)

        # 清理已不存在的消息ID (防止内存泄漏)
        self.processed_ids.intersection_update(current_history_ids)
        return new_msgs

    async def _consolidate_memory(self, buffer: List[Dict]):
        """
        调用 LLM 进行记忆提取。
        包含图谱抽取与记忆冲突消解的记忆巩固机制。
        """
        prompt = """
你是一个顶尖的认知科学家与记忆状态机。
你的任务是将短期的对话流转化为长期的事实、情景，并提取出【实体关系网络(图谱)】。

## 输出要求 (JSON)
1. `memories`: 独立的陈述事实。
   - 必须包含 `puid` (如 onebot:123456)。
   - `type`: core (核心画像), episodic (发生的事情), semantic (客观知识)。
   - `content`: 绝对独立的完整陈述。
   - `resolution_action`: 对于这条新记忆，它是全新的(ADD)，还是推翻/更新了之前的认知(UPDATE)，还是删除了之前的认知(DELETE)？默认填 ADD。
2. `graph_edges`: 提取出这句话里的主谓宾逻辑。
   - `source`: 主语实体 (如 用户名, "Aethel", "Python")
   - `target`: 宾语实体
   - `relation`: 关系动词 (如 "owns", "dislikes", "is_developing")
   - `context`: 具体语境说明
"""
        # 简化输入内容，只发送 role 和 content
        input_data = [{"role": m["role"], "content": m.get("content", "")} for m in buffer]
        input_text = json.dumps(input_data, ensure_ascii=False)

        try:
            # 使用 JSON Schema 约束输出
            schema = {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "puid": {"type": "string"},
                                "type": {"type": "string", "enum": ["core", "episodic", "semantic"]},
                                "content": {"type": "string"},
                                "resolution_action": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE"]}
                            },
                            "required": ["puid", "type", "content", "resolution_action"]
                        }
                    },
                    "graph_edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "puid": {"type": "string"},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relation": {"type": "string"},
                                "context": {"type": "string"}
                            },
                            "required": ["puid", "source", "target", "relation"]
                        }
                    }
                },
                "required": ["memories", "graph_edges"],
                "additionalProperties": False
            }

            resp = await self.api_client.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text}
                ],
                schema=schema
            )

            data = json.loads(resp["choices"][0]["message"]["content"])
            memories = data.get("memories", [])
            graph_edges = data.get("graph_edges", [])

            # 在保存记忆前，获取当前边缘系统情绪
            current_state = await self.limbic.get_state()
            mem_social = current_state.social_need
            mem_curiosity = current_state.curiosity
            mem_pressure = current_state.survival_pressure

            # 1. 存储节点与冲突消解
            for mem in memories:
                puid = mem.get("puid", "unknown")
                content = mem["content"]
                action = mem["resolution_action"]

                if mem["type"] == "core":
                    key = mem.get("key", f"extracted_{int(time.time())}")
                    await self.vector_store.save_core_memory(puid, key, content)
                else:
                    # 【核心：冲突消解逻辑】
                    if action == "UPDATE" or action == "DELETE":
                        # 利用混合检索找到要更新/删除的旧记忆
                        related_mems = await self.vector_store.search_memory(content, puid, limit=1)
                        if related_mems:
                            logger.info(f"🧠 [记忆消解] 识别到冲突，标记旧记忆状态失效: '{content}'")
                            await self.vector_store.update_memory_status(content, puid, "inactive")

                    if action != "DELETE":
                        if mem["type"] == "episodic":
                            em = EpisodicMemory(content=content, emotion_social=mem_social, emotion_curiosity=mem_curiosity, emotion_pressure=mem_pressure)
                            await self.vector_store.save_vector_memory(em, puid)
                        elif mem["type"] == "semantic":
                            sm = SemanticMemory(content=content, emotion_social=mem_social, emotion_curiosity=mem_curiosity, emotion_pressure=mem_pressure)
                            await self.vector_store.save_vector_memory(sm, puid)

            # 2. 存储图谱边
            if graph_edges:
                async with self.database.get_connection() as conn:
                    for edge in graph_edges:
                        await conn.execute(
                            "INSERT INTO graph_edges (source, target, relation, context, puid, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            (edge["source"], edge["target"], edge["relation"], edge.get("context", ""),
                             edge.get("puid", "unknown"), time.time())
                        )
                    await conn.commit()
                logger.info(f"🕸️ 提取并存储了 {len(graph_edges)} 条逻辑图谱关系。")

            if memories:
                logger.info(f"归档完成: 处理了 {len(memories)} 条记忆陈述")

        except Exception as e:
            logger.error(f"记忆转换与抽取失败: {e}", exc_info=True)

    async def review_tool_mistake(self, tool_name: str, original_args: Dict, error: str, fixed_args: Dict):
        """
        [Learning] 从工具调用错误中提取经验 (One-shot Learning)。
        """
        try:
            # 1. 构造 Prompt，让 LLM 总结规则
            prompt = f"""
Agent 在使用工具 `{tool_name}` 时失败并触发了自愈机制。
请对比原始参数和修正后的参数，结合错误信息，总结一条简短的、通用的“避坑指南”。

【错误信息】
{error}

【原始参数 (错误)】
{json.dumps(original_args, ensure_ascii=False)}

【修正参数 (正确)】
{json.dumps(fixed_args, ensure_ascii=False)}

【要求】
输出一条简短的规则，格式为：“使用 {tool_name} 时，[注意事项]...”。
例如：“使用 run_python_code 时，代码必须包含 print 输出才能被捕获。”
"""
            # 调用 LLM
            resp = await self.api_client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            rule = resp["choices"][0]["message"]["content"].strip()

            # 2. 存入语义记忆 (Semantic Memory)
            # 使用特殊的 tag 或前缀，以便 RAG 检索工具知识时更容易匹配
            memory_content = f"【工具经验】{rule}"

            # 存入向量库 (假设 puid='system' 或 'global' 代表通用知识)
            await self.vector_store.save_vector_memory(
                SemanticMemory(content=memory_content),
                user_id="global_tool_rules"
            )

            logger.info(f"🧠 [Hippocampus] 习得新经验: {rule}")
            return rule

        except Exception as e:
            logger.error(f"Failed to review tool mistake: {e}", exc_info=True)
            return None
