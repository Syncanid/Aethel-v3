# core/memory/hippocampus.py
import asyncio
import json
import logging
from typing import List, Dict, Set

from core.evolution.mimicry import SocialMimicry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.memory.schema import EpisodicMemory, SemanticMemory
from core.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Hippocampus:
    def __init__(self, config: Config, api_client: GenericAPIClient, database: Database, agent_history: List[Dict]):
        """
        :param agent_history: 对 AutonomousAgent.history 的直接引用
        """
        self.config = config
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

        # 初始标记：启动时已存在的消息不重复处理（可选，根据需求决定是否追溯）
        # 这里选择标记当前所有为已处理，只关注新增的，避免启动时重复归档旧梦
        for msg in self.history_ref:
            self.processed_ids.add(id(msg))

        # 并发启动两个独立循环
        await asyncio.gather(
            self._ingest_loop(),
            self._dream_loop()
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
                    await self.slow_lane_queue.put(msg)
            except Exception as e:
                logger.error(f"Ingest loop error: {e}")

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

                    # 任务 A: 记忆归档
                    await self._consolidate_memory(list(buffer))

                    # 任务 B: 社会化进化 (Mimicry)
                    await self.mimicry.evolve(list(buffer))

                    buffer.clear()
                    last_dream_time = current_time

            except Exception as e:
                logger.error(f"Dream loop error: {e}")
                await asyncio.sleep(5)

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
        """
        prompt = """
你是一个顶尖的认知科学家，也是 AI 的“海马体”（记忆整理中枢）。
你的任务是将短期的对话流转化为长期的、结构化的、高密度的记忆。

## 输入说明
输入是一段 JSON 格式的对话日志。
- `user` 角色消息通常包含 `[Event Received]` 和一段 JSON 数据。
- 你必须从该 JSON 数据中提取用户信息：`platform` 和 `user_id`。
- 组合唯一标识符 PUID: `{platform}:{user_id}` (例如 `onebot:123456`)。

## 任务要求
1. 身份识别：对于每一条提取出的记忆，必须明确它属于哪个 PUID。
2. 绝对事实化：
   - 生成的 `content` 必须是自包含的，即使脱离当前对话上下文也能被理解。
   - 错误示例："他喜欢吃苹果" (他是谁？)
   - 正确示例："用户(onebot:123456) 喜欢吃苹果" 或 "User[Axw] is a Python developer."
3. 分类提取：
   - Core (核心): 用户的固有属性（姓名、性格、职业、长期偏好）。
   - Episodic (情景): 发生了什么重要事件（时间、地点、人物、结果）。
   - Semantic (语义): 通用的世界知识或事实（不依附于特定用户的知识）。

## 输出 Schema
请输出 JSON 对象，包含 `memories` 列表。每个 memory 必须包含 `puid` 字段。
对于 Semantic 记忆，如果它不属于特定用户（是通用知识），`puid` 可填 "global"。
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
                                "puid": {
                                    "type": "string",
                                    "description": "The unique user ID (platform:user_id) this memory belongs to, or 'global'."
                                },
                                "type": {"type": "string", "enum": ["core", "episodic", "semantic"]},
                                "key": {"type": "string", "description": "Only for core memory (e.g. 'basic:name')"},
                                "content": {"type": "string", "description": "Absolute fact statement."}
                            },
                            "required": ["puid", "type", "key", "content"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["memories"],
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

            for mem in memories:
                puid = mem.get("puid", "unknown")
                content = mem["content"]

                # 执行存储
                if mem["type"] == "core":
                    # Core memory 依然需要 key 来覆盖旧值
                    key = mem.get("key", "misc")
                    await self.vector_store.save_core_memory(puid, key, content)
                elif mem["type"] == "episodic":
                    await self.vector_store.save_vector_memory(EpisodicMemory(content=content), puid)
                elif mem["type"] == "semantic":
                    await self.vector_store.save_vector_memory(SemanticMemory(content=content), puid)

            if memories:
                logger.info(f"归档完成: 为 {len(set(m['puid'] for m in memories))} 位用户生成了 {len(memories)} 条记忆")

        except Exception as e:
            logger.error(f"记忆转换失败: {e}")

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
                puid="global_tool_rules"
            )

            logger.info(f"🧠 [Hippocampus] 习得新经验: {rule}")
            return rule

        except Exception as e:
            logger.error(f"Failed to review tool mistake: {e}")
            return None
