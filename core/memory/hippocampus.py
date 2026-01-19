# core/memory/hippocampus.py
import asyncio
import json
import logging
from typing import List, Dict, Set

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
        self.processed_ids: Set[int] = set()  # 记录已处理的消息对象ID
        self.buffer: List[Dict] = []  # 待归档的临时缓冲区
        self.last_activity_time = 0

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

        while self.is_running:
            try:
                await self._scan_and_consolidate()
            except Exception as e:
                logger.error(f"海马体运行异常: {e}", exc_info=True)
            await asyncio.sleep(10)  # 检查频率提高，因为直接读内存开销很小

    async def _scan_and_consolidate(self):
        """扫描历史记录并处理"""
        current_time = asyncio.get_event_loop().time()
        new_msgs = []

        # 1. 扫描增量消息
        # 注意：history 是动态变化的（被修剪），但对象 ID 在内存中是唯一的
        # 我们遍历当前的 history，找出未见过的对象
        current_history_ids = set()

        for msg in list(self.history_ref):  # 浅拷贝防止迭代时修改
            msg_id = id(msg)
            current_history_ids.add(msg_id)

            # 过滤掉系统提示和工具原始输出（通常只关注对话流）
            # 根据需求，这里保留 user 和 assistant，以及重要的 tool 结果
            if msg_id not in self.processed_ids:
                # 只关注有内容的交互，忽略纯工具调用结果以免干扰语义
                if msg.get("role") in ["user", "assistant"]:
                    new_msgs.append(msg)
                    self.processed_ids.add(msg_id)
                    self.last_activity_time = current_time

        # 清理已修剪的消息 ID，防止 Set 无限增长
        self.processed_ids.intersection_update(current_history_ids)

        # 2. 加入缓冲区
        if new_msgs:
            self.buffer.extend(new_msgs)
            logger.debug(f"海马体捕获 {len(new_msgs)} 条新消息")

        # 3. 检查触发条件
        should_archive = False
        if len(self.buffer) >= self.BUFFER_LIMIT:
            should_archive = True
        elif len(self.buffer) > 0 and (current_time - self.last_activity_time > self.SILENCE_TIMEOUT):
            should_archive = True

        if should_archive:
            logger.info(f"触发记忆归档: {len(self.buffer)} 条消息")
            await self._consolidate_memory(list(self.buffer))
            self.buffer.clear()

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
- 组合唯一标识符 UID: `{platform}:{user_id}` (例如 `onebot:123456`)。

## 任务要求
1. 身份识别：对于每一条提取出的记忆，必须明确它属于哪个 UID。
2. 绝对事实化：
   - 生成的 `content` 必须是自包含的，即使脱离当前对话上下文也能被理解。
   - 错误示例："他喜欢吃苹果" (他是谁？)
   - 正确示例："用户(onebot:123456) 喜欢吃苹果" 或 "User[Axw] is a Python developer."
3. 分类提取：
   - Core (核心): 用户的固有属性（姓名、性格、职业、长期偏好）。
   - Episodic (情景): 发生了什么重要事件（时间、地点、人物、结果）。
   - Semantic (语义): 通用的世界知识或事实（不依附于特定用户的知识）。

## 输出 Schema
请输出 JSON 对象，包含 `memories` 列表。每个 memory 必须包含 `uid` 字段。
对于 Semantic 记忆，如果它不属于特定用户（是通用知识），`uid` 可填 "global"。
"""
        # 简化输入内容，只发送 role 和 content
        input_data = [{"role": m["role"], "content": m["content"]} for m in buffer]
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
                                "uid": {
                                    "type": "string",
                                    "description": "The unique user ID (platform:user_id) this memory belongs to, or 'global'."
                                },
                                "type": {"type": "string", "enum": ["core", "episodic", "semantic"]},
                                "key": {"type": "string", "description": "Only for core memory (e.g. 'basic:name')"},
                                "content": {"type": "string", "description": "Absolute fact statement."}
                            },
                            "required": ["uid", "type", "content"]
                        }
                    }
                },
                "required": ["memories"]
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
                uid = mem.get("uid", "unknown")
                content = mem["content"]

                # 执行存储
                if mem["type"] == "core":
                    # Core memory 依然需要 key 来覆盖旧值
                    key = mem.get("key", "misc")
                    await self.vector_store.save_core_memory(uid, key, content)
                elif mem["type"] == "episodic":
                    await self.vector_store.save_vector_memory(EpisodicMemory(content=content), uid)
                elif mem["type"] == "semantic":
                    await self.vector_store.save_vector_memory(SemanticMemory(content=content), uid)

            if memories:
                logger.info(f"归档完成: 为 {len(set(m['uid'] for m in memories))} 位用户生成了 {len(memories)} 条记忆")

        except Exception as e:
            logger.error(f"记忆转换失败: {e}")
