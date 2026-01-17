# core/memory/hippocampus.py
import asyncio
import json
import logging
import time
from typing import List, Dict

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.memory.schema import EpisodicMemory, SemanticMemory
from core.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Hippocampus:
    def __init__(self, config: Config, api_client: GenericAPIClient, database: Database):
        self.config = config
        self.api_client = api_client
        self.database = database
        self.vector_store = VectorStore(database, api_client)
        self.is_running = False

        # 归档配置
        self.BUFFER_LIMIT = 5  # 积攒多少条触发
        self.SILENCE_TIMEOUT = 300  # 静默多少秒触发

    async def start(self):
        """启动海马体循环"""
        self.is_running = True
        logger.info("🧠 海马体已启动: 正在监听记忆回响...")

        while self.is_running:
            try:
                await self._run_archival_loop()
            except Exception as e:
                logger.error(f"海马体运行异常: {e}", exc_info=True)
            await asyncio.sleep(60)  # 每分钟检查一次

    async def _run_archival_loop(self):
        """扫描活跃会话并处理"""
        # 这里简化处理，假设只有一个默认用户 "admin_console"
        # 生产环境应从 chat_logs 扫描最近活跃的用户
        user_id = "admin_console"
        group_id = "private_admin_console"  # 假设的 session key

        await self._process_session(user_id, group_id)

    async def _process_session(self, user_id: str, group_id: str):
        """处理单个会话"""
        # 1. 从 SQLite 获取未处理消息
        msgs = await self._get_unarchived_msgs(user_id)
        if not msgs: return

        # 2. 检查触发条件
        should_archive = False
        if len(msgs) >= self.BUFFER_LIMIT:
            should_archive = True
        elif time.time() - msgs[-1]['timestamp'] > self.SILENCE_TIMEOUT:
            should_archive = True

        if should_archive:
            logger.info(f"触发记忆归档 [{user_id}]: {len(msgs)} 条消息")
            await self._consolidate_memory(user_id, msgs)
            # 标记已处理 (简化逻辑: 更新最后处理时间或ID，这里暂略)

    async def _get_unarchived_msgs(self, user_id: str) -> List[Dict]:
        """(Mock) 获取未归档消息，实际应查数据库状态表"""
        async with self.database.get_connection() as conn:
            # 获取最近 10 条消息作为 Buffer 演示
            cursor = await conn.execute(
                "SELECT role, content, timestamp FROM chat_logs WHERE user_id=? ORDER BY id DESC LIMIT 10",
                (user_id,)
            )
            rows = await cursor.fetchall()
            # 转为正序
            return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows][::-1]

    async def _consolidate_memory(self, user_id: str, buffer: List[Dict]):
        """调用 LLM 转化记忆"""
        prompt = """
你是一个顶尖的认知科学家，也是 AI 的“海马体”（记忆整理中枢）。
你的任务是阅读【待处理对话】，将短期的对话流转化为长期的结构化记忆。

请遵循以下分类标准进行提取：

1. 核心记忆 (Core): 
   - 关于用户的长期事实（如：姓名、职业、喜好）。
   - 格式：key (如 `basic:name`) 和 content。

2. 情景记忆 (Episodic):
   - 记录发生的具体事件（谁做了什么）。
   - 过滤掉无效的闲聊（如“你好”），只保留有意义的交互。

3. 语义记忆 (Semantic):
   - 抽象的知识、事实或观点（如“Python 是一种语言”）。

请输出 JSON 格式:
{
  "memories": [
    {"type": "core", "key": "...", "content": "..."},
    {"type": "episodic", "content": "..."},
    {"type": "semantic", "content": "..."}
  ]
}
"""
        input_text = json.dumps(buffer, ensure_ascii=False)

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
                                "type": {"type": "string", "enum": ["core", "episodic", "semantic"]},
                                "key": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["type", "content"]
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
                if mem["type"] == "core":
                    await self.vector_store.save_core_memory(user_id, mem.get("key", "unknown"), mem["content"])
                elif mem["type"] == "episodic":
                    await self.vector_store.save_vector_memory(EpisodicMemory(content=mem["content"]), user_id)
                elif mem["type"] == "semantic":
                    await self.vector_store.save_vector_memory(SemanticMemory(content=mem["content"]), user_id)

            logger.info(f"归档完成: 生成 {len(memories)} 条记忆")

        except Exception as e:
            logger.error(f"记忆转换失败: {e}")
