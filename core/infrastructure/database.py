import logging
import os
from typing import Dict

import aiosqlite
import chromadb
from chromadb import Settings

from core.infrastructure.config_loader import Config

logger = logging.getLogger(__name__)

# 定义 SQLite 表结构
SCHEMA_SQL = {
    "core_memory": """
                   CREATE TABLE IF NOT EXISTS core_memory
                   (
                       user_id
                       TEXT,
                       key
                       TEXT,
                       content
                       TEXT,
                       last_updated
                       REAL,
                       source
                       TEXT,
                       PRIMARY
                       KEY
                   (
                       user_id,
                       key
                   )
                       );
                   """,
    "neuro_states": """
                    CREATE TABLE IF NOT EXISTS neuro_states
                    (
                        user_id
                        TEXT
                        PRIMARY
                        KEY,
                        data_json
                        TEXT,
                        last_update
                        REAL
                    )
                    """,
    "social_users": """
                    CREATE TABLE IF NOT EXISTS social_users
                    (
                        puid
                        TEXT
                        PRIMARY
                        KEY,
                        platform
                        TEXT,
                        user_id
                        TEXT,
                        nickname
                        TEXT,
                        data_json
                        TEXT,
                        last_seen
                        REAL
                    )
                    """
}


class Database:
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.get("storage.db_path", "data/storage.db")
        self.vector_path = config.get("storage.vector_path", "data/vector_store")

        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(
            path=self.vector_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.episodic_collection = self.chroma_client.get_or_create_collection("episodic_memory")
        self.semantic_collection = self.chroma_client.get_or_create_collection("semantic_memory")

    async def init(self):
        """初始化 SQLite 表结构"""
        async with aiosqlite.connect(self.db_path) as db:
            for table, sql in SCHEMA_SQL.items():
                try:
                    await db.execute(sql)
                except Exception as e:
                    logger.error(f"初始化表 {table} 失败: {e}")
            await db.commit()
        logger.info(f"数据库已就绪: {self.db_path}")

    # --- 以下是基础操作方法，后续根据需求添加具体查询逻辑 ---

    def get_connection(self):
        """获取 SQLite 连接上下文管理器"""
        return aiosqlite.connect(self.db_path)

    async def get_core_memory(self, user_id: str) -> Dict[str, str]:
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT key, content FROM core_memory WHERE user_id=?", (user_id,))
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}
