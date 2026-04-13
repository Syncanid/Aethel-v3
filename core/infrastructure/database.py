# core/infrastructure/database.py
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
    "s2_checkpoints": """
                      CREATE TABLE IF NOT EXISTS s2_checkpoints
                      (
                          task_id
                          TEXT
                          NOT
                          NULL,
                          checkpoint_id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          timestamp
                          REAL
                          NOT
                          NULL,
                          is_stable
                          BOOLEAN
                          NOT
                          NULL,
                          data_json
                          TEXT
                          NOT
                          NULL
                      );
                      """,
    "s2_checkpoints_idx": """
                          CREATE INDEX IF NOT EXISTS idx_task_time
                              ON s2_checkpoints (task_id, timestamp DESC);
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
                    """,
    "wal_buffer": """
                  CREATE TABLE IF NOT EXISTS wal_buffer
                  (
                      event_id
                      TEXT
                      PRIMARY
                      KEY,
                      content
                      TEXT,
                      role
                      TEXT,
                      metadata_json
                      TEXT,
                      created_at
                      REAL
                  )
                  """,
    "text_search_index": """
                         CREATE TABLE IF NOT EXISTS text_search_index
                         (
                             doc_id
                             TEXT
                             PRIMARY
                             KEY,
                             content
                             TEXT,
                             puid
                             TEXT,
                             type
                             TEXT
                         )
                         """,
    "preference_store": """
                        CREATE TABLE IF NOT EXISTS preference_store
                        (
                            key
                            TEXT
                            PRIMARY
                            KEY,
                            value
                            TEXT,
                            updated_at
                            REAL
                        )
                        """,
    "graph_edges": """
                   CREATE TABLE IF NOT EXISTS graph_edges
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       source
                       TEXT
                       NOT
                       NULL,
                       target
                       TEXT
                       NOT
                       NULL,
                       relation
                       TEXT
                       NOT
                       NULL,
                       context
                       TEXT,
                       puid
                       TEXT
                       NOT
                       NULL,
                       weight
                       REAL
                       DEFAULT
                       1.0,
                       timestamp
                       REAL
                   )
                   """,
    "memory_fts": """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            doc_id UNINDEXED,
            content,
            type UNINDEXED,
            puid UNINDEXED
        )
    """,
    "trigger_fts_insert": """
                          CREATE TRIGGER IF NOT EXISTS tsi_after_insert 
        AFTER INSERT ON text_search_index
                          BEGIN
                          INSERT INTO memory_fts(doc_id, content, type, puid)
                          VALUES (new.doc_id, new.content, new.type, new.puid);
                          END;
                          """,
    "trigger_fts_delete": """
                          CREATE TRIGGER IF NOT EXISTS tsi_after_delete 
        AFTER
                          DELETE
                          ON text_search_index
                          BEGIN
                          DELETE
                          FROM memory_fts
                          WHERE doc_id = old.doc_id;
                          END;
                          """,
    "trigger_fts_update": """
                          CREATE TRIGGER IF NOT EXISTS tsi_after_update 
        AFTER
                          UPDATE ON text_search_index
                          BEGIN
                          DELETE
                          FROM memory_fts
                          WHERE doc_id = old.doc_id;
                          INSERT INTO memory_fts(doc_id, content, type, puid)
                          VALUES (new.doc_id, new.content, new.type, new.puid);
                          END;
                          """,
    "daemon_scripts": """
                      CREATE TABLE IF NOT EXISTS daemon_scripts
                      (
                          name
                          TEXT
                          PRIMARY
                          KEY,
                          code
                          TEXT
                          NOT
                          NULL,
                          status
                          TEXT
                          DEFAULT
                          'stopped',
                          created_at
                          REAL,
                          updated_at
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
                    logger.error(f"初始化表 {table} 失败: {e}", exc_info=True)
            await db.commit()
        logger.info(f"数据库已就绪: {self.db_path}")

    def get_connection(self):
        """获取 SQLite 连接上下文管理器"""
        return aiosqlite.connect(self.db_path)

    async def get_core_memory(self, user_id: str) -> Dict[str, str]:
        async with self.get_connection() as conn:
            cursor = await conn.execute("SELECT key, content FROM core_memory WHERE user_id=?", (user_id,))
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}
