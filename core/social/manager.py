# core/social/manager.py
import json
import logging
import time
from typing import Optional, List

from core.infrastructure.database import Database
from core.social.schema import UserProfile

logger = logging.getLogger(__name__)


class UserManager:
    def __init__(self, database: Database):
        self.db = database
        self._cache: dict[str, UserProfile] = {}

    async def initialize(self):
        """初始化社交数据库"""
        async with self.db.get_connection() as conn:
            await conn.execute("""
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
                               """)
            await conn.commit()

    async def get_user(self, puid: str) -> Optional[UserProfile]:
        """获取用户 (Read Only)"""
        if puid in self._cache:
            return self._cache[puid]

        async with self.db.get_connection() as conn:
            cursor = await conn.execute("SELECT data_json FROM social_users WHERE puid=?", (puid,))
            row = await cursor.fetchone()
            if row:
                try:
                    profile = UserProfile.from_dict(json.loads(row[0]))
                    self._cache[puid] = profile
                    return profile
                except Exception as e:
                    logger.error(f"用户数据解析失败 {puid}: {e}", exc_info=True)
        return None

    async def list_users(self, limit: int = 20, offset: int = 0) -> List[UserProfile]:
        """
        获取用户列表（支持分页）。
        按最后活跃时间 (last_seen) 倒序排列。
        """
        async with self.db.get_connection() as conn:
            sql = "SELECT data_json FROM social_users ORDER BY last_seen DESC LIMIT ? OFFSET ?"
            cursor = await conn.execute(sql, (limit, offset))
            rows = await cursor.fetchall()

            profiles = []
            for row in rows:
                try:
                    profiles.append(UserProfile.from_dict(json.loads(row[0])))
                except Exception as e:
                    logger.error(f"用户数据解析失败: {e}", exc_info=True)
            return profiles

    async def save_user(self, profile: UserProfile):
        """保存/更新用户 (Write)"""
        profile.last_seen = time.time()
        self._cache[profile.puid] = profile

        async with self.db.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO social_users 
                (puid, platform, user_id, nickname, data_json, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.puid,
                    profile.platform,
                    profile.user_id,
                    profile.nickname,
                    json.dumps(profile.to_dict(), ensure_ascii=False),
                    profile.last_seen
                )
            )
            await conn.commit()
