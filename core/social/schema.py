# core/social/schema.py
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class UserProfile:
    # --- 身份标识 ---
    puid: str  # 全局唯一标识
    platform: str  # 来源平台
    user_id: str  # 平台原始 ID

    # --- 认知标识 ---
    nickname: str = ""  # Agent 对用户的称呼 (由 Agent 填写，默认为空)

    # --- 关系维度 (0-100) ---
    favorability: float = 0.0  # 好感度 (喜欢/讨厌)
    trust: float = 0.0  # 信任度 (相信/怀疑)
    intimacy: float = 0.0  # 亲密度 (熟悉/陌生)

    # --- 关系描述 ---
    # 关系标签 (e.g. ["customer", "developer", "rival"])
    relationship_tags: List[str] = field(default_factory=list)
    # 文字印象 (e.g. "技术很强但脾气暴躁的开发者")
    impression: str = ""

    # --- 元数据 ---
    meta: Dict[str, Any] = field(default_factory=dict)  # 扩展数据

    # 时间戳
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(**data)
