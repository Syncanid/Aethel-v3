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

    # --- 正交关系矩阵 ---
    # 情感极性轴: -100(极其厌恶) 到 100(极度喜爱)
    favorability: float = 0.0
    # 交互深度轴: 0(完全陌生) 到 100(知根知底)
    intimacy: float = 0.0
    # 信任轴: 0(警惕) 到 100(绝对信任)
    trust: float = 0.0

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

    def __post_init__(self):
        """物理边界锁：防止数值溢出导致的人设崩塌"""
        self.favorability = max(-100.0, min(100.0, float(self.favorability)))
        self.intimacy = max(0.0, min(100.0, float(self.intimacy)))
        self.trust = max(0.0, min(100.0, float(self.trust)))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(**data)


@dataclass
class GroupProfile:
    """群聊/环境档案：控制 Agent 的公共面具"""
    # --- 身份标识 ---
    group_id: str  # 全局唯一标识 (e.g., group_onebot:654321)
    platform: str

    # --- 环境属性 ---
    name: str = ""

    # 群环境熟悉度: 0(完全陌生的新环境，需潜水观察) 到 100(主场，可肆意妄为)
    familiarity: float = 0.0

    # 群氛围标签 (e.g., "技术硬核", "水群闲聊", "游戏开荒")
    vibe: str = ""

    # --- 元数据 ---
    meta: Dict[str, Any] = field(default_factory=dict)

    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self):
        self.familiarity = max(0.0, min(100.0, float(self.familiarity)))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroupProfile':
        return cls(**data)
