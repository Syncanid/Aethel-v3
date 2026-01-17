# core/memory/schema.py
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class MemoryType(Enum):
    CORE = "core"  # 核心画像 (用户是谁)
    EPISODIC = "episodic"  # 情景记忆 (发生了什么)
    SEMANTIC = "semantic"  # 语义知识 (什么是XX)


@dataclass
class BaseMemory:
    content: str
    created_at: float = field(default_factory=time.time)
    source_role: str = "user"  # 'user', 'assistant', 'system'

    # 有效期 (Unix Timestamp)，None 表示永久有效
    valid_until: Optional[float] = None
    # 状态：active (有效), inactive (失效/已解决), expired (过期)
    status: str = "active"

    def to_metadata(self) -> Dict[str, Any]:
        """转换为向量数据库 metadata"""
        meta = {
            "created_at": self.created_at,
            "source": self.source_role,
            "status": self.status,
            "type": "base"
        }
        if self.valid_until is not None:
            meta["valid_until"] = self.valid_until
        return meta

@dataclass
class EpisodicMemory(BaseMemory):
    """
    情景记忆：记录事件 (Events)
    """

    def to_metadata(self) -> Dict[str, Any]:
        meta = super().to_metadata()
        meta.update({"type": MemoryType.EPISODIC.value})
        return meta


@dataclass
class SemanticMemory(BaseMemory):
    """
    语义记忆：记录事实/知识 (Facts)
    """
    keywords: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        meta = super().to_metadata()
        meta.update({
            "type": MemoryType.SEMANTIC.value,
            "keywords": ",".join(self.keywords)
        })
        return meta
