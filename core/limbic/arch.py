# core/limbic/arch.py
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class TransmitterType(Enum):
    DOPAMINE = "dopamine"  # 动力/好奇心 (半衰期短)
    SEROTONIN = "serotonin"  # 满足感/稳定 (半衰期中)
    CORTISOL = "cortisol"  # 压力/焦虑 (半衰期长)
    OXYTOCIN = "oxytocin"  # 依恋/信任 (半衰期极长)


class DriveType(Enum):
    NONE = "none"
    SOCIAL_CONNECTION = "social_connection"  # 孤独 -> 想找人说话
    COGNITIVE_REST = "cognitive_rest"  # 疲劳 -> 想休息/拒绝任务
    CURIOSITY = "curiosity"  # 无聊 -> 想探索新事物
    SECURITY = "security"  # 恐惧 -> 想寻求确认/防御


@dataclass
class NeuroState:
    """
    神经化学状态
    """
    # 神经递质 (0.0 ~ 1.0)
    dopamine: float = 0.5
    serotonin: float = 0.5
    cortisol: float = 0.1
    oxytocin: float = 0.5

    # 生理稳态
    social_satiety: float = 0.2  # 社交饱腹感 (1.0=独处快乐, 0.0=极度孤独)
    cognitive_energy: float = 1.0  # 认知能量 (1.0=精力充沛, 0.0=宕机)

    # 元数据
    last_update: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuroState':
        return cls(**data)
