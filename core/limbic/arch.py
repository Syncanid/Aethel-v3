# core/limbic/arch.py
import dataclasses
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class NeuroState:
    """
    生理状态
    """
    # 核心内驱力 (0.0 ~ 1.0，随时间或事件累积)
    social_need: float = 0.0  # 社交渴望 (随时间增加)
    curiosity: float = 0.0  # 探索欲 (闲置时增加)
    survival_pressure: float = 0.0  # 生存压力 (由服务器报错、高负载直接驱动)

    # 物理稳态
    cognitive_energy: float = 1.0  # 认知能量 (高压下降，休息恢复)

    # 元数据
    last_update: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuroState':
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        # 过滤掉旧数据库中遗留的 dopamine, cortisol 等废弃字段
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
