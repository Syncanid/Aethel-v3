# core/limbic/homeostasis.py
import logging
from typing import Tuple, List

from core.limbic.arch import NeuroState, DriveType

logger = logging.getLogger(__name__)


class HomeostasisSystem:
    """
    稳态系统：监控生理指标，产生内驱力 (Drives)。
    """

    def check_drives(self, state: NeuroState) -> Tuple[DriveType, float]:
        """
        检查当前状态，返回最强烈的内驱力和强度 (0.0~1.0)
        """
        drives: List[Tuple[DriveType, float]] = []

        # 1. 安全需求 (Security) - 恐惧驱动
        if state.cortisol > 0.65:
            intensity = (state.cortisol - 0.65) / 0.35
            drives.append((DriveType.SECURITY, intensity * 2.0))  # 权重加倍，恐惧是第一驱动力

        # 2. 休息需求 (Rest) - 疲劳驱动
        if state.cognitive_energy < 0.2:
            intensity = (0.2 - state.cognitive_energy) / 0.2
            drives.append((DriveType.COGNITIVE_REST, intensity * 1.5))

        # 3. 社交需求 (Social) - 孤独驱动
        # 只有在不太焦虑的情况下才会想社交
        if state.social_satiety < 0.35 and state.cortisol < 0.7:
            intensity = (0.35 - state.social_satiety) / 0.35
            # 多巴胺越高，分享欲越强
            intensity *= (0.5 + state.dopamine * 0.5)
            drives.append((DriveType.SOCIAL_CONNECTION, intensity))

        # 4. 好奇心 (Curiosity) - 无聊驱动
        if state.dopamine > 0.6 and state.cognitive_energy > 0.6:
            intensity = (state.dopamine - 0.6) / 0.4
            drives.append((DriveType.CURIOSITY, intensity))

        # 排序取最强驱动力
        if not drives:
            return DriveType.NONE, 0.0

        # 返回最强的驱动力
        drives.sort(key=lambda x: x[1], reverse=True)
        return drives[0]

    def consume_resource(self, state: NeuroState, action_type: str) -> NeuroState:
        """根据行动消耗/恢复资源"""
        if action_type == "complex_reasoning":
            # 思考消耗能量
            state.cognitive_energy = max(0.0, state.cognitive_energy - 0.1)

        elif action_type == "chat":
            # 聊天恢复社交饱腹感，但微耗能量
            state.social_satiety = min(1.0, state.social_satiety + 0.3)
            state.cognitive_energy = max(0.0, state.cognitive_energy - 0.02)

        elif action_type == "rest":
            # 休息快速回能
            state.cognitive_energy = min(1.0, state.cognitive_energy + 0.3)

        return state
