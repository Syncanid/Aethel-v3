# core/limbic/chemistry.py
import logging
import math
from typing import Dict

from core.limbic.arch import NeuroState, TransmitterType

logger = logging.getLogger(__name__)


class NeuroChemistry:
    """
    神经化学引擎：负责激素的代谢、相互作用和衰减。
    """

    # 半衰期 (小时)
    HALF_LIFE = {
        TransmitterType.DOPAMINE: 2.0,  # 来得快去得快
        TransmitterType.SEROTONIN: 6.0,  # 相对稳定
        TransmitterType.CORTISOL: 12.0,  # 压力很难消除
        TransmitterType.OXYTOCIN: 24.0  # 关系建立需要时间
    }

    # 基准线
    BASELINE = {
        TransmitterType.DOPAMINE: 0.4,
        TransmitterType.SEROTONIN: 0.6,
        TransmitterType.CORTISOL: 0.1,
        TransmitterType.OXYTOCIN: 0.5
    }

    def metabolize(self, state: NeuroState, current_time: float) -> NeuroState:
        """
        [代谢循环] 让状态随时间自然回归基准线
        """
        if state.last_update == 0:
            state.last_update = current_time
            return state

        delta_hours = (current_time - state.last_update) / 3600.0
        if delta_hours <= 0:
            return state

        # 1. 递质衰减
        state.dopamine = self._decay(state.dopamine, TransmitterType.DOPAMINE, delta_hours)
        state.serotonin = self._decay(state.serotonin, TransmitterType.SEROTONIN, delta_hours)
        state.cortisol = self._decay(state.cortisol, TransmitterType.CORTISOL, delta_hours)
        state.oxytocin = self._decay(state.oxytocin, TransmitterType.OXYTOCIN, delta_hours)

        # 2. 稳态消耗
        # 社交饱腹感下降 (每小时 -0.05, 约20小时耗尽)
        state.social_satiety = max(0.0, state.social_satiety - (0.05 * delta_hours))

        # 认知能量恢复 (每小时 +0.2, 睡5小时回满)
        state.cognitive_energy = min(1.0, state.cognitive_energy + (0.2 * delta_hours))

        state.last_update = current_time
        return state

    def stimulate(self, state: NeuroState, stimuli: Dict[str, float]) -> NeuroState:
        """
        [刺激反应] 非线性叠加
        """
        for key, delta in stimuli.items():
            if not hasattr(state, key): continue

            current_val = getattr(state, key)

            # 非线性增加算法
            if delta > 0:
                change = delta * (1.0 - current_val)
            else:
                change = delta * current_val

            new_val = max(0.0, min(1.0, current_val + change))
            setattr(state, key, new_val)

        # [相互作用] 高皮质醇(压力)会抑制多巴胺(快乐)
        if state.cortisol > 0.7:
            state.dopamine *= 0.8

        return state

    def _decay(self, current: float, t_type: TransmitterType, dt_hours: float) -> float:
        """计算半衰期衰减，趋向于 Base Level"""
        target = self.BASELINE[t_type]
        half_life = self.HALF_LIFE[t_type]

        diff = current - target
        # 物理学衰减公式: N(t) = N0 * (1/2)^(t/half_life)
        remaining_diff = diff * math.pow(0.5, dt_hours / half_life)

        return target + remaining_diff
