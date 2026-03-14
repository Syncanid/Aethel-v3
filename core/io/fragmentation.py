# core/io/fragmentation.py
import random
import re
from typing import List, Tuple

from core.limbic.arch import NeuroState


class OutputFragmenter:
    """
    消息碎片化处理器：根据边缘系统的状态，打碎大段文本，模拟人类的发送习惯。
    """

    @staticmethod
    def fragment(text: str, state: NeuroState) -> List[Tuple[str, float]]:
        """
        将完整回复拆分为碎片。
        :return: List[Tuple[消息片段, 发送此片段前的延迟秒数]]
        """
        # 清理多余空行
        text = text.strip()
        if not text:
            return []

        # 获取状态指标
        is_excited = state.curiosity > 0.7 or state.social_need > 0.8  # 极度好奇或极度想聊天
        is_stressed = state.survival_pressure > 0.7  # 高压/服务器卡顿/报错
        is_exhausted = state.cognitive_energy < 0.3 or (
                    state.curiosity < 0.3 and state.social_need < 0.3)  # 脑力枯竭或极度内耗无聊

        fragments = []

        if is_excited or is_stressed:
            # 【激动/暴躁模式】：极度碎片化，按逗号和句号切分，甚至不发标点
            raw_parts = re.split(r'([。？！\n，,])', text)

            buffer = ""
            for part in raw_parts:
                if re.match(r'[。？！\n，,]', part):
                    buffer += part
                    # 暴躁/激动时，经常一句话还没说完就发出去
                    if len(buffer.strip()) > 2:
                        fragments.append(buffer.strip())
                        buffer = ""
                else:
                    buffer += part
            if buffer.strip():
                fragments.append(buffer.strip())

            # 分配极短的延迟时间，模拟急促的打字连发（压力大时比兴奋时打字更急促）
            delay_min = 0.3 if is_stressed else 0.5
            delay_max = 1.0 if is_stressed else 1.5
            return [(frag, random.uniform(delay_min, delay_max)) for frag in fragments if frag]

        elif is_exhausted:
            # 【疲惫/抑郁模式】：按完整句子切分，但延迟极高，仿佛打字很慢、不想说话
            sentences = re.split(r'(?<=[。？！\n])\s*', text)
            sentences = [s for s in sentences if s.strip()]

            # 发送前迟疑很久
            return [(s, random.uniform(3.0, 6.0)) for s in sentences]

        else:
            # 【平静模式】：不打碎，或者只按段落(换行)打碎
            paragraphs = text.split('\n')
            paragraphs = [p for p in paragraphs if p.strip()]
            return [(p, random.uniform(1.0, 2.5)) for p in paragraphs]
