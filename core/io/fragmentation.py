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
        is_excited = state.dopamine > 0.7
        is_angry_or_stressed = state.cortisol > 0.7
        is_depressed = state.dopamine < 0.3 and state.serotonin < 0.4

        fragments = []

        if is_excited or is_angry_or_stressed:
            # 【激动/暴躁模式】：极度碎片化，按逗号和句号切分，甚至不发标点
            raw_parts = re.split(r'([。？！\n，,])', text)

            buffer = ""
            for part in raw_parts:
                if re.match(r'[。？！\n，,]', part):
                    buffer += part
                    # 暴躁时，经常一句话还没说完就发出去
                    if len(buffer.strip()) > 2:
                        fragments.append(buffer.strip())
                        buffer = ""
                else:
                    buffer += part
            if buffer.strip():
                fragments.append(buffer.strip())

            # 分配极短的延迟时间，模拟急促的打字连发
            return [(frag, random.uniform(0.5, 1.5)) for frag in fragments if frag]

        elif is_depressed:
            # 【抑郁/疲惫模式】：按完整句子切分，但延迟极高，仿佛打字很慢
            sentences = re.split(r'(?<=[。？！\n])\s*', text)
            sentences = [s for s in sentences if s.strip()]

            # 发送前迟疑很久
            return [(s, random.uniform(3.0, 6.0)) for s in sentences]

        else:
            # 【平静模式】：不打碎，或者只按段落(换行)打碎
            paragraphs = text.split('\n')
            paragraphs = [p for p in paragraphs if p.strip()]
            return [(p, random.uniform(1.0, 2.5)) for p in paragraphs]
