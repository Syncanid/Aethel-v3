# core/io/fragmentation.py
import random
import re
from typing import List, Tuple

from core.limbic.arch import NeuroState


class OutputFragmenter:
    """
    配置驱动的消息碎片化处理器：
    基于角色卡设定的语速、碎片化倾向，以及边缘系统的实时状态，智能切割文本。
    """

    def __init__(self, role_config: dict = None):
        """
        初始化时注入角色卡配置。
        如果在框架层面不方便传递实例，也可以每次调用 fragment 时将 config 作为参数传入。
        """
        constraints = role_config.get('behavior_constraints', {})
        self.config = constraints.get('fragmentation', {})

        # 从角色卡读取碎片化基础属性，如果没有则使用默认值
        self.base_typing_speed = self.config.get('typing_speed_per_char', 0.08)  # 默认打字速度：每字 0.08 秒
        self.allow_micro_fragments = self.config.get('allow_micro_fragments', False)  # 是否允许逗号级的极度碎嘴
        self.merge_actions = self.config.get('merge_actions', True)  # 是否强制将 (动作) 绑定到上一句话

    def fragment(self, text: str, state: NeuroState = None) -> List[Tuple[str, float]]:
        """
        将完整回复拆分为碎片，并计算动态延迟。
        :return: List[Tuple[消息片段, 发送此片段前的延迟秒数]]
        """
        # 清理多余空行
        text = text.strip()
        if not text:
            return []

        # 1. 智能语义切分（防 RP 动作截断）
        fragments = self._smart_split(text)

        # 2. 计算边缘系统（状态）带来的语速倍率
        speed_multiplier = 1.0
        if state:
            # 高压/紧急情况：打字变急促
            if getattr(state, 'survival_pressure', 0) > 0.7:
                speed_multiplier *= 0.6
                if self.allow_micro_fragments:
                    fragments = self._micro_split(fragments)

            # 脑力耗尽/抑郁：打字变极慢
            elif getattr(state, 'cognitive_energy', 1.0) < 0.3:
                speed_multiplier *= 2.0

            # 极度想交流：回复变快
            elif getattr(state, 'social_need', 0) > 0.8:
                speed_multiplier *= 0.8

        # 3. 动态延迟计算 (基于字数)
        result = []
        for i, frag in enumerate(fragments):
            frag = frag.strip()
            if not frag:
                continue

            char_count = len(frag)

            if i == 0:
                final_delay = 0.0
            else:
                calc_delay = char_count * self.base_typing_speed * speed_multiplier
                # 增加 10%~20% 的拟人随机波动
                final_delay = max(0.1, calc_delay * random.uniform(0.9, 1.2))

            result.append((frag, round(final_delay, 2)))

        return result

    def _smart_split(self, text: str) -> List[str]:
        """
        智能切片：按换行符切分，但如果某一行是纯动作描写，则将其合并到上一句。
        """
        raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not self.merge_actions:
            return raw_lines

        merged_lines = []
        for line in raw_lines:
            # 正则匹配：判断整行是否是被括号包裹的动作/心理活动 (支持各种全半角括号)
            is_pure_action = bool(re.match(r'^[\W]*[(（\[【].*[)）\]】][\W]*$', line))

            if is_pure_action and merged_lines:
                # 如果这是一行纯动作，且前面有话，绝对不能单独发出去，拼接到上一句末尾
                merged_lines[-1] += f" {line}"
            else:
                merged_lines.append(line)

        return merged_lines

    def _micro_split(self, fragments: List[str]) -> List[str]:
        """
        针对某些角色在激动时的“微切分”（按逗号/句号切分）。
        同时确保不会切碎括号内的内容。
        """
        micro_frags = []
        for frag in fragments:
            # 简易保护：如果当前片段包含括号动作，为了不破坏结构，直接不切
            if re.search(r'[(（\[【]', frag):
                micro_frags.append(frag)
            else:
                # 仅对纯文本按句号、逗号等进行激进切片
                parts = re.split(r'([。？！，,])', frag)
                buffer = ""
                for part in parts:
                    if re.match(r'[。？！，,]', part):
                        buffer += part
                        if len(buffer.strip()) > 2:
                            micro_frags.append(buffer.strip())
                            buffer = ""
                    else:
                        buffer += part
                if buffer.strip():
                    micro_frags.append(buffer.strip())
        return [f for f in micro_frags if f]
