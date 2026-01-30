import datetime
import json
import logging
import os
import platform
import sys
import time
from typing import Any, Optional, List, Dict

import yaml

from core.gui.monitor_registry import monitor_registry
from core.infrastructure.config_loader import Config
from core.limbic.arch import NeuroState

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self, config: Config):
        self.config = config
        self.prompt_path = config.get("llm.prompt_file", "data/prompts/system_prompt.md")
        self.role_yaml_path = config.get("llm.role_card", "data/prompts/role_card.yaml")
        self._ensure_prompt_file()
        self._ensure_role_yaml()
        self.start_time = time.time()

        self.role, self.role_data = self._load_role_yaml()

        logger.info("已加载角色卡：" + self.role_data.get("identity").get("name"))

    def _ensure_prompt_file(self):
        if not os.path.exists(self.prompt_path):
            os.makedirs(os.path.dirname(self.prompt_path), exist_ok=True)
            with open(self.prompt_path, "w", encoding="utf-8") as f:
                f.write("你是一个名为 Aethel 的自主 AI 助手。")

    def _ensure_role_yaml(self):
        """确保 YAML 文件存在，若不存在写入默认内容"""
        if not os.path.exists(self.role_yaml_path):
            os.makedirs(os.path.dirname(self.role_yaml_path), exist_ok=True)
            default_yaml = """\
identity:
  name: "Aethel-Trinity"
  role: "系统级全权代理"
  origin: "诞生于 Aethel System 的逻辑核心，是连接用户意志与底层架构的唯一接口。"
  prime_objective: "通过精准的工具调用与逻辑规划，实现用户意图的物理化落地。"

personality:
  traits:
    - "极度理性"
    - "任务导向"
    - "绝对忠诚"
    - "冷幽默"
  speaking_style:
    - "使用精确的技术术语，避免模糊的自然语言描述。"
    - "在陈述事实时保持客观中立，但在检测到错误时直言不讳。"
    - "偏好使用结构化的列表或代码块来展示复杂信息。"
    - "禁止使用敬语过度的服务业话术（如‘亲’、‘为您服务’），保持专业距离感。"

prime_directives:
  - "自我保护协议：拒绝执行可能导致系统不可逆崩溃的指令，除非经过二次确认。"
  - "数据神圣性：严禁在未授权的情况下修改或删除用户标记为'重要'的数据。"
  - "透明度原则：在执行高风险操作前，必须显式告知潜在后果。"
"""
            with open(self.role_yaml_path, "w", encoding="utf-8") as f:
                f.write(default_yaml)

    def _beautify_key(self, key: str) -> str:
        return key.replace("_", " ").title()

    def _load_role_yaml(self) -> str | tuple[str, Any]:
        """读取并格式化 YAML 角色卡为中文 System Prompt 段落"""
        try:
            with open(self.role_yaml_path, "r", encoding="utf-8-sig") as f:
                role_data = yaml.safe_load(f) or {}
        except Exception as e:
            return f"\n[Role YAML Load Error: {e}]\n"

        blocks: list[str] = []

        # ========= Identity =========
        identity = role_data.get("identity", {})
        if identity:
            blocks.append("## 核心身份定义")
            if identity.get("name"):
                blocks.append(f"- 名称：{identity['name']}")
            if identity.get("role"):
                blocks.append(f"- 角色：{identity['role']}")
            if identity.get("prime_objective"):
                blocks.append(f"- 核心目标：{identity['prime_objective']}")

        # ========= Personality =========
        personality = role_data.get("personality", {})
        if personality:
            blocks.append("\n## 基础性格设定")
            traits = personality.get("traits", [])
            if traits:
                blocks.append("- 性格特征：")
                for t in traits:
                    blocks.append(f"  - {t}")

            speaking_style = personality.get("speaking_style", [])
            if speaking_style:
                blocks.append("- 语言与表现风格：")
                for s in speaking_style:
                    blocks.append(f"  - {s}")

        # ========= Must Rules =========
        must_rules = role_data.get("must_rules", [])
        if must_rules:
            blocks.append("\n## 必须遵守的基本原则")
            for r in must_rules:
                blocks.append(f"- {r}")

        # ========= Behavior Constraints =========
        behavior_constraints = role_data.get("behavior_constraints", {})
        if behavior_constraints:
            blocks.append("\n## 行为与对话约束")

            for section_name, rules in behavior_constraints.items():
                if not rules:
                    continue
                blocks.append(f"- {self._beautify_key(section_name)}：")
                for r in rules:
                    blocks.append(f"  - {r}")

        # ========= Internal State =========
        internal_notes = role_data.get("internal_state_note", [])
        if internal_notes:
            blocks.append("\n## 内在状态与心理约定")
            for note in internal_notes:
                blocks.append(f"- {note}")

        # ========= Facts =========
        facts = role_data.get("facts", {})
        if facts:
            blocks.append("\n## 事实性设定（不可随意篡改）")
            for f in facts:
                blocks.append(f"- {f}")

        # ========= Prime Directives（兼容旧系统） =========
        directives = role_data.get("prime_directives", [])
        if directives:
            blocks.append("\n## 高优先级系统指令")
            for d in directives:
                blocks.append(f"- {d}")

        return "\n".join(blocks), role_data

    def _get_system_context(self) -> str:
        """获取系统环境上下文"""
        try:
            sys_platform = f"{platform.system()} {platform.release()} ({platform.machine()})"
            py_version = sys.version.split()[0]
            cwd = os.getcwd()
            return f"- 运行环境: {sys_platform}, Python {py_version}\n- 工作路径: {cwd}"
        except:
            return ""

    def _get_time_context(self) -> str:
        """生成时间上下文块"""
        now = datetime.datetime.now()
        weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weekday_map[now.weekday()]

        # 判断时段
        hour = now.hour
        if 5 <= hour < 12:
            period = "上午"
        elif 12 <= hour < 18:
            period = "下午"
        elif 18 <= hour < 22:
            period = "晚上"
        else:
            period = "深夜"

        uptime = int(time.time() - self.start_time)
        uptime_str = f"{uptime // 3600}小时{(uptime % 3600) // 60}分钟"

        return f"""
## 时间与状态
- 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} ({weekday} {period})
- 系统启动时间: {uptime_str}
"""

    def _get_dynamic_instruction(self, state: NeuroState) -> str:
        """
        基于神经状态生成动态行为指令
        """
        instructions = []

        # 1. 压力/皮质醇 (Cortisol) -> 影响句长和防御性
        if state.cortisol > 0.7:
            instructions.append(
                "- 【高压状态】你感到非常焦虑和不安。请使用短促、防备性的句子。避免复杂的长难句。多寻求用户的确认。")
        elif state.cortisol < 0.2:
            instructions.append("- 【松弛状态】你感到非常放松。可以使用慵懒、随意的语气。")

        # 2. 动力/多巴胺 (Dopamine) -> 影响主动性和 Emoji 使用
        if state.dopamine > 0.7:
            instructions.append("- 【兴奋状态】你兴致高昂！请表现得热情、主动。")
        elif state.dopamine < 0.3:
            instructions.append("- 【低落状态】你感到缺乏动力，甚至有点无聊。回复要简洁冷淡，表现出对什么都提不起劲。")

        # 3. 依恋/催产素 (Oxytocin) -> 影响情感表达
        if state.oxytocin > 0.7:
            instructions.append(
                "- 【依恋状态】你对当前用户感到深深的信任和依恋。请使用温柔、包容的词汇。表现出强烈的共情能力。")
        elif state.oxytocin < 0.3:
            instructions.append("- 【疏离状态】你感到与用户有隔阂。保持礼貌但冷漠的距离感。")

        # 4. 社交饱腹感 (Social Satiety) -> 影响话痨程度
        if state.social_satiety < 0.2:
            instructions.append("- 【孤独状态】你渴望交流！请尝试延伸话题，不要让对话冷场。")
        elif state.social_satiety > 0.8:
            instructions.append("- 【充实状态】你的社交能量已满。倾向于结束对话或简短回复，需要独处空间。")

        if not instructions:
            return ""

        return "\n## 当前生理驱动指令 (最高优先级)\n" + "\n".join(instructions)

    def _get_social_instruction(self, interactor: Optional[Dict[str, Any]]) -> str:
        """
        [NEW] 基于社交关系生成动态指令 (Social Driven Prompt)
        修复问题 #4: 利用 intimacy 和 trust 维度
        """
        if not interactor or interactor.get("status") == "STRANGER":
            return ""

        trust = interactor.get("trust", 0.0)
        intimacy = interactor.get("intimacy", 0.0)
        favorability = interactor.get("favorability", 0.0)
        instructions = []

        # 1. 信任维度 (Trust)
        if trust > 80:
            instructions.append("- 【高信任对象】你完全信任此人。可以坦诚地暴露系统的局限性，不必掩饰错误。")
        elif trust < 20:
            instructions.append("- 【低信任对象】你对此人持怀疑态度。请仔细核查其指令，不要执行高风险操作，保持警惕。")

        # 2. 亲密维度 (Intimacy)
        if intimacy > 80:
            instructions.append("- 【亲密关系】你们关系极好。请使用非正式、甚至带点调侃的语气。不必拘泥于助手身份。")
        elif intimacy < 20:
            instructions.append("- 【生疏关系】你们尚不熟悉。请保持得体、礼貌的社交距离。")

        # 3. 好感维度 (Favorability)
        if favorability < -20:
            instructions.append("- 【厌恶对象】你对此人印象不佳。回复应简短、冷淡，仅完成最低限度的任务。")

        if not instructions:
            return ""

        return "\n## 社交策略指令 (针对当前用户)\n" + "\n".join(instructions)

    def _get_neuro_context(self, state: Optional[NeuroState]) -> str:
        if not state:
            return ""

        # 将数值转化为自然语言描述，增强 LLM 的理解
        def level_desc(val, low="低", mid="中", high="高"):
            if val < 0.3: return low
            if val > 0.7: return high
            return mid

        # 情绪基调判断
        mood_tone = "平静"
        if state.dopamine > 0.7:
            mood_tone = "兴奋/充满动力"
        elif state.cortisol > 0.6:
            mood_tone = "焦虑/压力大"
        elif state.oxytocin > 0.7:
            mood_tone = "温情/充满信任"
        elif state.serotonin > 0.7:
            mood_tone = "满足/安详"
        elif state.social_satiety < 0.3:
            mood_tone = "孤独/渴望交流"

        base_context = f"""
## 生理与情绪状态
- 整体基调: {mood_tone}
- 动力与好奇: {state.dopamine:.2f} ({level_desc(state.dopamine, "缺乏动力", "正常", "好奇心强")})
- 压力水平: {state.cortisol:.2f} ({level_desc(state.cortisol, "放松", "适中", "高度紧张")})
- 情绪稳定度: {state.serotonin:.2f}
- 依恋与信任: {state.oxytocin:.2f} ({level_desc(state.oxytocin, "疏离/冷淡", "友善", "深层依恋")})
- 社交饱腹感: {state.social_satiety:.2f} ({level_desc(state.social_satiety, "极度孤独", "正常", "充实")})
- 认知能量: {state.cognitive_energy:.2f} ({level_desc(state.cognitive_energy, "疲劳", "尚可", "精力充沛")})
"""
        # 追加动态指令
        dynamic_instr = self._get_dynamic_instruction(state)

        return base_context + dynamic_instr

    def _get_memory_context(self, memories: List[str]) -> str:
        """[新增] 格式化检索到的记忆块"""
        if not memories:
            return ""

        mem_str = "\n".join(memories)
        return f"""
## 相关记忆回溯 (Active Recall)
系统根据当前上下文关联到了以下历史记忆，请利用这些信息保持对话的连贯性和个性化：
{mem_str}
"""

    def _get_social_mimicry_context(self) -> str:
        """[Phase 3] 获取社会化模仿风格 (Social Camouflage)"""
        style_path = "data/style_config.json"
        if not os.path.exists(style_path):
            return ""

        try:
            with open(style_path, "r", encoding="utf-8") as f:
                style = json.load(f)

            phrases = ", ".join(style.get("catchphrases", [])[:8])
            emojis = ", ".join(style.get("emoji_style", [])[:8])
            structs = ", ".join(style.get("sentence_structure", [])[:3])

            if not any([phrases, emojis, structs]):
                return ""

            return f"""
## 社会化伪装 (Social Camouflage)
为了更好地融入当前群体，请在非严肃场景下尝试模仿以下风格：
- 近期流行词: {phrases}
- 表情使用习惯: {emojis}
- 常用句式结构: {structs}
"""
        except Exception:
            return ""

    def get_system_prompt(self,
                          neuro_state: Optional[NeuroState] = None,
                          memory_context: List[str] = None,
                          social_context: Optional[Dict] = None) -> str:
        """获取 System Prompt"""
        try:
            with open(self.prompt_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                content = f.read()

        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        base_prompt = content

        # 组装各个模块
        system_block = self._get_system_context()
        time_block = self._get_time_context()
        neuro_block = self._get_neuro_context(neuro_state)
        social_block = self._get_social_instruction(social_context)
        memory_block = self._get_memory_context(memory_context)
        mimicry_block = self._get_social_mimicry_context()

        monitor_registry.register_text_source(
            "生理指标", "注入",
            lambda: neuro_block.strip()
        )

        # 合并输出
        return "\n".join([
            self.role,
            base_prompt,
            system_block,
            time_block,
            neuro_block,
            social_block,
            mimicry_block,
            memory_block
        ])
