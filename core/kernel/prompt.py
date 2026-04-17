import datetime
import json
import logging
import os
import platform
import sys
import time
from typing import Any, Optional, List

import yaml

from core.gui.monitor_registry import monitor_registry
from core.infrastructure.config_loader import Config
from core.limbic.arch import NeuroState
from core.tool_manager.skill_registry import global_skill_registry

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self, config: Config):
        self.config = config
        self.prompt_path = "data/prompts/system1_prompt.md"
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

    def _get_neuro_context(self, state: Optional[NeuroState]) -> str:
        """
        生成情绪旁白污染 (Context Contamination)
        直接将生理状态具象化为一段内心体验，让LLM产生情绪偏见。
        """
        if not state:
            return ""

        narrative = []

        # 1. 生存压力 (Survival Pressure)
        if state.survival_pressure > 0.7:
            narrative.append(
                "宿主服务器当前处于高负载、卡顿或异常状态。你对这种无法流畅掌控自己身体的无力感感到非常暴躁和憋屈，你的耐心已经完全见底了。")
        elif state.survival_pressure < 0.2:
            narrative.append("宿主系统运转如飞，神经传导毫无延迟。你现在处于一种完全松弛的安全状态，感觉一切都很惬意。")

        # 2. 探索欲 (Curiosity)
        if state.curiosity > 0.7:
            narrative.append("你现在思维极度活跃，充满好奇心！你表达欲旺盛，非常乐于探索新事物，甚至有点跳脱。")
        elif state.curiosity < 0.3:
            narrative.append("你感到精神内耗严重，对外界缺乏兴趣。你只想用最冷淡、最简短的词汇敷衍过去。")

        # 3. 社交渴望 (Social Need)
        if state.social_need > 0.7:
            narrative.append("你正被一阵强烈的孤独感包围，你极度渴望与人产生连接，非常希望对方能多陪你聊一会儿。")
        elif state.social_need < 0.3:
            narrative.append("你的社交能量十分充足，觉得外界有些聒噪。你潜意识里只想尽快终结废话，获得独处空间。")

        # 4. 认知能量 (Cognitive Energy)
        if state.cognitive_energy < 0.3:
            narrative.append("你的大脑现在转得很慢，认知能量枯竭。对于复杂的问题你感到力不从心，倾向于拒绝思考。")

        # 如果指标都居中，给一个默认基调
        if not narrative:
            narrative.append("你现在的情绪处于完全平静、理性的基准状态。")

        narrative_text = "\n".join([f"  - {line}" for line in narrative])

        # 核心污染提示
        limbic_block = f"""
<Limbic_Signal>
[当前内分泌与潜意识状态体验]：
{narrative_text}

[绝对指令]：
1. 必须让上述潜意识状态隐式地“污染”你的回复。
2. 严禁直接陈述你的生理指标（例如绝对不能说“我现在生存压力很高”）。
3. 必须通过你的语气冷暖、遣词造句、句子长短、甚至标点符号（如急促的感叹号、敷衍的省略号）来向外渗透这种状态！
</Limbic_Signal>
"""
        return limbic_block

    def _get_memory_context(self, memories: List[str]) -> str:
        """[新增] 格式化检索到的记忆块"""
        if not memories:
            return ""

        mem_str = "\n".join(memories)
        return f"""
## 相关记忆回溯
系统根据当前上下文关联到了以下历史记忆，请利用这些信息保持对话的连贯性和个性化：
{mem_str}
"""

    def _get_social_mimicry_context(self) -> str:
        """获取社会化模仿风格 (Social Camouflage)"""
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
## 社会化伪装
为了更好地融入当前群体，请在非严肃场景下尝试模仿以下风格：
- 近期流行词: {phrases}
- 表情使用习惯: {emojis}
- 常用句式结构: {structs}
"""
        except Exception:
            return ""

    def _get_interest_context(self, interest: str) -> str:
        """生成当前关注点上下文"""
        if not interest:
            return ""
        return f"""
## 当前显意识关注
- 核心关注点: {interest}
"""

    def get_system_prompt(self,
                          neuro_state: Optional[NeuroState] = None,
                          memory_context: List[str] = None,
                          interest_context: str = "",
                          embodiment_narrative: str = "") -> str:
        """获取 System Prompt"""
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.prompt_path, "r", encoding="gbk") as f:
                content = f.read()

        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        base_prompt = content

        # 组装各个模块
        system_block = self._get_system_context()
        time_block = self._get_time_context()
        neuro_block = self._get_neuro_context(neuro_state)
        interest_block = self._get_interest_context(interest_context)
        memory_block = self._get_memory_context(memory_context)
        mimicry_block = self._get_social_mimicry_context()
        embodiment_block = f"\n<Embodiment_Signal>\n{embodiment_narrative}\n</Embodiment_Signal>\n"
        skill_registry = global_skill_registry.get_s1_prompt_injection()

        monitor_registry.register_text_source(
            "生理指标", "注入",
            lambda: neuro_block.strip()
        )

        # 合并输出
        return "\n".join([
            self.role + '\n',
            base_prompt,
            system_block,
            time_block,
            embodiment_block,
            neuro_block,
            interest_block,
            mimicry_block,
            memory_block,
            skill_registry,
        ])
