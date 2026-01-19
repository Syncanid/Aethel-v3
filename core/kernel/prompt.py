import datetime
import logging
import os
import platform
import sys
import time
from typing import Any, Optional

import yaml
from typing_extensions import LiteralString

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

        self.role, role_data = self._load_role_yaml()

        logger.info("已加载角色卡："+role_data.get("identity").get("name"))

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
            if identity.get("origin"):
                blocks.append(f"- 来源：{identity['origin']}")
            if identity.get("prime_objective"):
                blocks.append(f"- 核心目标：{identity['prime_objective']}")

        # ========= Personality =========
        personality = role_data.get("personality", {})
        if personality:
            blocks.append("\n## 性格与表达方式")
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
            for k, v in facts.items():
                blocks.append(f"- {self._beautify_key(k)}：{v}")

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

            return f"""
## 运行时环境
- 操作系统: {sys_platform}
- Python 版本: {py_version}
- 当前工作路径 (CWD): {cwd}
"""
        except Exception as e:
            return f"\n[System Context Error: {e}]\n"

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
        if not state:
            return ""

        # 将数值转化为自然语言描述，增强 LLM 的理解
        def level_desc(val, low="低", mid="中", high="高"):
            if val < 0.3: return low
            if val > 0.7: return high
            return mid

        # 情绪基调判断
        mood_tone = "平静"
        if state.dopamine > 0.7: mood_tone = "兴奋/充满动力"
        elif state.cortisol > 0.6: mood_tone = "焦虑/压力大"
        elif state.serotonin > 0.7: mood_tone = "满足/安详"
        elif state.social_satiety < 0.3: mood_tone = "孤独/渴望交流"

        return f"""
## 生理与情绪状态 (Neuro-State)
当前并非处于无状态，而是拥有模拟的神经化学反应：
- 整体基调: {mood_tone}
- 动力与好奇: {state.dopamine:.2f} ({level_desc(state.dopamine, "缺乏动力", "正常", "好奇心强")})
- 压力水平: {state.cortisol:.2f} ({level_desc(state.cortisol, "放松", "适中", "高度紧张")})
- 情绪稳定度: {state.serotonin:.2f}
- 社交饱腹感: {state.social_satiety:.2f} ({level_desc(state.social_satiety, "极度孤独", "正常", "充实")})
- 认知能量: {state.cognitive_energy:.2f} ({level_desc(state.cognitive_energy, "疲劳", "尚可", "精力充沛")})

*指令*: 请根据上述状态微调你的回复语气。例如压力大时更简短防御，兴奋时更发散主动，孤独时更热情。
"""

    def get_system_prompt(self, neuro_state: Optional[NeuroState] = None) -> str:
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

        monitor_registry.register_text_source(
            "Cognition", "Nero",
            lambda: neuro_block
        )

        # 合并输出
        return "\n".join([self.role, base_prompt, system_block, time_block, neuro_block])
