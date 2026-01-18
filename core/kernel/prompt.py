import datetime
import logging
import os
import platform
import sys
import time
from typing import Any

import yaml
from typing_extensions import LiteralString

from core.infrastructure.config_loader import Config

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

    def _load_role_yaml(self) -> str | tuple[LiteralString, Any]:
        """读取并格式化 YAML 角色卡为中文 System Prompt 段落"""
        try:
            with open(self.role_yaml_path, "r", encoding="utf-8-sig") as f:
                role_data = yaml.safe_load(f)
        except Exception as e:
            return f"\n[Role YAML Load Error: {e}]\n"

        # 构建可直接拼接的中文段落
        identity = role_data.get("identity", {})
        personality = role_data.get("personality", {})
        directives = role_data.get("prime_directives", [])

        blocks = []

        # Identity
        blocks.append("## 核心身份定义")
        blocks.append(f"- 名称: {identity.get('name', '')}")
        blocks.append(f"- 角色: {identity.get('role', '')}")
        blocks.append(f"- 来源: {identity.get('origin', '')}")
        blocks.append(f"- 核心目标: {identity.get('prime_objective', '')}")

        # Personality
        blocks.append("\n## 性格特征")
        traits = personality.get("traits", [])
        if traits:
            blocks.append("- 特征: " + ", ".join(traits))
        speaking_style = personality.get("speaking_style", [])
        if speaking_style:
            blocks.append("- 语言风格指南:")
            for s in speaking_style:
                blocks.append(f"  - {s}")

        # Prime directives
        if directives:
            blocks.append("\n## 最高指令")
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

    def get_system_prompt(self) -> str:
        """获取 System Prompt"""
        try:
            with open(self.prompt_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                content = f.read()

        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        base_prompt = content

        # 系统与时间上下文
        system_block = self._get_system_context()
        time_block = self._get_time_context()

        # 合并输出
        return "\n".join([self.role, base_prompt, system_block, time_block])
