# core/tool_manager/skill_registry.py
import logging
import os
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Agent Skills 静态元数据管理器"""

    def __init__(self, skills_dir: str = "data/skills"):
        self.skills_dir = skills_dir
        self.skills: Dict[str, Any] = {}
        self.reload()

    def reload(self):
        """扫描并重新加载所有技能的元数据 (yaml)"""
        self.skills.clear()
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir, exist_ok=True)
            return

        for item in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, item)
            if os.path.isdir(skill_path):
                yaml_path = os.path.join(skill_path, "skill.yaml")
                if os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, "r", encoding="utf-8") as f:
                            skill_data = yaml.safe_load(f)
                            if skill_data and "name" in skill_data:
                                self.skills[skill_data["name"]] = skill_data
                                logger.info(f"🧩 发现可用技能包: {skill_data['name']}")
                    except Exception as e:
                        logger.error(f"加载技能 {item} 的 yaml 文件失败: {e}")

    def get_s1_prompt_injection(self) -> str:
        """生成供 System 1 注入到 Prompt 中的可用技能列表"""
        if not self.skills:
            return "当前未挂载任何额外的技能 (Skills)。"

        lines = ["\n### 🟢 可调用的技能库 (Skills)",
                 "当前系统已挂载以下专用工作流。当用户需求与描述高度匹配时，你必须调用 `dispatch_skill_task` 工具来处理，绝不可直接回复或使用通用后台任务："]

        for name, data in self.skills.items():
            desc = data.get("description", "无描述")
            params = data.get("required_params", [])
            lines.append(f"- **{name}**: {desc} (需要通过 parameters 提取的参数: {params})")

        return "\n".join(lines)


# 实例化全局单例
global_skill_registry = SkillRegistry()
