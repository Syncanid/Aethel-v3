# tools/System2/skill_installer.py
import logging
import os
from typing import List, Optional

import aiofiles
import yaml

from core.tool_manager.registry import register
from core.tool_manager.skill_registry import global_skill_registry

logger = logging.getLogger(__name__)


@register()
async def create_custom_skill(
        skill_name: str,
        description: str,
        required_params: List[str],
        sop_markdown: str,
        tools_python_code: Optional[str] = None
) -> str:
    """
    [核心进化工具] 为系统创造或安装一个全新的专业技能。
    当用户要求你“学习一个新技能”、“固化这个工作流”，或者你从外部获取了优质的开源技能代码并完成安全审计后，调用此工具将其写入系统。

    :param skill_name: 技能的唯一英文短名 (如 'github-analyzer', 不能有空格)
    :param description: 技能的触发描述 (System 1 会根据这个描述来决定何时路由给你，请写得清晰准确)
    :param required_params: 触发该技能需要 System 1 从用户话语中提取的参数名列表 (如 ["url", "keyword"])，如果没有则传 []
    :param sop_markdown: 技能的标准作业程序 (SOP)。这是未来你执行该技能时的 System Prompt，请详细描述步骤。
    :param tools_python_code: (可选) 该技能专属的 Python 工具代码。必须符合 Aethel 的工具编写规范（使用 @register 装饰器）。
    """
    # 1. 基础校验
    if not skill_name.replace("-", "").replace("_", "").isalnum():
        return "Error: 技能名称不合法，只能包含字母、数字、横线和下划线。"

    skills_dir = os.path.join("data", "skills", skill_name)

    try:
        # 2. 创建目录
        os.makedirs(skills_dir, exist_ok=True)

        # 3. 写入 skill.yaml (供 S1 路由使用)
        yaml_content = {
            "name": skill_name,
            "description": description,
            "required_params": required_params
        }
        yaml_path = os.path.join(skills_dir, "skill.yaml")
        async with aiofiles.open(yaml_path, "w", encoding="utf-8") as f:
            await f.write(yaml.dump(yaml_content, allow_unicode=True, sort_keys=False))

        # 4. 写入 sop.md (供 S2 挂载使用)
        sop_path = os.path.join(skills_dir, "sop.md")
        async with aiofiles.open(sop_path, "w", encoding="utf-8") as f:
            await f.write(sop_markdown)

        # 5. 写入 tools.py (专属代码工具)
        if tools_python_code and tools_python_code.strip():
            # 简单的防呆校验，确保包含了必要的导入
            if "from core.tool_manager.registry import register" not in tools_python_code:
                tools_python_code = "from core.tool_manager.registry import register\n\n" + tools_python_code

            tools_path = os.path.join(skills_dir, "tools.py")
            async with aiofiles.open(tools_path, "w", encoding="utf-8") as f:
                await f.write(tools_python_code)

        # 6. 触发全局热重载，使得 S1 立即“学会”新技能
        global_skill_registry.reload()

        logger.info(f"🧬 自我进化成功：新技能 [{skill_name}] 已被创建并载入。")
        return (
            f"🎉 技能 [{skill_name}] 创建并落盘成功！\n"
            f"Skill Registry 已热重载。System 1 现在已经知道了这个新技能，随时可以触发路由。"
        )

    except Exception as e:
        logger.error(f"创建技能失败: {e}", exc_info=True)
        return f"创建技能过程中发生异常: {str(e)}"
