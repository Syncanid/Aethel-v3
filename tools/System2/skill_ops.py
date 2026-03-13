# tools/System2/skill_ops.py
import os
from typing import Optional

import aiofiles
import yaml

from core.tool_manager.aggregator import ToolManager
from core.tool_manager.registry import register


@register()
async def list_available_skills() -> str:
    """
    当你在执行任务时发现缺乏某种专业能力（如画图、分析特定数据、操作特定软件）时，调用此工具。
    它会列出本地系统所有已安装的技能包(Skills)及其触发条件和描述。
    """
    skills_dir = "data/skills"
    if not os.path.exists(skills_dir):
        return "当前系统未安装任何外部技能包。"

    skills_info = []
    for item in os.listdir(skills_dir):
        skill_path = os.path.join(skills_dir, item)
        if os.path.isdir(skill_path):
            yaml_path = os.path.join(skill_path, "skill.yaml")
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        name = data.get("name", item)
                        desc = data.get("description", "无描述")
                        skills_info.append(f"- **{name}**: {desc}")
                except Exception:
                    continue

    if not skills_info:
        return "未发现有效的技能包。"

    return "本地可用的技能列表：\n" + "\n".join(skills_info)


@register()
async def mount_skill(
        skill_name: str,
        tool_manager: ToolManager = None,
        agent_state: Optional[dict] = None
) -> str:
    """
    在任务执行中途，动态挂载一个专业技能。这会为你提供该技能的专属工具，并返回该技能的操作指南(SOP)。
    注意：同一时间只能挂载一个技能。挂载新技能会自动卸载当前已挂载的其他技能。

    :param skill_name: 要挂载的技能名称 (例如 'seo-audit')
    """
    if not tool_manager:
        return "Error: ToolManager not found in context."

    # 1. 动态挂载代码工具
    tool_manager.mount_skill_tools(skill_name)

    # 2. 提取并返回该技能的 SOP（让 S2 的 LLM 直接在工具结果中阅读 SOP）
    sop_path = f"data/skills/{skill_name}/sop.md"
    if os.path.exists(sop_path):
        try:
            async with aiofiles.open(sop_path, "r", encoding="utf-8") as f:
                sop_content = await f.read()

            # 更新内部状态的进度
            if agent_state:
                agent_state["progress_summary"] = f"已切换至 {skill_name} 专家模式"

            return (
                f"✅ 技能 [{skill_name}] 已成功挂载！相关的专属代码工具已就绪。\n\n"
                f"⚠️ 【强制执行规范 (SOP)】\n请立即阅读并严格按照以下工作流推进你的计划：\n"
                f"-----------------------------------\n{sop_content}\n-----------------------------------"
            )
        except Exception as e:
            return f"技能工具已挂载，但读取 SOP 失败: {e}"

    return f"✅ 技能 [{skill_name}] 专属工具已挂载成功。未发现独立的 SOP 文件，请直接调用新工具。"


@register()
async def unmount_current_skill(
        tool_manager: ToolManager = None,
        agent_state: Optional[dict] = None
) -> str:
    """
    卸载当前挂载的技能，清理上下文，恢复到基础环境。当一个专业领域的子任务（如爬虫、特定分析）完成，准备进行下一步通用推理前，务必调用此工具以释放上下文。
    """
    if not tool_manager:
        return "Error: ToolManager not found."

    active_skill = tool_manager.active_skill
    if not active_skill:
        return "当前没有挂载任何技能，无需卸载。"

    tool_manager.unmount_skill_tools()

    if agent_state:
        agent_state["progress_summary"] = f"已卸载技能 {active_skill}，恢复通用模式"

    return f"🧹 技能 [{active_skill}] 已成功卸载。环境已复原，之前的专用工具已不可用。"
