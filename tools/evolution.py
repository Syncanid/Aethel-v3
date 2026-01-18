import ast
import logging
import os

from core.tool_manager.aggregator import ToolManager  # 仅用于类型提示
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


@register()
async def patch_system_code(
        file_path: str,
        new_content: str,
        tool_manager: ToolManager  # 依赖注入
) -> str:
    """
    [Self-Dev] 修改系统代码并尝试热重载。
    警告：这是一个极其危险的操作。在调用前，你必须：
    1. 使用 read_source_code 读取旧代码。
    2. 确保新代码逻辑正确且无语法错误。

    Args:
        file_path: 要修改的文件路径，例如 'tools/senses.py'。
        new_content: 完整的、修改后的 Python 代码。
    """
    # 1. 安全沙箱：语法检查
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        return f"拒绝修改：新代码存在语法错误: {e}"

    # 2. 创建备份
    backup_path = file_path + ".bak"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_code = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(old_code)
    except Exception as e:
        return f"备份失败，操作取消: {e}"

    # 3. 写入新代码
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"写入文件失败: {e}"

    # 4. 尝试热重载
    # 将路径转换为模块名 (tools/senses.py -> tools.senses)
    module_name = file_path.replace("/", ".").replace("\\", ".").replace(".py", "")

    reload_result = await tool_manager.reload_tool_module(module_name)

    return f"代码已修补。备份位于 {backup_path}。\n重载结果: {reload_result}"


@register()
async def reload_module(
        module_name: str,
        tool_manager: ToolManager
) -> str:
    """
    [System] 热重载指定的系统模块。
    调用此工具会触发工具库的重新扫描。

    Args:
        module_name: 目标模块名 (e.g., 'tools.basic', 'tools.evolution')。
    """
    return await tool_manager.reload_tool_module(module_name)
