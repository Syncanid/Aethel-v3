# tools/filesystem.py
import logging
import os
import shutil
from typing import Literal, Optional

import aiofiles

from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


@register()
async def list_directory(path: str = ".", max_depth: int = 1) -> str:
    """
    [FileSystem] 列出指定目录下的文件结构。

    Args:
        path: 目录路径，默认为当前目录。
        max_depth: 递归深度，防止输出过长。默认为 1。
    """
    if not os.path.exists(path):
        return f"错误: 路径 '{path}' 不存在。"
    if not os.path.isfile(path) and not os.path.isdir(path):
        return f"错误: '{path}' 不是文件也不是目录。"

    def _build_tree(dir_path, level):
        if level > max_depth: return ""
        tree_str = ""
        try:
            items = sorted(os.listdir(dir_path))
        except Exception as e:
            return f"  [Access Denied: {e}]\n"

        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            prefix = "`-- " if is_last else "|-- "
            item_path = os.path.join(dir_path, item)

            tree_str += f"{'    ' * (level - 1)}{prefix}{item}"
            if os.path.isdir(item_path):
                tree_str += "/\n"
                tree_str += _build_tree(item_path, level + 1)
            else:
                tree_str += "\n"
        return tree_str

    if os.path.isfile(path):
        return f"文件: {path}"

    try:
        return f"目录结构 ({path}):\n" + _build_tree(path, 1)
    except Exception as e:
        return f"列出目录失败: {e}"


@register()
async def read_file(path: str, start_line: int = 1, end_line: int = -1) -> str:
    """
    [FileSystem] 读取文件内容。支持读取指定行范围。

    Args:
        path: 文件路径。
        start_line: 起始行号 (从1开始)。
        end_line: 结束行号 (-1 表示读到末尾)。
    """
    if not os.path.exists(path):
        return f"错误: 文件 '{path}' 不存在。"
    if os.path.isdir(path):
        return f"错误: '{path}' 是一个目录。"

    try:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            lines = await f.readlines()

        total_lines = len(lines)
        if end_line == -1: end_line = total_lines

        # 修正索引
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)

        content = "".join(lines[start_idx:end_idx])
        return f"--- 文件: {path} (行 {start_line}-{end_idx}/{total_lines}) ---\n{content}"
    except Exception as e:
        return f"读取失败: {e}"


@register()
async def edit_file(
        path: str,
        content: str,
        mode: Literal["overwrite", "append", "insert", "replace"] = "overwrite",
        line_number: Optional[int] = None,
        old_content: Optional[str] = None
) -> str:
    """
    [FileSystem] 编辑或创建文件。支持多种修改模式。

    Args:
        path: 文件路径。
        content: 要写入/插入的新内容。
        mode: 编辑模式:
            - 'overwrite': 覆盖整个文件 (默认)。
            - 'append': 追加到文件末尾。
            - 'insert': 插入到指定 'line_number' 之后。
            - 'replace': 将文件中的 'old_content' 替换为 'content'。
        line_number: (仅 insert 模式) 插入位置的行号。
        old_content: (仅 replace 模式) 要被替换的旧文本。
    """
    try:
        # 1. 覆盖模式
        if mode == "overwrite":
            dir_path = os.path.dirname(path)
            if dir_path: os.makedirs(dir_path, exist_ok=True)
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            return f"文件 '{path}' 已覆盖写入。"

        # 2. 追加模式
        if mode == "append":
            async with aiofiles.open(path, 'a', encoding='utf-8') as f:
                await f.write(content)
            return f"内容已追加到 '{path}' 末尾。"

        # 以下模式需要先读取文件
        if not os.path.exists(path):
            return f"错误: 文件 '{path}' 不存在，无法执行 insert/replace 操作。"

        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            lines = await f.readlines()
            full_text = "".join(lines)

        # 3. 插入模式
        if mode == "insert":
            if line_number is None:
                return "错误: insert 模式必须提供 'line_number'。"

            idx = line_number  # 在第 N 行之后插入，list insert 索引正好是 N
            if idx > len(lines): idx = len(lines)

            # 确保插入内容有换行
            if not content.endswith('\n'): content += '\n'

            lines.insert(idx, content)
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.writelines(lines)
            return f"内容已插入到 '{path}' 第 {line_number} 行之后。"

        # 4. 替换模式
        if mode == "replace":
            if not old_content:
                return "错误: replace 模式必须提供 'old_content'。"

            if old_content not in full_text:
                return f"错误: 在文件中未找到指定的 'old_content'。"

            new_text = full_text.replace(old_content, content)
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(new_text)
            return f"文件 '{path}' 中的内容已替换。"

        return f"错误: 未知的模式 '{mode}'。"

    except Exception as e:
        return f"编辑文件失败: {e}"


@register()
async def manage_file(
        action: Literal["move", "copy", "delete"],
        src_path: str,
        dest_path: Optional[str] = None
) -> str:
    """
    [FileSystem] 文件管理工具 (移动、复制、删除)。

    Args:
        action: 操作类型 ('move', 'copy', 'delete')。
        src_path: 源文件路径。
        dest_path: 目标路径 (move/copy 必需)。
    """
    try:
        if action == "delete":
            if os.path.isdir(src_path):
                shutil.rmtree(src_path)
            else:
                os.remove(src_path)
            return f"已删除: {src_path}"

        if not dest_path:
            return "错误: move/copy 操作需要 dest_path。"

        dir_path = os.path.dirname(dest_path)
        if dir_path: os.makedirs(dir_path, exist_ok=True)

        if action == "move":
            shutil.move(src_path, dest_path)
            return f"已移动: {src_path} -> {dest_path}"

        if action == "copy":
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
            return f"已复制: {src_path} -> {dest_path}"

        return "未知操作"

    except Exception as e:
        return f"文件管理操作 {action} 失败: {e}"
