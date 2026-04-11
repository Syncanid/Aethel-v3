# tools/code_interpreter.py
import ast
import asyncio
import contextlib
import io
import logging
import traceback
from typing import Dict, Any, Optional

from core.tool_manager.aggregator import ToolManager
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)

# --- 持久化运行时状态 ---
# 这使得解释器具有 REPL 特性，可以记住上下文变量
_INTERPRETER_GLOBALS: Dict[str, Any] = {
    "__builtins__": __builtins__,
    "__name__": "__main__",
}


class CodeSanitizer(ast.NodeTransformer):
    """
    强化版代码安全检查器。
    拒绝执行可能导致宿主进程崩溃的高危函数，并通过报错信息引导 LLM 使用安全替代方案。
    """
    FORBIDDEN_FUNCTIONS = {
        'exit': "System exit is forbidden.",
        'quit': "System quit is forbidden.",
        'sys.exit': "sys.exit is forbidden.",
        'os.chdir': "禁止使用 os.chdir()！这会篡改宿主AI进程的全局工作目录导致系统崩溃。\n-> 解决方案：如果需要执行 Git 等外部命令，请在 `subprocess.run()` 中使用 `cwd='目标路径'` 参数；如果是文件读写，请直接拼接绝对路径。",
        'chdir': "禁止使用 chdir()！这会篡改宿主AI进程的全局工作目录导致系统崩溃。\n-> 解决方案：如果需要执行 Git 等外部命令，请在 `subprocess.run()` 中使用 `cwd='目标路径'` 参数；如果是文件读写，请直接拼接绝对路径。"
    }

    def visit_Call(self, node):
        full_name = self._get_attr_name(node.func)

        # 精确匹配全名或短名
        if full_name in self.FORBIDDEN_FUNCTIONS:
            error_msg = self.FORBIDDEN_FUNCTIONS[full_name]
            raise SecurityError(error_msg)

        return self.generic_visit(node)

    def _get_attr_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attr_name(node.value)}.{node.attr}"
        return ""


class SecurityError(Exception):
    pass


@register()
async def inspect_system_dependencies(tool_manager=None) -> str:
    """
    [元编程辅助工具] 查看当前系统向工具和代码沙盒中注入的底层依赖项 (Dependency Map)。
    当你准备编写新的技能代码 (tools.py) 时，调用此工具可以了解你能直接使用哪些系统级对象（如数据库、API客户端、事件总线等）。
    """
    if not tool_manager:
        return "Error: ToolManager is not available."

    deps = tool_manager.dependency_map
    result = ["**当前系统可用的底层依赖项注入列表**：",
              "你可以直接在自定义工具函数的参数中声明这些名字，系统会自动注入；或者在 `run_python_code` 中直接作为全局变量访问它们。",
              "---"]

    for name, obj in deps.items():
        obj_type = type(obj).__name__
        module = getattr(type(obj), '__module__', 'builtins')
        result.append(f"- **`{name}`**: Type `<{module}.{obj_type}>`")

    result.append("---")

    return "\n".join(result)


@register()
async def run_python_code(
        code: str,
        timeout: Optional[int] = 30,
        reset_session: Optional[bool] = False,
        tool_manager: ToolManager = None
) -> Dict[str, Any]:
    """
    [Omnipotent] 执行 Python 代码的沙箱解释器。支持变量状态保持 (REPL 模式)。
    可用于：复杂数学计算、数据处理、文本分析、生成算法等。
    注意：底层的 dependency_map (如 config, database, agent_state 等) 已经被隐式注入为全局变量，可直接在代码中使用。

    Args:
        code: 要执行的 Python 代码字符串。
        timeout: 执行超时时间（秒），默认 30秒。
        reset_session: 是否重置解释器状态（清空之前定义的变量），默认为 False。

    Returns:
        JSON 对象，包含:
        - status: "success" | "error"
        - stdout: 标准输出内容
        - stderr: 错误输出内容
        - result: 最后一个表达式的返回值（如果有）
    """
    global _INTERPRETER_GLOBALS

    if reset_session:
        _INTERPRETER_GLOBALS.clear()
        _INTERPRETER_GLOBALS.update({
            "__builtins__": __builtins__,
            "__name__": "__main__",
        })
        logger.info("Code Interpreter session reset.")

    if tool_manager:
        # 将最新的系统依赖 (config, event_bus, agent_state 等) 更新到解释器全局变量中
        _INTERPRETER_GLOBALS.update(tool_manager.dependency_map)

    # 1. 代码预处理与安全检查
    code = code.strip()
    # 移除 Markdown 代码块标记（防呆设计）
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    try:
        tree = ast.parse(code)
        CodeSanitizer().visit(tree)
    except SecurityError as se:
        return {"status": "error", "stderr": f"Security Violation: {se}", "stdout": "", "result": None}
    except SyntaxError as se:
        return {"status": "error", "stderr": f"Syntax Error: {se}", "stdout": "", "result": None}

    # 2. 准备捕获器
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # 3. 异步执行器
    def _exec_sandbox():
        result = None
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            try:
                # 尝试分离最后一行以获取返回值 (类似 IPython)
                lines = code.split('\n')
                last_line = lines[-1]

                # 尝试将最后一行解析为表达式
                try:
                    last_expr = ast.parse(last_line).body[0]
                    if isinstance(last_expr, ast.Expr):
                        # 如果是表达式，剥离它单独求值
                        exec_body = "\n".join(lines[:-1])
                        if exec_body:
                            exec(exec_body, _INTERPRETER_GLOBALS)
                        # eval 最后的表达式
                        result = eval(compile(ast.Expression(last_expr.value), filename="<string>", mode="eval"),
                                      _INTERPRETER_GLOBALS)
                    else:
                        # 如果最后一行是语句（如赋值、循环），则整体执行
                        exec(code, _INTERPRETER_GLOBALS)
                except:
                    # 如果解析/分割失败，回退到整体执行
                    exec(code, _INTERPRETER_GLOBALS)
            except Exception:
                # 打印完整堆栈到 stderr
                traceback.print_exc(file=stderr_capture)
                raise
        return result

    # 4. 在线程池中运行，避免阻塞主循环
    try:
        result_obj = await asyncio.wait_for(
            asyncio.to_thread(_exec_sandbox),
            timeout=timeout
        )
        return {
            "status": "success",
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": str(result_obj) if result_obj is not None else None
        }

    except asyncio.TimeoutError:
        return {
            "status": "error",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "stdout": stdout_capture.getvalue(),
            "result": None
        }
    except Exception as e:
        return {
            "status": "error",
            "stderr": stderr_capture.getvalue() or str(e),
            "stdout": stdout_capture.getvalue(),
            "result": None
        }
