# tools/code_interpreter.py
import ast
import asyncio
import contextlib
import io
import logging
import traceback
from typing import Dict, Any, Optional

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
    基础代码安全检查器。
    拒绝执行可能导致进程退出的高危函数。
    """
    FORBIDDEN_FUNCTIONS = {'exit', 'quit', 'sys.exit'}

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_FUNCTIONS:
                raise SecurityError(f"Forbidden function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            # Check for sys.exit type calls
            full_name = self._get_attr_name(node.func)
            if full_name in self.FORBIDDEN_FUNCTIONS:
                raise SecurityError(f"Forbidden function call: {full_name}")
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
async def run_python_code(
        code: str,
        timeout: Optional[int] = 30,
        reset_session: Optional[bool] = False
) -> Dict[str, Any]:
    """
    [Omnipotent] 执行 Python 代码的沙箱解释器。支持变量状态保持 (REPL 模式)。
    可用于：复杂数学计算、数据处理、文本分析、生成算法等。

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
