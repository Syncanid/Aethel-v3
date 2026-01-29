# core/tool_manager/aggregator.py
import importlib
import inspect
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import Action
from core.kernel.agent import AutonomousAgent
from core.tool_manager.mcp_client import MCPClient
from core.tool_manager.registry import get_pending_functions, clear_pending
from core.tool_manager.schema_utils import SchemaGenerator

logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(self, config: Config, event_bus: EventBus, api_client: GenericAPIClient, database: Database,
                 agent_state: Dict[str, Any]):
        self.config = config
        self.event_bus = event_bus
        self.api_client = api_client
        self.database = database
        self.agent_state = agent_state

        self._local_tools: Dict[str, Any] = {}
        self._mcp_clients: Dict[str, MCPClient] = {}
        self._schemas: List[Dict[str, Any]] = []

        # 自愈配置
        self.max_self_heal_attempts = 1

        # 依赖注入容器
        self.dependency_map = {
            "config": config,
            "event_bus": event_bus,
            "api_client": api_client,
            "database": database,
            "agent_state": agent_state,
            "tool_manager": self,
        }

    def add_dependency(self, name: str, dependency):
        self.dependency_map[name] = dependency

    async def initialize(self):
        """初始化：加载本地工具 + 连接 MCP"""
        # 1. 加载本地工具
        self._load_local_tools()

        # 2. 连接 MCP Servers
        mcp_config = self.config.get("mcp_servers", {})
        if mcp_config:
            for name, conf in mcp_config.items():
                try:
                    client = MCPClient(name, conf["command"], conf["args"])
                    await client.start()
                    self._mcp_clients[name] = client
                    # 合并 Schema
                    self._schemas.extend(client.get_tools())
                except Exception as e:
                    logger.error(f"MCP服务器 {name} 加载失败: {e}")

    def _load_local_tools(self):
        """扫描 tools 目录并注册"""
        tools_dir = "tools"
        if not os.path.exists(tools_dir):
            os.makedirs(tools_dir)

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = f"tools.{filename[:-3]}"
                try:
                    importlib.import_module(module_name)
                    logger.debug(f"已导入工具: {module_name}")
                except Exception as e:
                    logger.error(f"工具组件 {module_name} 导入失败: {e}")

        # 获取所有通过 @register 注册的函数
        pending = get_pending_functions()
        for func, name in pending:
            tool_name = name or func.__name__

            # 生成 Schema (跳过依赖注入参数)
            schema = SchemaGenerator.get_function_schema(func, tool_name)

            # 过滤 Schema 中的依赖参数 (config, event_bus 等)
            props = schema["function"]["parameters"]["properties"]
            required = schema["function"]["parameters"]["required"]

            for dep in self.dependency_map.keys():
                if dep in props:
                    del props[dep]
                if dep in required:
                    required.remove(dep)

            self._local_tools[tool_name] = func
            self._schemas.append(schema)
            logger.debug(f"本地工具已注册: {tool_name}")

        clear_pending()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    async def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = None, allow_self_heal: bool = True) -> str:
        """
        统一执行入口，包含自愈机制。
        """
        try:
            # 实际执行逻辑
            result = await self._execute_tool_internal(name, args, context)

            # [Step 3] 结果检查：有些工具可能不抛异常，而是返回包含 "status": "error" 的 JSON 字符串
            # 针对 code_interpreter 特殊处理
            if isinstance(result, (dict, str)):
                res_str = str(result)
                if name == "run_python_code":
                    # 解析返回的 dict 检查 status
                    if isinstance(result, dict) and result.get("status") == "error":
                        raise RuntimeError(f"Interpreter Error: {result.get('stderr')}")
                    if "Syntax Error" in res_str or "Traceback" in res_str:
                         raise RuntimeError(f"Code Execution Error: {res_str}")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"工具 {name} 执行异常: {error_msg}")

            # 触发自愈回路
            if allow_self_heal and self.max_self_heal_attempts > 0:
                logger.info(f"🩹 触发自愈回路: {name}")

                # 1. 尝试自愈
                fixed_args = await self._attempt_self_heal(name, args, error_msg)

                if fixed_args:
                    logger.info(f"🩹 自愈成功，参数已修正。")

                    # === Step 2: 让模型“知道” (Awareness) ===
                    # 获取 Agent 实例
                    agent = self.dependency_map.get("agent")
                    learned_rule = None

                    if agent and isinstance(agent, AutonomousAgent):
                        # 异步触发学习过程 (不阻塞当前执行)
                        # 我们希望 Agent 记住这个教训，所以调用 Hippocampus
                        if hasattr(agent, "hippocampus"):
                            # 使用 asyncio.create_task 并行处理学习，不增加用户等待时间
                            # 但为了在本次回复中就能体现“我学会了”，也可以 await
                            learned_rule = await agent.hippocampus.review_tool_mistake(
                                tool_name=name,
                                original_args=args,
                                error=error_msg,
                                fixed_args=fixed_args
                            )

                        # 向 Agent 的思维流 (History) 插入系统通知
                        # 这让 Agent 在接下来的思考中知道刚才发生了什么
                        notice_content = f"【系统自愈报告】工具 `{name}` 初次调用失败（{error_msg}）。系统已自动修正参数并重试。"
                        if learned_rule:
                            notice_content += f"\n💡 新习得经验: {learned_rule}"

                        agent.history.append({
                            "role": "system",
                            "content": notice_content,
                            "metadata": {"ephemeral": True} # 标记为临时消息，不一定永久归档
                        })

                    # === Step 3: 重试 (Retry) ===
                    # 递归调用，但关闭自愈以防止无限递归
                    retry_result = await self.execute_tool(name, fixed_args, context, allow_self_heal=False)

                    # 返回结果时带上标记，表明这是修复后的结果
                    return f"[Self-Healed] {retry_result}"

            # 无法自愈，返回原始错误
            return f"Error executing '{name}': {error_msg}"

    async def _execute_tool_internal(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = None):
        """内部执行逻辑 """
        # 1. 尝试本地工具
        if name in self._local_tools:
            func = self._local_tools[name]
            # 准备参数 (Args + Dependencies)
            call_kwargs = args.copy()

            # 合并静态依赖和动态上下文
            runtime_deps = self.dependency_map.copy()
            if context:
                runtime_deps.update(context)

            sig = inspect.signature(func)
            for param_name in sig.parameters:
                # 如果参数在依赖中，且未在 args 中显式提供，则注入
                if param_name in runtime_deps and param_name not in call_kwargs:
                    call_kwargs[param_name] = runtime_deps[param_name]

            try:
                if inspect.iscoroutinefunction(func):
                    return await func(**call_kwargs)
                else:
                    return func(**call_kwargs)
            except Exception as e:
                logger.error(f"本地工具执行错误: {e}", exc_info=True)
                return f"本地工具错误: {str(e)}"

        # 2. 尝试 MCP 工具 (name 格式 server__tool)
        if "__" in name:
            server_name = name.split("__")[0]
            if server_name in self._mcp_clients:
                try:
                    return await self._mcp_clients[server_name].call_tool(name, args)
                except Exception as e:
                    return f"MCP工具错误: {str(e)}"

        return f"错误: 未找到工具 '{name}'"

    async def _attempt_self_heal(self, tool_name: str, original_args: Dict[str, Any], error_msg: str) -> Optional[
        Dict[str, Any]]:
        """
        [Reflexive Self-Healing]
        调用 LLM 上下文来分析错误并修正参数。
        """

        # 查找工具 Schema 以提供正确的格式参考
        schema = next((s for s in self._schemas if s["function"]["name"] == tool_name), None)
        schema_str = json.dumps(schema, ensure_ascii=False) if schema else "Unknown Schema"

        prompt = f"""
你是一个自动纠错系统。Agent 刚才尝试调用工具 `{tool_name}` 失败了。
请根据错误信息修正调用参数。

【工具定义】
{schema_str}

【原始调用参数】
{json.dumps(original_args, ensure_ascii=False)}

【错误信息】
{error_msg}

请输出修正后的 JSON 参数 (仅 JSON):
"""
        try:
            # 使用 API Client 调用 (强制 JSON 模式)
            response = await self.api_client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                schema={
                    "type": "object",
                    "description": "Fixed arguments for the tool",
                    "additionalProperties": True
                },
                tool_choice="none"
            )

            content = response["choices"][0]["message"]["content"]
            fixed_args = json.loads(content)

            # 简单的防呆检查：防止返回空或者完全不相关的结构
            if not fixed_args and original_args:
                return None

            return fixed_args

        except Exception as e:
            logger.error(f"Self-healing failed: {e}")
            return None

    async def reload_tool_module(self, module_name: str) -> str:
        """
        [系统核心] 热重载指定的工具模块。
        """
        # 1. 检查模块是否已加载
        if module_name not in sys.modules:
            return f"模块 {module_name} 未加载，无法重载。"

        try:
            # 2. 获取旧模块对象
            module = sys.modules[module_name]

            # 3. 使用 importlib.reload 强制重新编译并执行模块代码
            importlib.reload(module)

            # 4. 重新注册该模块下的工具
            # 注意：这需要你的 _load_local_tools 逻辑能处理更新
            # 简单做法是：重新运行一遍 _load_local_tools 里的扫描逻辑，或者针对该模块单独处理

            # 清除旧的 schema 缓存（简化处理，这里建议重新生成全部）
            self._local_tools.clear()
            self._schemas.clear()
            self._load_local_tools()  # 重新扫描所有，确保依赖关系正确

            logger.info(f"模块 {module_name} 热重载成功")
            return f"模块 {module_name} 已重载，新代码已生效。"

        except Exception as e:
            logger.error(f"热重载失败: {e}", exc_info=True)
            return f"重载失败: {str(e)}"
