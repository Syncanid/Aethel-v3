# core/tool_manager/aggregator.py
import importlib
import inspect
import logging
import os
from typing import Dict, Any, List

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
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

        # 依赖注入容器
        self._dependency_map = {
            "config": config,
            "event_bus": event_bus,
            "api_client": api_client,
            "database": database,
            "agent_state": agent_state,
            "scheduler": None
        }

        self.scheduler = None

    def set_scheduler(self, scheduler):
        """后续注入调度器"""
        self.scheduler = scheduler
        self._dependency_map["scheduler"] = scheduler

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

            for dep in self._dependency_map.keys():
                if dep in props:
                    del props[dep]
                if dep in required:
                    required.remove(dep)

            self._local_tools[tool_name] = func
            self._schemas.append(schema)
            logger.info(f"本地工具已注册: {tool_name}")

        clear_pending()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    async def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """统一执行入口"""
        # 1. 尝试本地工具
        if name in self._local_tools:
            func = self._local_tools[name]
            # 准备参数 (Args + Dependencies)
            call_kwargs = args.copy()

            # 合并静态依赖和动态上下文
            runtime_deps = self._dependency_map.copy()
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
