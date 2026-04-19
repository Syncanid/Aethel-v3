# core/tool_manager/aggregator.py
import asyncio
import hashlib
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
from core.tool_manager.mcp_client import MCPClient
from core.tool_manager.registry import get_pending_functions, clear_pending
from core.tool_manager.schema_utils import SchemaGenerator

logger = logging.getLogger(__name__)


class ToolManager:
    def __init__(self, tools_dir: str, config: Config, event_bus: EventBus, api_client: GenericAPIClient,
                 database: Database,
                 agent_state: Dict[str, Any]):
        self.tools_dir = tools_dir
        self.config = config
        self.event_bus = event_bus
        self.api_client = api_client
        self.database = database
        self.agent_state = agent_state

        self._local_tools: Dict[str, Any] = {}
        self._mcp_clients: Dict[str, MCPClient] = {}
        self._schemas: List[Dict[str, Any]] = []

        self._base_local_tools: Dict[str, Any] = {}
        self._base_schemas: List[Dict[str, Any]] = []
        self._active_skill: Optional[str] = None

        # 自愈配置
        self.max_self_heal_attempts = 1

        # 用于防死锁检测的签名记录
        self._last_failed_signature: Optional[str] = None
        self._failed_signature_count = 0

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
        """
        初始化工具管理器，动态扫描并挂载工具模块。
        """
        # 从路径中提取系统标识 (例如 "System1" 或 "System2")
        system_name = os.path.basename(os.path.normpath(self.tools_dir))
        config_changed = False

        if not os.path.exists(self.tools_dir):
            logger.warning(f"工具目录不存在: {self.tools_dir}")
            return

        for root, _, files in os.walk(self.tools_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    module_name = file[:-3] # 去除 .py 后缀

                    # 1. 构造配置键名 (例如: tools.System1.core_tool)
                    config_key = f"tools.{system_name}.{module_name}"

                    # 2. 读取配置
                    is_enabled = self.config.get(config_key)

                    # 3. 如果配置中不存在该文件开关，则自动生成并默认开启
                    if is_enabled is None:
                        self.config.set(config_key, True)
                        is_enabled = True
                        config_changed = True
                        logger.info(f"✨ 发现新工具文件 [{system_name}/{module_name}]，已自动在配置中注册并默认启用。")

                    # 4. 拦截判定
                    if not is_enabled:
                        logger.info(f"🚫 工具文件 [{system_name}/{module_name}] 已在配置中被禁用，跳过加载。")
                        continue

                    # 5. 允许加载：拼接完整的模块导入路径
                    # 假设 self.tools_dir 是 "tools/System1"
                    rel_path = os.path.relpath(root, start=os.getcwd())
                    module_path = rel_path.replace(os.sep, '.') + f".{module_name}"

                    try:
                        importlib.import_module(module_path)
                        logger.debug(f"成功加载工具模块: {module_path}")
                    except Exception as e:
                        logger.error(f"加载工具模块 {module_path} 失败: {e}", exc_info=True)

        # 6. 如果在扫描期间发现了新文件并补充了配置，触发物理落盘
        if config_changed:
            self.config.save()

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
                    logger.error(f"MCP服务器 {name} 加载失败: {e}", exc_info=True)

        self._base_local_tools = self._local_tools.copy()
        self._base_schemas = list(self._schemas)
        logger.info(f"💾 工具环境基线已保存，共 {len(self._base_local_tools)} 个本地工具。")

    def _load_local_tools(self):
        """扫描 tools 目录并注册"""
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir)

        system_name = os.path.basename(os.path.normpath(self.tools_dir))
        config_changed = False

        for filename in os.listdir(self.tools_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_short_name = filename[:-3]
                config_key = f"tools.{system_name}.{module_short_name}"

                # 1. 检查配置，如果不存在则自动生成并默认开启
                is_enabled = self.config.get(config_key)
                if is_enabled is None:
                    self.config.set(config_key, True)
                    is_enabled = True
                    config_changed = True
                    logger.info(f"✨ 发现新工具文件 [{system_name}/{module_short_name}]，已自动注册并默认启用。")

                # 2. 如果配置为 false，直接阻断物理导入
                if not is_enabled:
                    logger.info(f"🚫 工具文件 [{system_name}/{module_short_name}] 已被禁用，跳过主动加载。")
                    continue

                # 3. 允许加载
                module_name = f"{self.tools_dir.replace('/', '.')}.{module_short_name}"
                try:
                    importlib.import_module(module_name)
                    logger.debug(f"已导入工具: {module_name}")
                except Exception as e:
                    logger.error(f"工具组件 {module_name} 导入失败: {e}", exc_info=True)

        # 如果在此次扫描中发现了新文件，触发配置落盘保存
        if config_changed:
            self.config.save()

        # 获取所有通过 @register 注册的函数
        pending = get_pending_functions()

        for func, name in pending:
            tool_name = name or func.__name__

            module_path = getattr(func, "__module__", "")
            if module_path.startswith("tools."):
                parts = module_path.split('.')
                if len(parts) >= 3:
                    sys_name = parts[1]
                    file_name = parts[2]
                    config_key = f"tools.{sys_name}.{file_name}"

                    # 检查其原产地文件是否在配置中被明确禁用
                    if self.config.get(config_key) is False:
                        logger.info(f"🛡️ 深度拦截: 剔除被动连带导入的禁用工具 [{tool_name}] (源自 {file_name}.py)")
                        continue

            # 生成 Schema (跳过依赖注入参数)
            schema = SchemaGenerator.get_function_schema(func, tool_name)

            # 过滤 Schema 中的依赖参数 (config, event_bus 等)
            props = schema["function"]["parameters"]["properties"]
            required = schema["function"]["parameters"]["required"]

            for dep in list(self.dependency_map.keys()):
                if dep in props:
                    del props[dep]
                if dep in required:
                    required.remove(dep)

            self._local_tools[tool_name] = func
            self._schemas.append(schema)
            logger.debug(f"本地工具已注册: {tool_name}")

        clear_pending()

    def mount_skill_tools(self, skill_name: str):
        """动态挂载特定技能专属的代码工具"""
        self.unmount_skill_tools()  # 先清理上一任，确保环境干净

        skill_tools_path = os.path.join("data", "skills", skill_name, "tools.py")
        if not os.path.exists(skill_tools_path):
            return  # 没有专属代码工具，直接返回

        module_name = f"skill_tools_{skill_name.replace('-', '_')}"

        try:
            # 动态加载绝对路径下的 python 模块
            spec = importlib.util.spec_from_file_location(module_name, skill_tools_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 捕获该 tools.py 中通过 @register 注册的专属函数
            pending = get_pending_functions()
            mounted_count = 0
            for func, name in pending:
                tool_name = name or func.__name__
                schema = SchemaGenerator.get_function_schema(func, tool_name)

                # 处理依赖注入剔除
                props = schema["function"]["parameters"]["properties"]
                required = schema["function"]["parameters"]["required"]
                for dep in list(self.dependency_map.keys()):
                    if dep in props: del props[dep]
                    if dep in required: required.remove(dep)

                # 将专属工具塞入当前 S2 引擎
                self._local_tools[tool_name] = func
                self._schemas.append(schema)
                mounted_count += 1
                logger.debug(f"🔧 已动态挂载技能专属工具: {tool_name}")

            clear_pending()
            self._active_skill = skill_name
            if mounted_count > 0:
                logger.info(f"🧩 技能 [{skill_name}] 挂载完毕，共载入 {mounted_count} 个专属工具。")

        except Exception as e:
            logger.error(f"挂载技能 {skill_name} 专属工具失败: {e}", exc_info=True)

    def unmount_skill_tools(self):
        """卸载当前技能专属工具，恢复到 System 2 基础环境"""
        if self._active_skill:
            self._local_tools = self._base_local_tools.copy()
            self._schemas = list(self._base_schemas)

            module_name = f"skill_tools_{self._active_skill.replace('-', '_')}"
            if module_name in sys.modules:
                del sys.modules[module_name]  # 从内存剔除模块

            logger.info(f"🧹 技能 [{self._active_skill}] 专属工具已卸载，S2 环境已复原。")
            self._active_skill = None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    async def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = None,
                           allow_self_heal: bool = True) -> str:
        """
        统一执行入口，包含自愈机制。
        """
        # --- 1. 计算调用签名，防死锁拦截 ---
        args_str = json.dumps(args, sort_keys=True)
        current_signature = hashlib.md5(f"{name}:{args_str}".encode()).hexdigest()

        # 检查模型是否在机械重复上一次的失败动作
        if current_signature == self._last_failed_signature:
            self._failed_signature_count += 1
            if self._failed_signature_count >= 2:
                logger.warning(f"🛑 [防死锁熔断] 检测到模型连续重复提交必然失败的动作: {name}")
                return (
                    f"【SYSTEM ERROR - DEADLOCK INTERCEPTED】防死锁系统已拦截此请求！\n"
                    f"你正在重复执行与上一次完全相同的错误操作，这证明你的策略已陷入无限死循环。\n"
                )
        else:
            # 如果动作改变，重置死锁计数器
            self._last_failed_signature = None
            self._failed_signature_count = 0

        # --- 2. 瞬态错误重试与分类执行环 ---
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                result = await self._execute_tool_internal(name, args, context)

                # 检查内置状态错误 (如 Code Interpreter 的特殊返回)
                if isinstance(result, dict) and result.get("status") == "error":
                    raise RuntimeError(f"Semantic/Execution Error: {result.get('stderr')}")
                if isinstance(result, str) and ("Syntax Error" in result or "Traceback" in result):
                    raise RuntimeError(f"Semantic/Execution Error: {result}")

                # 成功执行，清除失败记录并正常返回
                self._last_failed_signature = None
                self._failed_signature_count = 0
                return result

            except Exception as e:
                error_msg = str(e)
                error_lower = error_msg.lower()

                # A. 瞬态错误 - 指数退避静默重试
                is_transient = any(k in error_lower for k in ["timeout", "connection", "502", "503", "rate limit", "too many requests"])
                if is_transient:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"🌐 瞬态网络错误 [{name}] ({error_msg})，{delay}s 后进行第 {attempt+1} 次重试...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # 重试耗尽，降级为终端错误
                        self._last_failed_signature = current_signature
                        return f"【Terminal Error】网络或服务持续不可用。错误详情：{error_msg}。请暂时放弃使用此工具。"

                # B. 终端错误 (Terminal Error) - 直接拦截
                is_terminal = any(k in error_lower for k in ["permission denied", "unauthorized", "401", "403", "not found"])
                if is_terminal:
                    self._last_failed_signature = current_signature
                    return f"【Terminal Error】权限被拒绝或目标不存在。错误详情：{error_msg}。请改变计划。"

                # C. 语义错误 (Semantic Error) - 触发自愈
                self._last_failed_signature = current_signature
                should_heal = any(k in error_lower for k in ["argument", "missing", "type", "value", "json", "format", "invalid"])

                if allow_self_heal and self.max_self_heal_attempts > 0 and should_heal:
                    logger.info(f"🩹 触发参数级自愈回路: {name}")
                    fixed_args = await self._attempt_self_heal(name, args, error_msg)

                    if fixed_args:
                        # 解耦式的认知更新：尝试获取 TaskEngine 或 Agent 的 history
                        notice_content = f"【系统自愈报告】工具 `{name}` 初次调用由于格式错误失败。系统已自动修正参数重试。"

                        task_engine = self.dependency_map.get("task_engine")
                        if task_engine and hasattr(task_engine, "history"):
                            task_engine.history.append({"role": "system", "content": notice_content})
                        else:
                            agent = self.dependency_map.get("agent")
                            if agent and hasattr(agent, "history"):
                                agent.history.append({"role": "system", "content": notice_content})

                        retry_result = await self.execute_tool(name, fixed_args, context, allow_self_heal=False)
                        return f"【系统提示：原参数错误，系统已自动修正为 {fixed_args}】\n执行结果:\n{retry_result}"

                # 彻底失败，返回带有明确指导的语义错误
                return (
                    f"【Semantic Error】工具调用逻辑失败。错误详情：{error_msg}\n"
                    f"请反思 `inner_monologue` 中的逻辑，修改参数后重试，或转换策略。"
                )

        self._last_failed_signature = current_signature
        logger.error(f"🚨 [System Crash] 工具 {name} 的执行流异常跌穿了重试循环！")
        return f"【SYSTEM FATAL】工具执行流崩溃，重试循环未能产生任何有效状态。"

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
                model=self.api_client.small_model,
                schema={
                    "type": "object",
                    "description": "Fixed arguments for the tool",
                    "additionalProperties": True
                },
                require_tools=False
            )
            fixed_args = response.get("content", {})
            if not fixed_args and original_args:
                return None

            return fixed_args

        except Exception as e:
            logger.error(f"Self-healing failed: {e}", exc_info=True)
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

    @property
    def active_skill(self):
        return self._active_skill
