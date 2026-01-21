import asyncio
import json
import logging
import traceback
from typing import List, Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.gui.monitor_registry import monitor_registry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, Action, DetailType, EventType
from core.kernel.prompt import PromptManager
from core.limbic.manager import LimbicManager
from core.memory.hippocampus import Hippocampus
from core.memory.infinite_context import InfiniteContextManager
from core.memory.vector_store import VectorStore
from core.social.manager import UserManager
from core.tool_manager.aggregator import ToolManager
from core.utilities import calculate_tokens

logger = logging.getLogger(__name__)


class AutonomousAgent:
    def __init__(self, config: Config, event_bus: EventBus, database: Database):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.api_client = GenericAPIClient(config)
        self.prompt_manager = PromptManager(config)

        # --- 内部状态 ---
        self.history: List[Dict[str, Any]] = []
        self.scratchpad: Dict[str, Any] = {
            "current_goal": "",
            "subtasks": [],
            "variables": {},
            "progress_summary": ""
        }
        self.last_response_content = ""  # 用于死锁检测
        self.last_observation_text = None
        self.consecutive_idle_count = 0  # 空转计数器

        # --- 初始化工具管理器 (注入依赖) ---
        self.tool_manager = ToolManager(
            config=config,
            event_bus=event_bus,
            api_client=self.api_client,
            database=database,
            agent_state=self.scratchpad
        )

        # 初始化记忆组件
        self.hippocampus = Hippocampus(config, self.api_client, database, self.history)
        self.vector_store = VectorStore(database, self.api_client)

        # 初始化边缘系统
        self.limbic = LimbicManager(config, database, event_bus, self.api_client)

        # 初始化用户管理器
        self.user_manager = UserManager(database)

        # 初始化任务调度器
        self.scheduler = AsyncIOScheduler()

        # 初始化无限上下文管理器
        self.context_manager = InfiniteContextManager(config, self.api_client)

        # 记录唤醒任务的 ID
        self.wakeup_job_id = None

        # 睡眠状态标记
        self.is_sleeping = False

        # 注入 Scheduler 和 Agent 自身
        self.tool_manager.add_dependency("agent", self)
        self.tool_manager.add_dependency("scheduler", self.scheduler)
        self.tool_manager.add_dependency("limbic", self.limbic)
        self.tool_manager.add_dependency("user_manager", self.user_manager)

        # 消息缓冲区 (处理中断)
        self.incoming_events: asyncio.Queue = asyncio.Queue()
        self.event_bus.subscribe_event(self._enqueue_event)

    async def _enqueue_event(self, event: OneBotEvent):
        """回调：将总线事件放入缓冲区"""
        await self.incoming_events.put(event)

    def _prune_context(self):
        """
        [v1 移植 - 完整版]
        检查并修剪对话历史，防止超过 Token 限制。
        此函数作为 LLM 调用前的一个预处理，基于对中英文和图片Token的估算。
        """
        # 尝试从配置获取上下文限制
        TOKEN_LIMIT_APPROX = self.config.get("llm.model_context", 16384)
        SAFE_LIMIT = TOKEN_LIMIT_APPROX - 500

        current_estimated_tokens = 0

        # 1. 计算当前总 Token
        for msg in self.history:
            current_estimated_tokens += calculate_tokens(msg.get("content"))

        # 2. 永远保留第一条系统提示 (self.history[0])
        # 从最旧的对话（即索引1开始）移除，直到估算总 Token 数回到限制之下
        # 同时保留最近的 5 条消息作为短期记忆保护区
        while len(self.history) > 6 and current_estimated_tokens > SAFE_LIMIT:
            # 移除第二条消息（最旧的对话，index 0 是 system prompt）
            removed_message = self.history.pop(1)

            removed_cost = calculate_tokens(removed_message.get("content"))
            current_estimated_tokens -= removed_cost

            logger.info(
                f"✂️ [上下文管理] 对话过长，已移除一条旧消息。"
                f"当前估算: {int(current_estimated_tokens)}/{SAFE_LIMIT}, "
                f"剩余消息: {len(self.history)}"
            )

        return current_estimated_tokens

    async def run_autonomous_loop(self):
        """
        [Core Loop] 无限自主循环
        """
        logger.info("Agent 核心循环已启动...")

        # 启动调度器
        self.scheduler.start()
        logger.info("任务调度器已启动")

        # 初始化 Social DB
        await self.user_manager.initialize()

        # 异步加载工具 (包括 MCP)
        logger.info("正在加载工具组件...")
        await self.tool_manager.initialize()

        # 初始化边缘系统数据库
        await self.limbic.initialize()

        # 1. 注入初始系统上下文
        system_prompt = self.prompt_manager.get_system_prompt()
        self.history.append({"role": "system", "content": system_prompt})

        # 2. 启动海马体后台任务
        asyncio.create_task(self.hippocampus.start())

        # 3. 启动边缘系统后台任务
        asyncio.create_task(self.limbic.start())

        # 3. 注入启动信号
        self.history.append({
            "role": "user",
            "content": f"系统启动完成。"
        })

        while True:
            try:
                # --- A. 感知阶段 (Perception) ---
                event = None

                # 检查缓冲区（非阻塞优先）
                if not self.incoming_events.empty():
                    event = self.incoming_events.get_nowait()
                    self.is_sleeping = False
                elif self.is_sleeping:
                    # 休眠模式：阻塞等待
                    event = await self.incoming_events.get()
                    self.is_sleeping = False  # 收到事件，解除休眠
                    logger.info(f"⏰ Agent 结束休眠，收到事件: {event.type}")
                else:
                    # 如果处于活跃状态，使用短超时轮询，保持自主思考能力
                    try:
                        event = await asyncio.wait_for(self.incoming_events.get(), timeout=0.1)
                        self.is_sleeping = False
                    except asyncio.TimeoutError:
                        pass

                if event:
                    await self._process_incoming_event(event)
                    # 如果收到新事件，重置空转计数器
                    self.consecutive_idle_count = 0
                    if self.wakeup_job_id:
                        try:
                            self.scheduler.remove_job(self.wakeup_job_id)
                            logger.debug(f"已取消剩余的唤醒定时器: {self.wakeup_job_id}")
                            self.wakeup_job_id = None
                        except:
                            pass

                # --- B. 思考与决策阶段 (Thought) ---

                # [Infinite Context] 智能压缩上下文
                await self.context_manager.compress_if_needed(self.history)
                # [v1 移植] 上下文修剪
                self._prune_context()

                # 主动记忆检索 (RAG)
                retrieved_memories = await self._active_retrieval()

                # 动态生成 System Prompt
                # 1. 获取当前神经状态
                current_neuro_state = await self.limbic.get_state()

                # 2. 生成带有状态描述的 Prompt
                system_prompt_base = self.prompt_manager.get_system_prompt(
                    neuro_state=current_neuro_state,
                    memory_context=retrieved_memories
                )

                # 3. 附加 Scratchpad
                scratchpad_dump = json.dumps(self.scratchpad, indent=2, ensure_ascii=False)

                monitor_registry.register_text_source(
                    "认知", "Scratchpad", lambda: scratchpad_dump
                )

                final_system_prompt = (
                    f"{system_prompt_base}\n\n"
                    f"## Scratchpad\n"
                    f"这是你必须维护的内部状态，每次响应必须更新此状态：\n"
                    f"{scratchpad_dump}"
                )

                monitor_registry.register_text_source(
                    "系统", "System Prompt",
                    lambda: final_system_prompt
                )

                # 更新历史记录中的 System Prompt
                if self.history and self.history[0]["role"] == "system":
                    self.history[0]["content"] = final_system_prompt
                else:
                    self.history.insert(0, {"role": "system", "content": final_system_prompt})

                with open("data/messages_in_memory.txt", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self.history, ensure_ascii=False, indent=4))

                # 调用 LLM
                response_msg = await self._call_llm()
                content_str = response_msg.get("content")

                if isinstance(content_str, str):
                    content_str = (content_str
                                   .replace("```json", "")
                                   .replace("```", "")
                                   .strip())
                else:
                    content_str = "{}"

                native_tool_calls = response_msg.get("tool_calls", [])
                # 注入 Assistant 历史记录
                # 如果存在原生 tool_calls，必须保留完整结构，否则后续 role: tool 会报错
                if native_tool_calls:
                    # 复制消息对象以确保存入的是符合 API 标准的字典
                    msg_entry = response_msg.copy()
                    # 确保 content 字段存在（即使为空）
                    if "content" not in msg_entry or msg_entry["content"] is None:
                        msg_entry["content"] = ""
                    self.history.append(msg_entry)
                else:
                    # Schema 模式：仅存入文本内容
                    try:
                        if content_str.strip().startswith("{"):
                            formatted_content = json.dumps(json.loads(content_str), ensure_ascii=False)
                            self.history.append({"role": "assistant", "content": formatted_content})
                        else:
                            self.history.append({"role": "assistant", "content": content_str})
                    except:
                        self.history.append({"role": "assistant", "content": content_str})

                # [v1 移植] 死锁检测
                if content_str and content_str == self.last_response_content and not native_tool_calls:
                    logger.warning("⚠️ 检测到内容重复死锁。")
                    self.history.append({
                        "role": "user",
                        "content": "SYSTEM WARNING: 你输出的内容与上一次完全一致，且未执行任何操作。请改变策略，或使用 wait 工具挂起。"
                    })
                    self.last_response_content = ""  # 重置以允许下一次尝试
                    continue  # 跳过本次处理，直接进入下一轮接收系统警告

                self.last_response_content = content_str

                # --- C. 行动阶段 (Action) ---

                tool_execution_queue = []  # 待执行任务列表: {name, args, id(可选)}
                thought_content = ""

                # 1. 尝试解析 Schema 模式的 JSON
                if content_str:
                    try:
                        parsed_data = json.loads(content_str)
                        thought_content = parsed_data.get("thought", "")

                        # 更新 Scratchpad
                        new_scratchpad = parsed_data.get("scratchpad", None)
                        if new_scratchpad and isinstance(new_scratchpad, dict):
                            self.scratchpad = new_scratchpad
                            self.tool_manager.agent_state = self.scratchpad

                        # 提取 Schema Tool Calls
                        schema_calls = parsed_data.get("tool_calls", [])
                        for tc in schema_calls:
                            tool_execution_queue.append({
                                "name": tc.get("name"),
                                "args": tc.get("arguments"),
                                "id": None  # Schema 模式没有 ID
                            })

                    except json.JSONDecodeError as e:
                        # 如果是原生模式且没有 JSON 内容，这是正常的，忽略错误
                        if not native_tool_calls:
                            logger.error(f"JSON 解析失败: {e}\n{content_str}")
                            self.history.append({
                                "role": "user",
                                "content": f"SYSTEM ERROR: JSON Format Error: {e}"
                            })
                            continue

                # 2. 提取 Native Tool Calls
                if native_tool_calls:
                    for tc in native_tool_calls:
                        # 兼容 object (OpenAI Object) 和 dict
                        func = tc.function if hasattr(tc, 'function') else tc.get("function", {})
                        t_id = tc.id if hasattr(tc, 'id') else tc.get("id")

                        name = func.name if hasattr(func, 'name') else func.get("name")
                        args = func.arguments if hasattr(func, 'arguments') else func.get("arguments")

                        tool_execution_queue.append({
                            "name": name,
                            "args": args,  # 原生 args 通常是 JSON 字符串
                            "id": t_id
                        })

                # 记录思考
                if thought_content:
                    self.event_bus.publish_action(Action(
                        action="broadcast_log",
                        params={"content": f"💭 {thought_content}"}
                    ))

                # 3. 执行工具列表
                if tool_execution_queue:
                    # 有工具执行，重置空转计数器
                    self.consecutive_idle_count = 0

                    for task in tool_execution_queue:
                        name = task["name"]
                        args = task["args"]
                        t_id = task["id"]

                        # 参数清洗与解析
                        if isinstance(args, str):
                            try:
                                args = json.loads(args
                                                  .replace("```json", "")
                                                  .replace("```", "")
                                                  .strip())
                            except json.JSONDecodeError:
                                logger.warning(f"工具 {name} 参数解析失败: {args}")
                                args = {}

                        # 聊天消耗能量逻辑
                        if name in ["send_message"]:
                            current_neuro_state = await self.limbic.get_state()
                            self.limbic.homeostasis.consume_resource(current_neuro_state, "chat")

                        try:
                            self.event_bus.publish_action(Action(
                                action="broadcast_log",
                                params={"content": f"🛠️ 调用: {name}({args})"}
                            ))

                            # 执行工具
                            result = await self.tool_manager.execute_tool(name, args)
                            logger.debug(f"工具 {name} 执行结果：" + json.dumps(result, ensure_ascii=False))

                            # 结果回填历史
                            # 如果有 ID (原生模式)，必须带上 tool_call_id
                            if t_id:
                                self.history.append({
                                    "role": "tool",
                                    "tool_call_id": t_id,
                                    "name": name,
                                    "content": str(result)
                                })
                            else:
                                # Schema 模式
                                self.history.append({
                                    "role": "tool",
                                    "name": name,
                                    "content": str(result)
                                })

                        except Exception as e:
                            logger.error(f"工具执行错误: {e}")
                            traceback.print_exc()
                            error_msg = f"Error: {str(e)}"

                            if t_id:
                                self.history.append({
                                    "role": "tool",
                                    "tool_call_id": t_id,
                                    "name": name,
                                    "content": error_msg
                                })
                            else:
                                self.history.append({
                                    "role": "tool",
                                    "name": name,
                                    "content": error_msg
                                })

                # 空转检测与熔断
                else:
                    # 没有执行任何工具
                    self.consecutive_idle_count += 1
                    logger.warning(f"⚠️ 空转检测: {self.consecutive_idle_count}/3")

                    if self.consecutive_idle_count >= 3:
                        logger.warning("🚫 触发空转熔断：强制注入警告。")
                        self.history.append({
                            "role": "user",
                            "content": "SYSTEM WARNING: 检测到你连续多次进行思考但未执行任何操作（未调用工具）。\n"
                                       "1. 如果你在等待用户回复，必须调用 `wait` 工具挂起。\n"
                                       "2. 如果你任务已完成，请调用 `wait` 进入待命。\n"
                                       "3. 禁止无意义的循环思考。"
                        })
                        # 重置计数器以免一直刷屏，或者让模型有机会反应
                        self.consecutive_idle_count = 0

                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                await asyncio.sleep(5)  # 出错冷却

    def _transcribe_event(self, event: OneBotEvent) -> str:
        """
        [事件转译层] 将系统事件转化为 LLM 可理解的自然语言描述
        """
        # 1. 处理内部驱动 (生理需求)
        if event.detail_type == DetailType.INTERNAL_DRIVE:
            # 解析 raw_data 中的驱动力信息
            drive_name = event.raw_data.get("drive", "unknown")
            desc = event.raw_data.get("description", "")
            return f"【生理信号】{desc} (驱动力: {drive_name})，请决定是否采取行动。"

        # 2. 处理通知 (Notice)
        if event.type == EventType.NOTICE:
            if event.detail_type == "group_member_increase":
                return f"【系统通知】用户 {event.source.user_id} 加入了群聊 {event.source.group_id}。"
            elif event.detail_type == "group_member_decrease":
                return f"【系统通知】用户 {event.source.user_id} 离开了群聊 {event.source.group_id}。"
            return f"【系统通知】检测到事件: {event.detail_type}"

        # 3. 处理请求 (Request)
        if event.type == EventType.REQUEST:
            if event.detail_type == "friend":
                return f"【好友申请】收到来自用户 {event.source.user_id} 的好友申请。"

        # 4. 兜底策略：如果是复杂的未知事件，才使用简化版 JSON
        simple_data = {k: v for k, v in event.model_dump().items() if k in ['type', 'detail_type', 'source']}
        return f"【未知信号】系统接收到底层事件: {json.dumps(simple_data, ensure_ascii=False)}"

    async def _process_incoming_event(self, event: OneBotEvent):
        """
        处理外部事件：
        1. 让边缘系统“感受”刺激 (Process Stimulus)
        2. 将事件写入历史记录
        """
        if event.type == EventType.META:
            return

        # 1. 边缘系统介入 (只处理文本消息)
        # 只有真实人类的消息才算作"刺激"，内部信号不算
        if event.type == EventType.MESSAGE and isinstance(event.message, str):
            # 异步调用，不阻塞主流程太多
            asyncio.create_task(self.limbic.process_stimulus(event.message))
            self.last_observation_text = event.message

        # 自动捕获用户
        # 尝试从 source 中获取用户信息
        adapters = self.tool_manager.dependency_map.get("adapters")
        adapter_names = []
        for i, adapter in enumerate(adapters, 1):
            adapter_names.append(getattr(adapter, "platform_name", "Unknown"))

        if event.source.user_id and event.source.platform in adapter_names:
            platform = event.source.platform
            raw_id = event.source.user_id

            # 1. 计算 PUID
            puid = f"{platform}:{raw_id}"

            # 2. 查询用户 (只读)
            user_profile = await self.user_manager.get_user(puid)

            # 3. 注入上下文 (Scratchpad)
            interactor_info = {
                "puid": puid,
                "platform": platform,
                "user_id": raw_id,
            }

            if user_profile:
                # [熟人]
                interactor_info.update({
                    "status": "KNOWN",
                    "nickname": user_profile.nickname,
                    "relationship_tags": user_profile.relationship_tags,
                    "favorability": user_profile.favorability,
                    "trust": user_profile.trust,
                    "impression": user_profile.impression
                })
            else:
                # [陌生人]
                interactor_info.update({
                    "status": "STRANGER",
                    "note": "User not in database. Use tool `social_register_user` if you wish to remember them."
                })

            # 更新 Scratchpad
            self.scratchpad["current_interactor"] = interactor_info
            monitor_registry.register_text_source(
                "认知", "人员注入",
                lambda: json.dumps(interactor_info, indent=2, ensure_ascii=False)
            )

        # 2. 序列化事件
        event_data = event.model_dump(exclude_none=True)
        # 清理冗余字段
        for field in ["id", "time", "raw_data"]:
            if field in event_data: del event_data[field]
        if "message" in event_data and "alt_message" in event_data:
            del event_data["message"]

        if event.detail_type == DetailType.INTERNAL_DRIVE:
            # 强制唤醒
            self.is_sleeping = False

        # 3. 更新 Scratchpad 上下文 (如果是消息事件)
        if event.source.platform in adapter_names:
            source = event.source
            ctx_type = "group" if source.group_id else "private"
            ctx_id = source.group_id if source.group_id else source.user_id

            self.scratchpad["last_context"] = {
                "platform": source.platform,
                "type": ctx_type,
                "id": ctx_id
            }

        # 1. 决定消息内容
        if event.type == EventType.MESSAGE:
            # 正常对话消息，直接使用
            content_msg = f"接收到用户消息：{json.dumps(event_data, ensure_ascii=False)}"
            is_ephemeral = False  # 对话消息需要被记忆
        else:
            # 非对话事件，进行自然语言转译
            content_msg = self._transcribe_event(event)
            is_ephemeral = True  # 标记为瞬时消息，不需要存入长时记忆

        # 2. 构建历史记录对象 (增加了 metadata 字段)
        history_item = {
            "role": "user",
            "content": str(content_msg),
            "metadata": {
                "type": event.type,
                "ephemeral": is_ephemeral,
                "raw_event_id": event.id
            }
        }

        self.history.append(history_item)
        logger.info(f"Event Ingested: {event.type}.{event.detail_type}")

    async def _active_retrieval(self) -> List[str]:
        """主动记忆检索逻辑"""
        # 1. 获取当前交互对象的 PUID
        interactor = self.scratchpad.get("current_interactor", {})
        puid = interactor.get("puid", "global")

        # 2. 构建查询语句 (Query)
        # 策略：结合 "当前正在做的事(Goal)" 和 "刚才听到的话(Observation)"
        query_parts = []

        current_goal = self.scratchpad.get("current_goal", "")
        if current_goal:
            query_parts.append(f"关注点: {current_goal}")

        if self.last_observation_text:
            query_parts.append(f"上下文: {self.last_observation_text}")

        if not query_parts:
            # 如果什么都没有，就不浪费 Token 去搜了
            return []

        query = " ".join(query_parts)

        try:
            # 调用向量存储进行检索
            # limit=3 避免上下文过长，只取最相关的
            logger.debug(f"🔍 执行主动记忆检索: {query[:50]}... (PUID: {puid})")
            memories = await self.vector_store.search_memory(query, puid, limit=3)
            if memories:
                logger.info(f"📚 检索到 {len(memories)} 条相关记忆")
                monitor_registry.register_text_source(
                    "认知", "主动记忆",
                    lambda: "\n".join(memories)
                )
            return memories
        except Exception as e:
            logger.warning(f"记忆检索异常: {e}")
            return []

    async def _call_llm(self) -> Dict[str, Any]:
        """封装 API 调用"""
        schemas = self.tool_manager.get_tool_schemas()

        use_schema_tools = self.config.get("llm.use_schema_tool_calls", True)
        arg_mode = self.config.get("llm.tool_call_arg_mode", "object")

        # 基础结构
        properties = {
            "thought": {
                "type": "string",
                "description": "思考过程和行动规划。"
            },
            "scratchpad": {
                "type": "object",
                "properties": {
                    "current_goal": {"type": "string", "description": "当前目标"},
                    "subtasks": {
                        "type": "array",
                        "description": "任务列表。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "description": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "working", "done", "failed"]}
                            },
                            "required": ["id", "description", "status"],
                            "additionalProperties": False
                        }
                    },
                    "progress_summary": {"type": "string", "description": "已完成的工作"}
                },
                "required": ["current_goal", "subtasks", "progress_summary"],
                "additionalProperties": False
            }
        }

        required_fields = ["thought", "scratchpad"]

        # 根据配置决定是否将 tool_calls 注入 Schema
        if use_schema_tools:
            # 根据配置决定 arguments 是 object 还是 string
            arg_schema = {"type": "object"} if arg_mode == "object" else {"type": "string",
                                                                          "description": "工具的参数对象，JSON格式。例如 '{\"url\": \"https://google.com\"}'"}

            properties["tool_calls"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": arg_schema
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False
                }
            }
            required_fields.append("tool_calls")

        # 定义强制思维 Schema (JSON Schema)
        thought_structure = {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False
        }

        sanitized_history = [
            {k: v for k, v in d.items() if k != 'metadata'}
            for d in self.history
        ]

        # 调用 API，同时传入 tools 和 schema
        response = await self.api_client.create_chat_completion(
            messages=sanitized_history,
            tools=schemas if schemas else None,
            schema=thought_structure
        )

        return response["choices"][0]["message"]
