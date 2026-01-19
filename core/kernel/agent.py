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
from core.memory.vector_store import VectorStore
from core.social.manager import UserManager
from core.tool_manager.aggregator import ToolManager

logger = logging.getLogger(__name__)


class AutonomousAgent:
    def __init__(self, config: Config, event_bus: EventBus, database: Database):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.api_client = GenericAPIClient(config)
        self.prompt_manager = PromptManager(config)

        # 初始化记忆组件
        self.hippocampus = Hippocampus(config, self.api_client, self.event_bus, database)
        self.vector_store = VectorStore(database, self.api_client)

        # 初始化边缘系统
        self.limbic = LimbicManager(config, database, event_bus, self.api_client)

        # 初始化用户管理器
        self.user_manager = UserManager(database)

        # 初始化任务调度器
        self.scheduler = AsyncIOScheduler()

        # 记录唤醒任务的 ID
        self.wakeup_job_id = None

        # 睡眠状态标记
        self.is_sleeping = False

        # --- 内部状态 ---
        self.history: List[Dict[str, Any]] = []
        self.scratchpad: Dict[str, Any] = {
            "current_goal": "",
            "subtasks": [],
            "variables": {},
            "progress_summary": ""
        }
        self.last_response_content = ""  # 用于死锁检测

        # --- 初始化工具管理器 (注入依赖) ---
        self.tool_manager = ToolManager(
            config=config,
            event_bus=event_bus,
            api_client=self.api_client,
            database=database,
            agent_state=self.scratchpad
        )

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

    def _calculate_tokens(self, content: Any) -> float:
        """
        [辅助方法] 计算单个内容块的 Token 估算值
        逻辑源自 v1 planner.py，区分中英文和图片
        """
        # 定义中英文的字符-Token比例（根据实测数据校准）
        CHARS_PER_CHINESE_TOKEN = 1.0  # 中文：286 字 / 177 Token ≈ 1.6 字/Token
        CHARS_PER_ENGLISH_TOKEN = 5.0  # 英文：1009 字 / 183 Token ≈ 5.5 字/Token
        SCREENSHOT_TOKEN_COST_APPROX = 30  # 1080p 截图的 Token 估算值

        estimated = 0.0

        if isinstance(content, list):  # 处理多模态内容（列表）
            for part in content:
                if part.get("type") == "text":
                    text = str(part.get("text", ""))
                    english_chars = sum(1 for char in text if ord(char) < 128)
                    chinese_chars = len(text) - english_chars
                    estimated += (english_chars / CHARS_PER_ENGLISH_TOKEN) + (chinese_chars / CHARS_PER_CHINESE_TOKEN)
                elif part.get("type") == "image_url" or part.get("type") == "image_base64":
                    estimated += SCREENSHOT_TOKEN_COST_APPROX
        elif isinstance(content, str):
            text = content
            english_chars = sum(1 for char in text if ord(char) < 128)
            chinese_chars = len(text) - english_chars
            estimated += (english_chars / CHARS_PER_ENGLISH_TOKEN) + (chinese_chars / CHARS_PER_CHINESE_TOKEN)

        return estimated

    def _prune_context(self):
        """
        [v1 移植 - 完整版]
        检查并修剪对话历史，防止超过 Token 限制。
        此函数作为 LLM 调用前的一个预处理，基于对中英文和图片Token的估算。
        """
        # 尝试从配置获取上下文限制
        TOKEN_LIMIT_APPROX = self.config.get("llm", {}).get("context_window", 8192)

        # 预留 1000 token 给回复
        SAFE_LIMIT = TOKEN_LIMIT_APPROX - 1000

        current_estimated_tokens = 0

        # 1. 计算当前总 Token
        for msg in self.history:
            current_estimated_tokens += self._calculate_tokens(msg.get("content"))

        # 2. 永远保留第一条系统提示 (self.history[0])
        # 从最旧的对话（即索引1开始）移除，直到估算总 Token 数回到限制之下
        # 同时保留最近的 5 条消息作为短期记忆保护区
        while len(self.history) > 6 and current_estimated_tokens > SAFE_LIMIT:
            # 移除第二条消息（最旧的对话，index 0 是 system prompt）
            removed_message = self.history.pop(1)

            removed_cost = self._calculate_tokens(removed_message.get("content"))
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

        # 3. 初始 Prompt 注入检索到的记忆 (RAG)
        # 简单起见，这里先检索 "Context" 相关的
        mems = await self.vector_store.search_memory("重要信息", "system_boot")
        mem_context = "\n".join(mems) if mems else "暂无记忆"

        # 1. 注入初始系统上下文
        system_prompt = self.prompt_manager.get_system_prompt()
        self.history.append({"role": "system", "content": system_prompt + f"\n\n# Memory Context\n{mem_context}"})

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
                    if self.wakeup_job_id:
                        try:
                            self.scheduler.remove_job(self.wakeup_job_id)
                            logger.debug(f"已取消剩余的唤醒定时器: {self.wakeup_job_id}")
                            self.wakeup_job_id = None
                        except:
                            pass

                # --- B. 思考与决策阶段 (Thought) ---

                # [v1 移植] 上下文修剪
                self._prune_context()

                # 动态生成 System Prompt
                # 1. 获取当前神经状态
                current_neuro_state = await self.limbic.get_state()

                # 2. 生成带有状态描述的 Prompt
                system_prompt_base = self.prompt_manager.get_system_prompt(neuro_state=current_neuro_state)

                # 3. 附加 Scratchpad
                scratchpad_dump = json.dumps(self.scratchpad, indent=2, ensure_ascii=False)

                monitor_registry.register_text_source(
                    "Cognition", "Scratchpad", lambda: scratchpad_dump
                )

                final_system_prompt = (
                    f"{system_prompt_base}\n\n"
                    f"## Scratchpad\n"
                    f"这是你必须维护的内部状态，每次响应必须更新此状态：\n"
                    f"{scratchpad_dump}"
                )

                monitor_registry.register_text_source(
                    "System", "System Prompt",
                    lambda: final_system_prompt
                )

                # 更新历史记录中的 System Prompt
                if self.history and self.history[0]["role"] == "system":
                    self.history[0]["content"] = final_system_prompt
                else:
                    self.history.insert(0, {"role": "system", "content": final_system_prompt})

                # 调用 LLM
                response_msg = await self._call_llm()
                content_str = response_msg.get("content", "")

                # [v1 移植] 死锁检测
                if content_str and content_str == self.last_response_content:
                    logger.warning("⚠️ 检测到死锁：模型输出与上一次完全一致。")
                    self.history.append({
                        "role": "user",
                        "content": "SYSTEM WARNING: 你陷入了死循环。请改变策略。"
                    })
                    self.last_response_content = ""  # 重置以允许下一次尝试
                    continue  # 跳过本次处理，直接进入下一轮接收系统警告

                self.last_response_content = content_str

                # --- C. 行动阶段 (Action) ---

                # 1. 解析 JSON
                try:
                    parsed_data = json.loads(content_str)
                    thought_content = parsed_data.get("thought", "")
                    # 获取工具列表，默认为空列表
                    tool_calls = parsed_data.get("tool_calls", [])
                    new_scratchpad = parsed_data.get("scratchpad", None)

                    # [v1 移植] 强制状态更新
                    if new_scratchpad and isinstance(new_scratchpad, dict):
                        self.scratchpad = new_scratchpad
                        # 同时更新 ToolManager 中的引用
                        self.tool_manager.agent_state = self.scratchpad

                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析失败: {e}")
                    self.history.append({
                        "role": "assistant",
                        "content": content_str
                    })
                    # 注入格式错误提示
                    self.history.append({
                        "role": "user",
                        "content": f"SYSTEM ERROR: JSON Format Error: {e}"
                    })
                    continue

                # 记录思考
                if thought_content:
                    self.event_bus.publish_action(Action(
                        action="broadcast_log",
                        params={"content": f"💭 {thought_content}"}
                    ))

                # 存入历史
                self.history.append({"role": "assistant", "content": content_str})

                # 3. 处理并行工具调用
                if tool_calls:
                    for tool_call in tool_calls:
                        name = tool_call.get("name")
                        args = tool_call.get("arguments", {})

                        # 如果 Agent 决定使用聊天工具，这会消耗能量并恢复社交值
                        # 可以在 execute_tool 内部 hook，或者在这里手动调用 limbic
                        if name in ["send_message"]:
                            self.limbic.homeostasis.consume_resource(current_neuro_state, "chat")

                        try:
                            self.event_bus.publish_action(Action(
                                action="broadcast_log",
                                params={"content": f"🛠️ 调用: {name}({args})"}
                            ))

                            # 执行工具
                            result = await self.tool_manager.execute_tool(name, args)

                            logger.debug("执行结果：" + json.dumps(result, indent=4, ensure_ascii=False))

                            # 将结果存回历史
                            self.history.append({
                                "role": "tool",
                                "name": name,
                                "content": str(result)
                            })

                        except Exception as e:
                            logger.error(f"工具执行错误: {e}")
                            traceback.print_exc()
                            self.history.append({
                                "role": "tool",
                                "name": name,
                                "content": f"Error: {str(e)}"
                            })

                # 避免过热空转
                if not tool_calls:
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                await asyncio.sleep(5)  # 出错冷却

    async def _process_incoming_event(self, event: OneBotEvent):
        """
        处理外部事件：
        1. 让边缘系统“感受”刺激 (Process Stimulus)
        2. 将事件写入历史记录
        """
        # 1. 边缘系统介入 (只处理文本消息)
        # 只有真实人类的消息才算作"刺激"，内部信号不算
        if event.type == EventType.MESSAGE and isinstance(event.message, str):
            # 异步调用，不阻塞主流程太多
            asyncio.create_task(self.limbic.process_stimulus(event.message))

        # 自动捕获用户
        # 尝试从 source 中获取用户信息
        adapters = self.tool_manager.dependency_map.get("adapters")
        adapter_names = []
        for i, adapter in enumerate(adapters, 1):
            adapter_names.append(getattr(adapter, "platform_name", "Unknown"))


        if event.source.user_id and event.source.platform in adapter_names:
            platform = event.source.platform
            raw_id = event.source.user_id

            # 1. 计算 UID
            uid = self.user_manager.resolve_uid(platform, raw_id)

            # 2. 查询用户 (只读)
            user_profile = await self.user_manager.get_user(uid)

            # 3. 注入上下文 (Scratchpad)
            interactor_info = {
                "uid": uid,
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

        # 2. 序列化事件
        event_data = event.model_dump(exclude_none=True)
        # 清理冗余字段
        for field in ["id", "time", "raw_data"]:
            if field in event_data: del event_data[field]
        if "message" in event_data and "alt_message" in event_data:
            del event_data["message"]

        # [特殊处理] 如果是内部驱动信号，添加高亮提示
        prefix = ""
        if event.detail_type == DetailType.INTERNAL_DRIVE:
            prefix = "[INTERNAL SIGNAL - URGENT]\n"
            # 强制唤醒
            self.is_sleeping = False

        context_msg = f"{prefix}[Event Received]\n{json.dumps(event_data, ensure_ascii=False)}"

        # 3. 更新 Scratchpad 上下文 (如果是消息事件)
        if event.source.platform != "internal":
            source = event.source
            ctx_type = "group" if source.group_id else "private"
            ctx_id = source.group_id if source.group_id else source.user_id

            self.scratchpad["last_context"] = {
                "platform": source.platform,
                "type": ctx_type,
                "id": ctx_id
            }

        self.history.append({"role": "user", "content": context_msg})
        logger.info(f"Event Ingested: {event.type}.{event.detail_type}")

    async def _call_llm(self) -> Dict[str, Any]:
        """封装 API 调用"""
        schemas = self.tool_manager.get_tool_schemas()

        # 定义强制思维 Schema (JSON Schema)
        thought_structure = {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "你的思考过程，分析当前状态、系统反馈和下一步计划。必须包含对失败原因的分析（如果有）。"
                },
                "scratchpad": {
                    "type": "object",
                    "description": "一个用于追踪任务状态、变量和进度的结构化工作区。",
                    "properties": {
                        "current_goal": {"type": "string", "description": "当前正在执行的具体目标"},
                        "subtasks": {
                            "type": "array",
                            "description": "为实现目标而设定的子任务列表。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "description": {"type": "string"},
                                    "status": {"type": "string", "enum": ["pending", "working", "done", "failed"]}
                                },
                                "required": ["id", "desc", "status"]
                            }
                        },
                        "variables": {"type": "object", "description": "存储临时数据或ID"},
                        "progress_summary": {"type": "string", "description": "简要总结已完成的工作"}
                    },
                    "required": ["current_goal", "subtasks", "progress_summary"]
                },
                "tool_calls": {
                    "type": "array",
                    "description": "需要执行的工具列表。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"}
                        },
                        "required": ["name", "arguments"]
                    }
                }
            },
            "required": ["thought", "scratchpad", "tool_calls"],
            "additionalProperties": False
        }

        # 调用 API，同时传入 tools 和 schema
        response = await self.api_client.create_chat_completion(
            messages=self.history,
            tools=schemas if schemas else None,
            schema=thought_structure
        )

        return response["choices"][0]["message"]
