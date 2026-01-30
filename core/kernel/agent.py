# core/kernel/agent.py
import asyncio
import json
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, Awaitable, Callable

import aiofiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.gui.monitor_registry import monitor_registry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, Action, DetailType, EventType
from core.io.middleware import MiddlewareManager
from core.kernel.attention import AttentionFilter, ReactionType
from core.kernel.prompt import PromptManager
from core.limbic.manager import LimbicManager
from core.memory.hippocampus import Hippocampus
from core.memory.infinite_context import InfiniteContextManager
from core.memory.vector_store import VectorStore
from core.social.manager import UserManager
from core.tool_manager.aggregator import ToolManager
from core.utilities import calculate_tokens

logger = logging.getLogger(__name__)

AGENT_STATE_KEY = "AGENT_CORE_SNAPSHOT"

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
            "progress_summary": "",
            "current_interactor": None,
            "last_context": None
        }
        self.last_response_content = ""  # 用于死锁检测
        self.last_observation_text = None
        self.consecutive_idle_count = 0  # 空转计数器

        # 初始化注意力门控系统
        self.attention = AttentionFilter(
            config=config,
            prompt=self.prompt_manager,
            api_client=self.api_client,
            database=database
        )

        # --- 强制休眠标记 ---
        self.force_sleep = False

        # --- 初始化工具管理器 ---
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

        # 中间件系统
        self.middleware = MiddlewareManager()
        self._setup_middlewares()

        # 消息缓冲区
        self.incoming_events: asyncio.Queue = asyncio.Queue()
        self.event_bus.subscribe_event(self._enqueue_event)

        # 临时状态：当前事件的反应决定
        self._current_reaction: Optional[ReactionType] = None
        self._should_think_after_event: bool = False

    def _setup_middlewares(self):
        """注册中间件链"""
        # 1. 上下文富化
        self.middleware.register(self._mw_enrich_context)
        # 2. 消息转录
        self.middleware.register(self._mw_transcribe_and_log)
        # 3. 注意力门控
        self.middleware.register(self._mw_attention_filter)

    # --- 中间件实现 ---

    async def _mw_enrich_context(self, event: OneBotEvent, next_call: Callable[[], Awaitable[None]]):
        """
        [中间件] 社交上下文富化
        识别用户身份，读取档案，并注入到 Scratchpad 中。
        """
        # 自动捕获用户
        adapters = self.tool_manager.dependency_map.get("adapters", [])
        adapter_names = [getattr(a, "platform_name", "Unknown") for a in adapters]

        if event.source.user_id and event.source.platform in adapter_names:
            platform = event.source.platform
            raw_id = event.source.user_id
            puid = f"{platform}:{raw_id}"

            # 查询用户
            user_profile = await self.user_manager.get_user(puid)

            interactor_info = {
                "puid": puid,
                "platform": platform,
                "user_id": raw_id,
            }

            if user_profile:
                # [熟人] - 注入完整社交维度
                interactor_info.update({
                    "status": "KNOWN",
                    "nickname": user_profile.nickname,
                    "relationship_tags": user_profile.relationship_tags,
                    "favorability": user_profile.favorability,
                    "trust": user_profile.trust,
                    "intimacy": user_profile.intimacy,  # 新增
                    "impression": user_profile.impression
                })
            else:
                # [陌生人]
                interactor_info.update({
                    "status": "STRANGER",
                    "note": "User not in database. Use tool `social_record_user` to remember them."
                })

            # 更新 Scratchpad
            self.scratchpad["current_interactor"] = interactor_info
            monitor_registry.register_text_source(
                "认知", "人员注入",
                lambda: json.dumps(interactor_info, indent=2, ensure_ascii=False)
            )

            # 更新上下文位置
            ctx_type = "group" if event.source.group_id else "private"
            ctx_id = event.source.group_id if event.source.group_id else raw_id
            self.scratchpad["last_context"] = {
                "platform": platform,
                "type": ctx_type,
                "id": ctx_id
            }

        await next_call()

    async def _mw_transcribe_and_log(self, event: OneBotEvent, next_call: Callable[[], Awaitable[None]]):
        """
        [中间件] 事件转译与历史记录
        """
        if event.type == EventType.META:
            await next_call()
            return

        # 1. 边缘系统“感受”刺激
        if event.type == EventType.MESSAGE and isinstance(event.message, str):
            asyncio.create_task(self.limbic.process_stimulus(event.message))
            self.last_observation_text = event.message

        # 2. 决定消息内容
        if event.type == EventType.MESSAGE:
            # 序列化清理
            event_data = event.model_dump(exclude_none=True)
            for field in ["id", "time", "raw_data", "message", "alt_message"]:
                if field in event_data: del event_data[field]

            content_msg = f"接收到用户消息：{json.dumps(event_data, ensure_ascii=False)} 内容：{event.alt_message}"
            is_ephemeral = False
        else:
            # 非对话事件，进行自然语言转译
            content_msg = self._transcribe_event(event)
            is_ephemeral = True

        # 3. 写入历史
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

        # 如果是内部驱动，唤醒系统
        if event.detail_type == DetailType.INTERNAL_DRIVE:
            self.is_sleeping = False
            self.force_sleep = False

        await next_call()

    async def _mw_attention_filter(self, event: OneBotEvent, next_call: Callable[[], Awaitable[None]]):
        """
        [中间件] 注意力门控
        """
        # 1. 评估
        reaction = await self.attention.evaluate(event)
        self._current_reaction = reaction

        # 2. 决策
        if reaction in [ReactionType.OBSERVE, ReactionType.IGNORE]:
            logger.info(f"🤐 [Observer] 决定保持沉默: {event.id}")
            # 如果当前没有任务且被忽略，进入待机
            current_goal = self.scratchpad.get("current_goal")
            if not current_goal:
                self.force_sleep = True

            # 依然调用 next，因为可能需要其他处理，但标记不思考
            self._should_think_after_event = False
        else:
            logger.info(f"🗣️ [Responder] 决定介入 ({reaction.value}): {event.id}")
            self._should_think_after_event = True

            # 如果收到新事件，重置空转计数器
            self.consecutive_idle_count = 0
            if self.wakeup_job_id:
                try:
                    self.scheduler.remove_job(self.wakeup_job_id)
                    self.wakeup_job_id = None
                except:
                    pass

        await next_call()

    async def _enqueue_event(self, event):
        """回调：将总线事件放入缓冲区"""
        await self.incoming_events.put(event)

    async def _final_event_handler(self, event: OneBotEvent):
        """中间件链的终点"""
        pass

    def _prune_context(self):
        """
        检查并修剪对话历史，防止超过 Token 限制。
        此函数作为 LLM 调用前的一个预处理，基于对中英文和图片Token的估算。
        智能识别 Tool Call 对，确保成对删除。
        """
        # 尝试从配置获取上下文限制
        TOKEN_LIMIT_APPROX = self.config.get("llm.model_context", 16384)
        SAFE_LIMIT = TOKEN_LIMIT_APPROX - 500

        current_estimated_tokens = 0

        # 1. 计算当前总 Token
        for msg in self.history:
            current_estimated_tokens += calculate_tokens(msg.get("content"))

        # 始终保留 System Prompt (index 0) 和最近的 5 条消息
        while len(self.history) > 6 and current_estimated_tokens > SAFE_LIMIT:
            # 从 index 1 开始检查 (跳过 system)
            candidate_idx = 1
            msg_to_remove = self.history[candidate_idx]

            # 判断是否是 Tool Call 的发起者
            # 兼容 OpenAI 格式 (tool_calls 字段) 和旧格式
            is_tool_call_msg = (msg_to_remove.get("role") == "assistant" and
                                (msg_to_remove.get("tool_calls") or msg_to_remove.get("function_call")))

            count_to_remove = 1

            if is_tool_call_msg:
                # 如果删除了 tool_calls，必须删除后面紧跟的所有 role='tool' 消息
                # 扫描后续消息
                scan_idx = candidate_idx + 1
                while scan_idx < len(self.history):
                    next_msg = self.history[scan_idx]
                    if next_msg.get("role") == "tool":
                        count_to_remove += 1
                        scan_idx += 1
                    else:
                        break

                logger.info(f"✂️ [Context] 检测到工具调用链，将批量移除 {count_to_remove} 条消息。")

            # 执行移除
            for _ in range(count_to_remove):
                if len(self.history) > 1:  # 再次检查防止越界
                    removed = self.history.pop(candidate_idx)
                    current_estimated_tokens -= calculate_tokens(removed.get("content"))

            logger.info(f"✂️ [Context] 修剪后估算: {int(current_estimated_tokens)}")

        return current_estimated_tokens

    # 核心状态持久化方法
    async def _save_snapshot(self):
        """
        固化 Agent 核心状态到数据库
        应在思考步骤结束后或关键状态变更时调用
        """
        try:
            snapshot = {
                "scratchpad": self.scratchpad,
                "consecutive_idle_count": self.consecutive_idle_count,
                "is_sleeping": self.is_sleeping,
                "force_sleep": self.force_sleep,
                "timestamp": time.time()
            }
            json_str = json.dumps(snapshot, ensure_ascii=False)

            async with self.database.get_connection() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO neuro_states (user_id, data_json, last_update) VALUES (?, ?, ?)",
                    (AGENT_STATE_KEY, json_str, time.time())
                )
                await conn.commit()
            logger.debug("核心状态快照已保存")
        except Exception as e:
            logger.error(f"状态快照保存失败: {e}")

    # 核心状态恢复方法
    async def _load_snapshot(self):
        """
        从数据库恢复 Agent 状态
        """
        try:
            async with self.database.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT data_json FROM neuro_states WHERE user_id=?",
                    (AGENT_STATE_KEY,)
                )
                row = await cursor.fetchone()
                if row:
                    snapshot = json.loads(row[0])
                    # 恢复状态
                    self.scratchpad.update(snapshot.get("scratchpad", {}))
                    self.consecutive_idle_count = snapshot.get("consecutive_idle_count", 0)
                    self.is_sleeping = snapshot.get("is_sleeping", False)
                    self.force_sleep = snapshot.get("force_sleep", False)

                    logger.info(f"🔄 成功恢复 Agent 核心状态 (上次保存: {time.ctime(snapshot.get('timestamp', 0))})")
                    logger.info(f"   当前目标: {self.scratchpad.get('current_goal')}")
        except Exception as e:
            logger.error(f"状态恢复失败: {e}")

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
        # 获取当前兴趣
        initial_interest = await self.attention.get_current_interest_text()

        # 获取初始生理状态
        initial_state = await self.limbic.get_state()

        # 生成完整 Prompt
        system_prompt = self.prompt_manager.get_system_prompt(
            neuro_state=initial_state,
            interest_context=initial_interest
        )
        self.history.append({"role": "system", "content": system_prompt})

        # 2. 尝试恢复核心状态
        await self._load_snapshot()

        # 3. 启动后台任务
        asyncio.create_task(self.hippocampus.start())
        asyncio.create_task(self.limbic.start())

        # 4. 注入启动信号
        self.history.append({
            "role": "user",
            "content": f"系统启动完成。"
        })

        while True:
            try:
                # --- A. 感知阶段 (Perception) ---
                event = None

                # 状态重置
                self._should_think_after_event = False
                self._current_reaction = None

                # 强制休眠逻辑
                # 如果系统判定当前应当“安静等待”且不是手动休眠模式
                if self.force_sleep and not self.is_sleeping:
                    # 阻塞式等待：直到有新事件才唤醒，彻底避免空转
                    event = await self.incoming_events.get()
                    self.force_sleep = False  # 收到事件，解除强制休眠

                # 正常非阻塞/超时获取逻辑
                elif not self.incoming_events.empty():
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

                # 通过中间件管道处理事件
                self._should_think_after_event = False
                if event:
                    await self.middleware.process_event(event, self._final_event_handler)

                # 决策：是否进入思考循环
                # 1. 如果中间件决定忽略 (OBSERVE/IGNORE) 且没有待处理事件 -> 跳过
                # 2. 如果 event 为空，但之前可能有任务在进行 -> 继续
                if event and not self._should_think_after_event:
                    continue

                await self._save_snapshot()

                # --- B. 思考与决策阶段 (Thought) ---

                # 上下文压缩与修剪
                await self.context_manager.compress_if_needed(self.history)
                self._prune_context()

                # 主动记忆检索 (RAG)
                retrieved_memories = await self._active_retrieval()

                # 动态生成 System Prompt
                # 1. 获取当前神经状态
                current_neuro_state = await self.limbic.get_state()
                current_interactor = self.scratchpad.get("current_interactor")
                current_interest = await self.attention.get_current_interest_text()

                # 2. 生成带有状态描述的 Prompt
                system_prompt_base = self.prompt_manager.get_system_prompt(
                    neuro_state=current_neuro_state,
                    memory_context=retrieved_memories,
                    social_context=current_interactor,
                    interest_context=initial_interest
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
                if self.history[0]["role"] == "system":
                    self.history[0]["content"] = final_system_prompt
                else:
                    self.history.insert(0, {"role": "system", "content": final_system_prompt})

                try:
                    async with aiofiles.open("data/messages_in_memory.txt", "w", encoding="utf-8") as f:
                        await f.write(json.dumps(self.history, ensure_ascii=False, indent=4))

                    async with aiofiles.open("data/prompt_in_memory.txt", "w", encoding="utf-8") as f:
                        await f.write(final_system_prompt)
                except Exception as e:
                    logger.warning(f"Failed to write debug logs: {e}")

                # 调用 LLM
                response_msg = await self._call_llm()

                # 解析 LLM 响应
                content_str = response_msg.get("content")

                if isinstance(content_str, str):
                    content_str = (content_str
                                   .replace("```json", "")
                                   .replace("```", "")
                                   .strip())
                else:
                    content_str = "{}"

                # 处理原生 Tool Calls
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
                    # Schema 模式回填
                    try:
                        # 尝试格式化 JSON 以美观存储
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

                        # 提取双层信息
                        monologue = parsed_data.get("inner_monologue", {})

                        # 构造思考日志
                        emotion = monologue.get("emotion_check", "Neutral")
                        plan = monologue.get("planning", "No plan")

                        # 更新 Scratchpad
                        new_scratchpad = parsed_data.get("scratchpad", None)
                        if new_scratchpad and isinstance(new_scratchpad, dict):
                            self.scratchpad = new_scratchpad
                            self.tool_manager.agent_state = self.scratchpad

                        # 广播思考过程 (EventBus)
                        self.event_bus.publish_action(Action(
                            action="broadcast_log",
                            params={"content": f"🧠 心流: [{emotion}] {plan}"}
                        ))

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
                    max_allowed_idle = 2
                    # 没有执行任何工具
                    self.consecutive_idle_count += 1
                    logger.warning(f"⚠️ 空转检测: {self.consecutive_idle_count}/{max_allowed_idle}")

                    if self.consecutive_idle_count >= max_allowed_idle:
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
            # 1. 内心独白层
            "inner_monologue": {
                "type": "object",
                "description": "在回复前的思考过程。",
                "properties": {
                    "emotion_check": {
                        "type": "string",
                        "description": "自检当前的生理状态和情绪基调。"
                    },
                    "intention": {
                        "type": "string",
                        "description": "明确当前的行动意图。例如：'安抚用户情绪'、'执行搜索任务' 或 '结束对话'。"
                    },
                    "planning": {
                        "type": "string",
                        "description": "具体的思维链推理过程。"
                    }
                },
                "required": ["emotion_check", "intention", "planning"],
                "additionalProperties": False
            },
            # 2. 状态层
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

        required_fields = ["inner_monologue", "scratchpad"]

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
            schema=thought_structure,
            tool_choice="none" if use_schema_tools else "auto"
        )

        return response["choices"][0]["message"]
