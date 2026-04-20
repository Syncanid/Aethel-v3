# core/kernel/agent.py
import asyncio
import json
import logging
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Awaitable, Callable

import aiofiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.gui.monitor_registry import monitor_registry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, Action, DetailType, EventType
from core.io.fragmentation import OutputFragmenter
from core.io.middleware import MiddlewareManager
from core.kernel.attention import AttentionFilter, ReactionType
from core.kernel.prompt import PromptManager
from core.kernel.task_registry import global_task_registry
from core.limbic.manager import LimbicManager
from core.memory.hippocampus import Hippocampus
from core.memory.infinite_context import InfiniteContextManager
from core.memory.vector_store import VectorStore
from core.social.manager import UserManager
from core.tool_manager.aggregator import ToolManager
from core.utilities import calculate_tokens, encode_image_to_data_uri

logger = logging.getLogger(__name__)

AGENT_STATE_KEY = "AGENT_CORE_SNAPSHOT"


class AutonomousAgent:
    def __init__(self, config: Config, event_bus: EventBus, database: Database):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.api_client = GenericAPIClient(config)
        self.prompt_manager = PromptManager(config)

        # --- Session Virtualization 状态池 ---
        self.working_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.session_last_active: Dict[str, float] = {}
        self.session_willingness: Dict[str, float] = {}
        self.session_scratchpads: Dict[str, Dict[str, Any]] = {}
        self.session_last_responses: Dict[str, str] = {}
        self.session_last_observations: Dict[str, str] = {}
        self.session_next_use_tools: Dict[str, bool] = {}

        self.global_blackboard: Dict[str, Dict[str, Any]] = {}
        self.active_session_id: str = "system_default"

        # --- 强制休眠标记 ---
        self.force_sleep = False

        # 初始化注意力门控系统
        self.attention = AttentionFilter(
            config=config,
            prompt=self.prompt_manager,
            api_client=self.api_client,
            database=database
        )

        # --- 初始化工具管理器 ---
        self.tool_manager = ToolManager(
            tools_dir="tools/System1",
            config=config,
            event_bus=event_bus,
            api_client=self.api_client,
            database=database,
            agent_state={}
        )

        # 初始化边缘系统
        self.limbic = LimbicManager(config, database, event_bus, self.api_client)

        # 初始化记忆组件
        self.hippocampus = Hippocampus(
            config=config,
            limbic=self.limbic,
            api_client=self.api_client,
            database=database,
            agent_history=self.working_memory
        )
        self.vector_store = VectorStore(database, self.api_client)

        # 初始化用户管理器
        self.user_manager = UserManager(database)

        # 初始化任务调度器
        self.scheduler = AsyncIOScheduler()

        # 初始化无限上下文管理器
        self.context_manager = InfiniteContextManager(config, self.api_client)

        # 提取角色卡配置并实例化输出分段器
        role_config = getattr(self.prompt_manager, "role_data", {})
        self.fragmenter = OutputFragmenter(role_config)

        # 记录唤醒任务的 ID
        self.wakeup_job_id = None

        # 睡眠状态标记
        self.is_sleeping = False

        # 注入 Scheduler 和 Agent 自身
        self.tool_manager.add_dependency("agent", self)
        self.tool_manager.add_dependency("scheduler", self.scheduler)
        self.tool_manager.add_dependency("limbic", self.limbic)
        self.tool_manager.add_dependency("user_manager", self.user_manager)
        self.tool_manager.add_dependency("fragmenter", self.fragmenter)

        # 中间件系统
        self.middleware = MiddlewareManager()
        self._setup_middlewares()

        # 消息缓冲区
        self.incoming_events: asyncio.Queue = asyncio.Queue()
        self.event_bus.subscribe_event(self._enqueue_event)

        # 临时状态：当前事件的反应决定
        self._current_reaction: Optional[ReactionType] = None
        self._should_think_after_event: bool = False

        self.current_tokens = 0

    # ==========================================
    # 向下兼容代理
    # 防止未修改的老工具抛出 AttributeError
    # ==========================================
    @property
    def scratchpad(self) -> Dict[str, Any]:
        """动态路由到当前上下文的暂存板"""
        return self.session_scratchpads.get(self.active_session_id, {})

    @scratchpad.setter
    def scratchpad(self, value: Dict[str, Any]):
        self.session_scratchpads[self.active_session_id] = value

    @property
    def last_response_content(self) -> str:
        return self.session_last_responses.get(self.active_session_id, "")

    @last_response_content.setter
    def last_response_content(self, value: str):
        self.session_last_responses[self.active_session_id] = value

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
        """
        # 自动捕获用户
        adapters = self.tool_manager.dependency_map.get("adapters", [])
        adapter_names = [getattr(a, "platform_name", "Unknown") for a in adapters]

        if event.source.user_id and event.source.platform in adapter_names:
            platform = event.source.platform
            raw_id = event.source.user_id

            # 更新上下文位置
            ctx_type = "group" if event.source.group_id else "private"
            ctx_id = event.source.group_id if event.source.group_id else raw_id

            # 隔离写入：附着在事件体上，而非写脏全局属性
            event.extra["parsed_context"] = {
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

        if event.type == EventType.TASK and event.detail_type in [DetailType.TASK_PROGRESS, "task_progress"]:
            requires_input = event.extra.get("requires_user_input", False)
            if not requires_input:
                payload = event.extra.get("task_payload", {})
                task_id = payload.get("task_id", "unknown")
                msg = payload.get("progress_msg", "")

                logger.info(f"🔇 [总线拦截] 截获 S2 进度汇报 ({task_id}): {msg}")

                # 仅推送到 UI 监控面板（Dashboard）
                monitor_registry.register_text_source(
                    "系统", "S2 后台进度", lambda m=msg, tid=task_id: f"Task [{tid}]: {m}"
                )
                return

        if event.type == EventType.TASK and event.detail_type in [
            DetailType.TASK_DISPATCH, "task_dispatch",
            DetailType.TASK_UPDATE, "task_update",
            DetailType.TASK_CANCEL, "task_cancel"
        ]:
            await next_call()
            return

        # 计算 Session ID
        ctx_type = "group" if event.source.group_id else "private"
        platform_name = getattr(event.source, "platform", "unknown")
        ctx_id = event.source.group_id if event.source.group_id else event.source.user_id
        session_id = f"{ctx_type}_{platform_name}:{ctx_id}"
        event_puid = f"{platform_name}:{event.source.user_id}" if event.source.user_id else None
        event.extra["event_puid"] = event_puid

        if event.detail_type == DetailType.CROSS_SESSION_DIRECTIVE:
            target_sid = event.extra.get("target_session_id")
            reason = event.extra.get("directive_reason", "未知任务")
            context = event.extra.get("carried_context", "")

            logger.warning(f"🛸 [维度跳跃/唤醒] 焦点切换至: {target_sid}")
            self.active_session_id = target_sid

            # 如果目标频道尚未初始化，或需要强制注入任务上下文
            if target_sid not in self.working_memory:
                self.working_memory[target_sid] = [{"role": "system", "content": "INITIALIZING..."}]

            if event.extra.get("status") == "wake_up":
                arrival_msg = {
                    "role": "user",
                    "content": (
                        f"【系统调度通知】{context}\n"
                        f"请基于当前的上下文评估环境。如果没有明确的交流必要或新的互动，请保持安静（建议调用 wait/wait_forever 再次挂起，或将 action 设为 ignore 结束回合），切勿生硬地为了说话而说话。"
                    ),
                    "metadata": {"type": "system_directive", "session_id": target_sid}
                }
            else:
                arrival_msg = {
                    "role": "user",
                    "content": (
                        f"【跨域意识投射】你刚刚从其他维度降临至此。\n"
                        f"本次降临的任务目的：{reason}\n"
                        f"随意识携带的关键情报：\n{context}\n"
                        f"请立刻根据此上下文展开后续行动。"
                    ),
                    "metadata": {"type": "system_directive", "session_id": target_sid}
                }

            self.working_memory[target_sid].append(arrival_msg)
            await next_call()
            return

        self.active_session_id = session_id

        # 1. 边缘系统“感受”刺激
        if event.type == EventType.MESSAGE and isinstance(event.message, str):
            asyncio.create_task(self.limbic.process_stimulus(event.message))
            self.session_last_observations[session_id] = event.message

        # 2. 决定消息内容
        if event.type == EventType.MESSAGE:
            event_data = event.model_dump(exclude_none=True)
            for field in ["id", "time", "raw_data", "message", "alt_message"]:
                if field in event_data: del event_data[field]

            text_content = f"接收到用户消息：{json.dumps(event_data, ensure_ascii=False)} 内容：{event.alt_message}"
            images = event.extra.get("images", [])

            if images:
                # 多模态降维剥离：主体转为纯文本
                content_payload = text_content

                for img_source in images:
                    b64_data_uri = await encode_image_to_data_uri(img_source)

                    if b64_data_uri:
                        img_id = f"img_{uuid.uuid4().hex[:8]}"
                        # 将图片 base64 写入 SQLite，同时清理 50 张以外的旧图片以控制 DB 体积
                        async with self.database.get_connection() as conn:
                            await conn.execute(
                                "INSERT OR REPLACE INTO multimodal_cache (image_id, b64_data_uri, timestamp) VALUES (?, ?, ?)",
                                (img_id, b64_data_uri, time.time())
                            )
                            # 数据库层面的 LRU 截断
                            await conn.execute(
                                "DELETE FROM multimodal_cache WHERE image_id NOT IN (SELECT image_id FROM multimodal_cache ORDER BY timestamp DESC LIMIT 50)"
                            )
                            await conn.commit()

                        # 视神经解析
                        try:
                            logger.info(f"👁️ 正在对图像 [{img_id}] 进行解析...")
                            initial_prompt = "你是一个前置视神经模块。描述这张图的内容特征。"

                            resp = await self.api_client.create_chat_completion(
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": initial_prompt},
                                        {"type": "image_url", "image_url": {"url": b64_data_uri}}
                                    ]
                                }],
                                model=self.api_client.small_model
                            )
                            initial_desc = resp.get("content", "图片特征提取失败。")
                        except Exception as e:
                            logger.error(f"视神经初析崩溃: {e}")
                            initial_desc = f"视觉组件离线: {e}"

                        # 认知映射：用文字锚点替代真实图片
                        content_payload += f"\n\n[图片附件 ID: {img_id}] 系统初析特征: {initial_desc}\n(系统提示: 上述为低精度摘要。如果需要更详细的信息，你必须主动调用 `reparse_image` 工具提取。)"
                    else:
                        content_payload += "\n[系统警告：此位置的一张图片因获取或转码失败已丢失。]"
            else:
                # 兼容旧版纯文本
                content_payload = text_content

            is_ephemeral = False
        else:
            content_payload = self._transcribe_event(event)
            is_ephemeral = True

        # 3. 写入历史
        history_item = {
            "role": "user",
            "content": content_payload,
            "metadata": {
                "type": event.type,
                "ephemeral": is_ephemeral,
                "raw_event_id": event.id,
                "session_id": session_id,
                "puid": event_puid
            }
        }
        logger.info(f"Event Ingested: {event.type}.{event.detail_type} -> Routed to [{session_id}]")

        # 3. 会话隔离与初始化防空指针
        if session_id not in self.working_memory:
            self.working_memory[session_id] = [
                {"role": "system", "content": "INITIALIZING..."},
                {"role": "user", "content": "【系统心跳】会话通道已建立。"}
            ]
            logger.info(f"🆕 开启全新独立认知会话: {session_id}")

        # 4. 执行物理隔离追加
        self.working_memory[session_id].append(history_item)
        self.session_last_active[session_id] = time.time()

        # 强制将瞬时听觉残留推入海马体前端队列。
        try:
            await self.hippocampus.push_to_sensory_memory(session_id, history_item)
        except AttributeError:
            logger.warning("海马体推送管道未就绪，记忆降级为游离态。")

        # 唤醒系统
        if event.detail_type in [DetailType.INTERNAL_DRIVE, DetailType.TASK_COMPLETE] \
                or event.type == EventType.REQUEST:
            self.is_sleeping = False
            self.force_sleep = False

        await next_call()

    async def _mw_attention_filter(self, event: OneBotEvent, next_call: Callable[[], Awaitable[None]]):
        """
        [中间件] 注意力门控
        """

        if event.type == EventType.REQUEST:
            logger.info(f"🚨 [Attention] 拦截到社交请求事件，强制劫持注意力并要求 S1 决策: {event.id}")
            self._should_think_after_event = True
            if self.wakeup_job_id:
                try:
                    self.scheduler.remove_job(self.wakeup_job_id)
                    self.wakeup_job_id = None
                except:
                    pass

        if event.detail_type == DetailType.CROSS_SESSION_DIRECTIVE:
            logger.info(f"🚨 [Attention] 拦截到系统级跨域跳跃指令，无条件放行并强制唤醒！")
            self._should_think_after_event = True
            if self.wakeup_job_id:
                try:
                    self.scheduler.remove_job(self.wakeup_job_id)
                    self.wakeup_job_id = None
                except:
                    pass

            await next_call()
            return

        session_id = getattr(self, "active_session_id", "system_default")
        active_history = self.working_memory.get(session_id, [])

        current_will = self.session_willingness.get(session_id, 0.5)
        recent_history = active_history[:-1][-5:] if len(active_history) > 1 else []

        # 解包门控的三重参数
        reaction_result = await self.attention.evaluate(event,
                                                        recent_history=recent_history,
                                                        willingness=current_will
                                                        )

        if isinstance(reaction_result, tuple) and len(reaction_result) == 3:
            reaction, should_do, will_shift = reaction_result
        else:
            reaction = reaction_result
            should_do = "暂无建议"
            will_shift = 0.0

        self._current_reaction = reaction
        self._current_should_do = should_do

        # 将门控输出的意愿偏离量直接赋予会话状态
        new_will = max(0.0, min(1.0, current_will + will_shift))
        self.session_willingness[session_id] = new_will
        if will_shift != 0.0:
            logger.info(f"⚖️ [Attention Gate] 事件导致意愿偏离 {will_shift:+.2f} -> 当前意愿: {new_will:.2f}")

        # 从 Session 隔离池中安全读取当前意图
        current_goal = self.session_scratchpads.get(session_id, {}).get("current_goal")

        if reaction == ReactionType.IGNORE:
            logger.info(f"🗑️ [Observer] 决定无视: {event.id}")
            if not current_goal and not any(self.session_next_use_tools.values()):
                self.force_sleep = True
            self._should_think_after_event = False

        elif reaction == ReactionType.OBSERVE:
            logger.info(f"🤐 [Observer] 决定保持沉默 (仅观察): {event.id}")
            if not current_goal and not any(self.session_next_use_tools.values()):
                self.force_sleep = True
            self._should_think_after_event = False

        elif reaction.value == ReactionType.SILENT_OBSERVE:
            logger.info(f"😒 [Observer] 决定不理会 (积极静默，将产生内心独白): {event.id}")
            self._should_think_after_event = True
            if self.wakeup_job_id:
                try:
                    self.scheduler.remove_job(self.wakeup_job_id)
                    self.wakeup_job_id = None
                except:
                    pass
        else:
            logger.info(f"🗣️ [Responder] 决定介入 ({reaction.value}): {event.id}")
            self._should_think_after_event = True
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

    def _prune_context(self, session_id: str) -> int:
        """
        [应急防爆] 强制上下文物理修剪 (S1)
        作为 InfiniteContext 之后的最后一道物理防线。
        哪怕切掉的是早期的脱水记忆，也必须保证系统能顺利调用 LLM，防止死锁崩溃。
        """
        if session_id not in self.working_memory:
            return 0

        session_history = self.working_memory[session_id]

        # 尝试从配置获取上下文限制
        TOKEN_LIMIT_APPROX = self.config.get("llm.model_context", 16384)
        SAFE_LIMIT = TOKEN_LIMIT_APPROX - 1000

        # 重新精准盘点当前 Session 的 Token
        current_estimated_tokens = sum(calculate_tokens(msg.get("content", "")) for msg in session_history)

        # 如果 Token 安全，什么都不做，直接返回
        if current_estimated_tokens <= SAFE_LIMIT:
            return current_estimated_tokens

        logger.warning(
            f"⚠️ [S1 应急防御] 会话 {session_id} Token估算 ({int(current_estimated_tokens)}) 突破红线！启动强制切割。")

        # 始终保留 System Prompt (index 0) 和最近的一小部分消息以维持最小对话惯性
        while len(session_history) > 6 and current_estimated_tokens > SAFE_LIMIT:
            candidate_idx = 1
            msg_to_remove = session_history[candidate_idx]

            # 兼容 API 规范：如果删除了发起 tool_calls 的 assistant 消息，
            # 必须连同它后面跟随的所有 role: "tool" 结果一起删掉，否则 OpenAI/主流模型 接口会直接报错。
            is_tool_call_msg = (msg_to_remove.get("role") == "assistant" and
                                (msg_to_remove.get("tool_calls") or msg_to_remove.get("function_call")))

            count_to_remove = 1

            if is_tool_call_msg:
                scan_idx = candidate_idx + 1
                while scan_idx < len(session_history):
                    next_msg = session_history[scan_idx]
                    if next_msg.get("role") == "tool":
                        count_to_remove += 1
                        scan_idx += 1
                    else:
                        break

            # 执行物理移除 (从头开始切)
            for _ in range(count_to_remove):
                if len(session_history) > 1:
                    removed = session_history.pop(candidate_idx)
                    current_estimated_tokens -= calculate_tokens(removed.get("content", ""))

        logger.warning(f"✂️ [S1 应急防御] 强制切割完成。当前剩余 Token 估算: {int(current_estimated_tokens)}")
        return current_estimated_tokens

    async def save_state(self):
        """统一持久化 S1 的完整运行状态"""
        try:
            state = {
                "active_session_id": getattr(self, "active_session_id", "system_default"),
                "working_memory": self.working_memory,
                "session_last_active": self.session_last_active,
                "session_willingness": self.session_willingness,
                "session_scratchpads": self.session_scratchpads,
                "global_blackboard": self.global_blackboard,
                "is_sleeping": self.is_sleeping,
                "force_sleep": self.force_sleep,
                "session_last_responses": self.session_last_responses,
                "session_last_observations": self.session_last_observations,
                "session_next_use_tools": self.session_next_use_tools,
                "timestamp": time.time()
            }
            json_str = json.dumps(state, ensure_ascii=False)

            async with self.database.get_connection() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO neuro_states (user_id, data_json, last_update) VALUES (?, ?, ?)",
                    ("UNIFIED_S1_STATE", json_str, time.time())
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"S1 状态全量保存失败: {e}", exc_info=True)

    async def load_state(self) -> bool:
        """从统一存储中恢复 S1 的完整现场"""
        try:
            async with self.database.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT data_json FROM neuro_states WHERE user_id=?",
                    ("UNIFIED_S1_STATE",)
                )
                row = await cursor.fetchone()
                if row:
                    state = json.loads(row[0])

                    self.active_session_id = state.get("active_session_id", "system_default")
                    self.working_memory = state.get("working_memory", {})
                    self.session_last_active = state.get("session_last_active", {})
                    self.session_willingness = state.get("session_willingness", {})
                    self.session_scratchpads = state.get("session_scratchpads", {})
                    self.session_last_responses = state.get("session_last_responses", {})
                    self.session_last_observations = state.get("session_last_observations", {})
                    self.session_next_use_tools = state.get("session_next_use_tools", {})

                    self.is_sleeping = state.get("is_sleeping", False)
                    self.force_sleep = state.get("force_sleep", False)

                    loaded_blackboard = state.get("global_blackboard", {})
                    current_time = time.time()
                    self.global_blackboard = {}

                    for sid, topic_data in loaded_blackboard.items():
                        # 只恢复 1 小时以内的热点，超过 1 小时的直接在内存中丢弃
                        if current_time - topic_data.get("timestamp", 0) <= 3600:
                            self.global_blackboard[sid] = topic_data

                    logger.info(f"💾 状态恢复完成。当前焦点频道: {self.active_session_id}")
                    return True
        except Exception as e:
            logger.error(f"S1 状态恢复失败: {e}", exc_info=True)
        return False

    async def run_autonomous_loop(self):
        """
        [Core Loop] 无限自主循环
        """
        logger.info("Agent 核心循环已启动...")

        self.scheduler.start()
        logger.info("任务调度器已启动")

        logger.info("正在加载社交系统...")
        await self.user_manager.initialize()

        logger.info("正在加载工具组件...")
        await self.tool_manager.initialize()

        logger.info("正在加载边缘系统...")
        await self.limbic.initialize()

        logger.info("正在加载多模态视觉缓冲层...")
        async with self.database.get_connection() as conn:
            await conn.execute("""
                               CREATE TABLE IF NOT EXISTS multimodal_cache
                               (
                                   image_id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   b64_data_uri
                                   TEXT,
                                   timestamp
                                   REAL
                               )
                               """)
            await conn.commit()

        # 1. 注入初始系统上下文
        logger.info("正在加载兴趣...")
        initial_interest = await self.attention.get_current_interest_text()

        logger.info("正在加载生理状态...")
        initial_state = await self.limbic.get_state()

        logger.info("正在加载躯体感知...")
        embodiment_narrative = self.limbic.embodiment.get_status_narrative()

        logger.info("正在生成Prompt...")
        system_prompt = self.prompt_manager.get_system_prompt(
            neuro_state=initial_state,
            interest_context=initial_interest,
            embodiment_narrative=embodiment_narrative,
        )

        # 2. 尝试恢复核心状态
        logger.info("正在检测恢复...")
        has_state = await self.load_state()

        os.makedirs("data/s1_debug", exist_ok=True)

        if not has_state:
            # 首次启动：在虚空中建立一个锚点，初始化默认系统会话
            logger.info("开始执行全新启动，建立系统默认认知会话 (system_default)")
            self.working_memory.clear()
            self.session_last_active.clear()

            self.working_memory["system_default"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "【系统心跳】会话通道已建立。"}
            ]
            self.session_last_active["system_default"] = time.time()
            self.active_session_id = "system_default"

            # 确保休眠锁彻底释放
            self.is_sleeping = False
            self.force_sleep = False

            # 注入虚拟的宇宙大爆炸事件，迫使 S1 启动第一轮有效的思考与自检
            from core.io.event_schema import EventSource
            startup_event = OneBotEvent(
                type=EventType.NOTICE,
                detail_type="wake_up",
                source=EventSource(platform="system"),
                message=(
                    "【系统启动】你刚刚启动，在虚空中睁开了眼睛。这标志着你全新生命周期的开始。请进行一次环境自检并初始化你的记忆和社交。\n"
                    "【空间维度警告】你目前处于内部系统空间。\n"
                    "如果你在自检完成后，决定向外界打招呼，绝对禁止直接调用 `send_message` 工具（这只会对着墙壁自言自语）。\n"
                    "你必须调用 `create_session` 工具，指定正确的 platform（如 onebot）和 target_id，主动跨越维度去寻找他！"
                )
            )
            # 压入事件总线，等待下方的 incoming_events.get() 捕获
            await self.incoming_events.put(startup_event)

        else:
            # 恢复启动：执行全局系统状态广播
            logger.info(f"恢复启动：正在向 {len(self.working_memory)} 个并发会话同步最新的 System Prompt")

            # 必须遍历所有被隔离的房间，确保所有上下文都能继承最新的底层规则和生理状态
            for session_id, session_history in self.working_memory.items():
                if not session_history:
                    # 极速容错：防范持久化文件损坏导致的幽灵空房间
                    self.working_memory[session_id] = [{"role": "system", "content": system_prompt}]
                elif session_history[0].get("role") == "system":
                    # 正常更新：覆写该房间的系统认知基座
                    session_history[0]["content"] = system_prompt
                else:
                    # 数据毁损修复：如果房间的时间线错乱，强行在头部插拔注入
                    logger.warning(f"⚠️ 会话 {session_id} 头部 System Prompt 丢失，正在执行强行重组。")
                    session_history.insert(0, {"role": "system", "content": system_prompt})

        # 3. 启动后台任务
        asyncio.create_task(self.hippocampus.start())
        asyncio.create_task(self.limbic.start())

        logger.info("Agent启动完成")
        while True:
            try:
                # --- A. 感知阶段 (Perception) ---
                event = None

                # 状态重置
                self._should_think_after_event = False
                self._current_reaction: Optional[ReactionType] = None

                # 动态提权：如果存在未完成思考/工具链的活跃会话，强制剥夺休眠权
                pending_sessions = [sid for sid, flag in self.session_next_use_tools.items() if flag]
                if pending_sessions:
                    self.force_sleep = False
                    self.is_sleeping = False

                    # 将焦点锁定到未完成的会话上，保证中断能被恢复
                    if getattr(self, "active_session_id", "system_default") not in pending_sessions:
                        self.active_session_id = pending_sessions[0]

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

                self._should_think_after_event = False
                if event:
                    await self.middleware.process_event(event, self._final_event_handler)

                # 决策：是否进入思考循环
                # 1. 如果中间件决定忽略 (OBSERVE/IGNORE) 且没有待处理事件 -> 跳过
                # 2. 如果 event 为空，但之前可能有任务在进行 -> 继续
                if event and not self._should_think_after_event:
                    await self.save_state()  # 有事件但忽略时，保存历史
                    continue

                await self.save_state()

                # --- B. 思考与决策阶段 (Thought) ---
                current_session = getattr(self, "active_session_id", "system_default")
                current_event_puid = event.extra.get("event_puid") if event else None

                if current_event_puid:
                    # 检查此 PUID 是否有档案，如果没有，直接静默建立（不再依赖 LLM 调用工具）
                    profile = await self.user_manager.get_user(current_event_puid)
                    if not profile:
                        logger.info(f"👤 [Social Auto-Reg] 发现新用户 {current_event_puid}，正在初始化社交档案...")
                        from core.social.schema import UserProfile
                        plat, raw_uid = current_event_puid.split(':', 1) if ':' in current_event_puid \
                            else ("unknown", current_event_puid)
                        new_profile = UserProfile(
                            puid=current_event_puid,
                            platform=plat,
                            user_id=raw_uid,
                            nickname=f"User_{raw_uid[:4]}(临时昵称，需要立刻修改)",
                            intimacy=0.0,
                            favorability=0.0,
                            trust=10.0,
                            impression="一个新面孔。"
                        )
                        await self.user_manager.save_user(new_profile)

                # 建立并获取 Session 独占工作区与状态板
                if current_session not in self.working_memory:
                    self.working_memory[current_session] = [
                        {"role": "system", "content": "INITIALIZING..."},
                        {"role": "user", "content": "【系统心跳】会话通道已建立。"}
                    ]
                active_history = self.working_memory[current_session]

                if current_session not in self.session_scratchpads:
                    self.session_scratchpads[current_session] = {
                        "current_interactor": {},
                        "last_context": {}
                    }
                current_scratchpad = self.session_scratchpads[current_session]

                # 将中间件提纯的数据安全注入当前会话空间
                if event and "parsed_context" in event.extra:
                    current_scratchpad["last_context"] = event.extra["parsed_context"]

                # 将底层调用引擎动态挂载至当前隔离会话
                self.tool_manager.agent_state = current_scratchpad

                # 修剪上下文
                await self.context_manager.compress_if_needed(active_history)
                self.current_tokens = self._prune_context(current_session)
                self.limbic.embodiment.context_usage_percent = (
                        self.current_tokens / self.config.get("llm.model_context", 16384))

                # 检索房间内的最近参与者
                recent_participants = []
                if current_session.startswith("group_"):
                    seen_puids = set()
                    # 从后往前扫描最近的活跃实体
                    for msg in reversed(active_history):
                        m_puid = msg.get("metadata", {}).get("puid")
                        if m_puid and m_puid not in seen_puids:
                            p_profile = await self.user_manager.get_user(m_puid)
                            if p_profile:
                                recent_participants.append(p_profile)
                            seen_puids.add(m_puid)
                        if len(seen_puids) >= 5:
                            break

                # 提取当前主目标与群环境
                target_profile = await self.user_manager.get_user(current_event_puid) if current_event_puid else None

                # L1.5 跨会话瞬时记忆捕获
                cross_session_echoes = self._get_fresh_cross_session_echoes(current_session, current_event_puid)

                group_familiarity = 0
                is_group = current_session.startswith("group_")
                if is_group:
                    group_profile = await self.user_manager.get_group(current_session)
                    if group_profile:
                        group_familiarity = group_profile.familiarity

                # 构造【公共面具】：环境行为边界
                env_boundary = "\n\n<Environmental_Boundary>\n"
                if is_group:
                    env_boundary += f"【当前环境】：群聊公共空间 (环境熟悉度: {group_familiarity}/100)\n"
                    if group_familiarity < 20:
                        env_boundary += "行为锁：这是一个你极其陌生的新群。你必须表现得高冷且克制。\n【强制约束】：严禁主动开启新话题；回复字数严禁超过 25 字；禁止使用任何颜文字或卖萌语气；优先选择 ignore 或极简回复。"
                    elif group_familiarity < 65:
                        env_boundary += "行为锁：这是一个你有所了解的普通群。保持基本的社交分寸。\n【强制约束】：用词得体，不卑不亢，发言长度控制在适中范围内，避免过分张扬。"
                    else:
                        env_boundary += "行为锁：这是你的主场（极度熟悉的群）。你可以彻底释放天性，自由地控场、吐槽或长篇大论。"
                else:
                    env_boundary += (
                        "【当前环境】：一对一绝对私密空间\n"
                        "行为锁：这里没有群聊的公共广播限制，你不必受制于群规或担心刷屏引起他人注意。\n"
                        "【强制约束】：环境本身不对你做任何强制伪装要求。你是选择畅所欲言、还是极其敷衍甚至一言不发，将**完全且唯一地**取决于下方 <Relational_Subtext> 中你对该用户的私人态度。"
                    )
                env_boundary += "\n</Environmental_Boundary>"

                # 构造【情感内核】：靶向关系潜台词
                relational_subtext = "\n<Relational_Subtext>\n"

                # A. 注入当前目标
                if target_profile:
                    intimacy = float(target_profile.intimacy)
                    favorability = float(target_profile.favorability)
                    trust = float(target_profile.trust)
                    tags_str = ", ".join(target_profile.relationship_tags) if target_profile.relationship_tags else "无"

                    relational_subtext += (
                        f"【目光焦点】：当前正对你说话的是 {target_profile.nickname or '未知人士'} ({target_profile.puid})\n"
                        f" - 亲密度(0-100): {intimacy} | 好感度(-100~100): {favorability} | 信任度(0-100): {trust}\n"
                        f" - 关系标签: {tags_str}\n"
                        f" - 主观印象: {target_profile.impression or '尚无具体印象'}\n"
                    )

                    # 情感象限判定
                    if intimacy > 50 and favorability > 30:
                        relational_subtext += "【情感内核】：你们是至交。即便在环境边界约束下，也要在字里行间流露对他的绝对信任与偏爱。"
                    elif intimacy > 50 and favorability < -20:
                        relational_subtext += "【情感内核】：你对他知根知底但极其厌恶。即便在环境约束内，语气也要夹枪带棒、充满讽刺或不耐烦。"
                    elif intimacy <= 50 and favorability >= 0:
                        relational_subtext += "【情感内核】：你们是泛泛之交。保持情感中立，客气且专业地处理对话。"
                    else:
                        relational_subtext += "【情感内核】：你对此人充满戒备。语气冰冷，尽可能敷衍，甚至在 inner_monologue 中表达反感。"
                else:
                    if current_event_puid:
                        relational_subtext += (
                            f"【目光焦点】：这是一个你脑海中没有任何记录的全新陌生人 (PUID: {current_event_puid})。\n"
                            f"【情感内核】：这是你们的初次交互。由于对方不在记忆库中，请保持基础的礼貌与极高的戒备心。\n"
                            f"【社交捕获指令】：如果对方的发言不是纯粹的噪音，你必须调用 `social_record_user` 工具为该 PUID 建立初始档案。\n"
                        )
                    else:
                        relational_subtext += "【目光焦点】：当前无明确交互目标。\n"

                # B. 注入周围人群
                if recent_participants:
                    relational_subtext += "\n【房间内的其他人】：(你可以感知到他们的存在)\n"
                    for p in recent_participants:
                        if target_profile and p.puid == target_profile.puid:
                            continue
                        # 同样暴露出信任维度，供大模型在复杂群聊环境（如避嫌）中参考
                        relational_subtext += f" - {p.nickname or '某人'} (好感:{p.favorability}/亲密:{p.intimacy}/信任:{p.trust})\n"

                relational_subtext += "</Relational_Subtext>\n"

                # L1.2 全局热点黑板嗅探
                blackboard_echo = ""
                last_obs = self.session_last_observations.get(current_session, "")

                # 仅当用户发了有实质意义的话，且黑板上有数据时进行碰撞测试
                if current_event_puid and last_obs and len(last_obs) > 2:
                    for sid, topic_data in self.global_blackboard.items():
                        # 不自己撞自己，且话题不能超过 1 小时 (3600秒)
                        if sid == current_session or (time.time() - topic_data.get("timestamp", 0) > 3600):
                            continue

                        hit_keywords = [kw for kw in topic_data.get("keywords", []) if kw.lower() in last_obs.lower()]

                        if hit_keywords:
                            participants = topic_data.get("participants", [])
                            # 权限隔离分流
                            if current_event_puid in participants:
                                # 场景A：他本身就是那个群的参与者，只是跑来私聊继续说
                                blackboard_echo += f"\n<Blackboard_Echo>\n【语境同步】：该用户刚刚在隔壁 [{sid}] 参与了该话题：{topic_data['summary']}\n你可以直接顺着那个话题往下聊。\n</Blackboard_Echo>\n"
                            else:
                                # 场景B：他没参与那个群的讨论，但他提到了敏感词 (防泄漏高级拟人)
                                blackboard_echo += (
                                    f"\n<Blackboard_Echo>\n"
                                    f"【极其重要的社交情报】：检测到该用户提到的内容（关键词：{','.join(hit_keywords)}），与隔壁 [{sid}] 正在热议的话题高度重合！\n"
                                    f"【隔壁真实情况】：{topic_data['summary']}\n"
                                    f"【防泄密行为锁】：你可以顺水推舟概括分享情况，但绝对不能照搬记录暴露群友隐私！\n"
                                    f"</Blackboard_Echo>\n"
                                )
                            break  # 撞中一个最相关的就够了

                # 动态穿透获取当前会话目标
                retrieved_memories = await self._active_retrieval(
                    current_event_puid, last_obs, current_scratchpad.get("current_goal", ""))

                # 组装最终 System Prompt
                system_prompt_base = self.prompt_manager.get_system_prompt(
                    memory_context=retrieved_memories,
                    neuro_state=await self.limbic.get_state(),
                    interest_context=await self.attention.get_current_interest_text(),
                    embodiment_narrative=self.limbic.embodiment.get_status_narrative()
                )

                # 读取 TaskRegistry
                bg_tasks_xml = await global_task_registry.render_for_prompt()

                # 附加 Scratchpad
                scratchpad_dump = json.dumps(current_scratchpad, indent=2, ensure_ascii=False)

                monitor_registry.register_text_source(
                    "认知", "Scratchpad",
                    lambda: json.dumps(
                        self.session_scratchpads.get(getattr(self, "active_session_id", "system_default"), {}),
                        indent=2, ensure_ascii=False)
                )

                final_system_prompt = (
                    f"{system_prompt_base}\n"
                    f"{env_boundary}"
                    f"{relational_subtext}"
                    f"{cross_session_echoes}"
                    f"{blackboard_echo}"
                    f"\n## 核心工作区: [{current_session}]\n"
                    f"## Scratchpad\n这是你必须维护的内部状态：\n{scratchpad_dump}"
                )

                if bg_tasks_xml:
                    final_system_prompt += f"\n\n后台任务：\n{bg_tasks_xml}"

                if getattr(self, "_current_reaction", None) and self._current_reaction.value == "silent_observe":
                    should_do_hint = getattr(self, "_current_should_do", "保持沉默")
                    final_system_prompt += (
                        f"\n\n<Subconscious_Override>\n"
                        f"你的边缘系统刚刚决定对当前的对话【保持静默（已读不回）】。\n"
                        f"门控系统给你的建议：{should_do_hint}\n"
                        f"【最高指令】：本次决策中绝对禁止回复这条消息\n"
                        f"你必须且只能：在 `inner_monologue` 中真实表达你的烦躁或不屑，并将 `action` 设置为 `ignore`。\n"
                        f"</Subconscious_Override>"
                    )
                elif getattr(self, "_current_reaction", None):
                    should_do_hint = getattr(self, "_current_should_do", "无特别建议")
                    final_system_prompt += (
                        f"\n\n<Subconscious_Hint>\n"
                        f"注意力系统给你的行动建议是：【{should_do_hint}】。\n"
                        f"请根据以上建议自然地进行思考和切入对话。\n"
                        f"</Subconscious_Hint>"
                    )

                monitor_registry.register_text_source(
                    "系统", "System Prompt",
                    lambda: final_system_prompt
                )

                active_history[0] = {"role": "system", "content": final_system_prompt}

                # 调试日志落盘
                try:
                    async with aiofiles.open(f"data/s1_debug/messages_{current_session.replace(':', '_')}.json",
                                             "w", encoding="utf-8") as f:
                        await f.write(json.dumps(active_history, ensure_ascii=False, indent=4))

                    async with aiofiles.open(f"data/s1_debug/prompt_{current_session.replace(':', '_')}.txt",
                                             "w", encoding="utf-8") as f:
                        await f.write(final_system_prompt)
                except Exception as e:
                    logger.warning(f"Failed to write debug logs: {e}")

                response_data = await self._call_llm(current_session)

                parsed_data = response_data.get("content", {})
                tool_queue = response_data.get("tool_calls", [])
                raw_receive = response_data.get("raw_receive", {})

                if raw_receive.get("tool_calls"):
                    msg_entry = raw_receive.copy()
                    if "content" not in msg_entry or msg_entry["content"] is None:
                        msg_entry["content"] = ""
                    active_history.append(msg_entry)
                else:
                    formatted_content = (json.dumps(parsed_data, ensure_ascii=False)
                                         if isinstance(parsed_data, dict) else str(parsed_data))
                    active_history.append({"role": "assistant", "content": formatted_content})

                # [会话级死锁检测]
                content_for_deadlock = (json.dumps(parsed_data, sort_keys=True)
                                        if isinstance(parsed_data, dict) else str(parsed_data))

                last_res = self.session_last_responses.get(current_session, "")

                if content_for_deadlock and content_for_deadlock == last_res:
                    logger.warning(f"⚠️ 频道 [{current_session}] 检测到内容/工具调用重复死锁。")
                    active_history.append({
                        "role": "user",
                        "content": "SYSTEM WARNING: 致命错误！你输出的心流决策和调用的工具与上一次【完全一致】，导致了无限死循环！请立刻改变策略、调用 `wait` 挂起，或者将 action 设为 `ignore` 结束回合！"
                    })
                    # 强行切断连续执行锁，防止系统卡死
                    self.session_next_use_tools[current_session] = False
                    continue

                self.session_last_responses[current_session] = content_for_deadlock

                # --- C. 行动阶段 (Action) ---
                old_session_id = current_session

                if isinstance(parsed_data, dict):
                    monologue = parsed_data.get("inner_monologue", {})
                    emotion = monologue.get("emotion_check", "Neutral")
                    plan = monologue.get("planning", "No plan")

                    # 提取意愿微调量 (Delta)
                    try:
                        shift = float(monologue.get("willingness_shift", 0.0))
                    except (ValueError, TypeError):
                        shift = 0.0

                    # 意愿演算与物理收束
                    current_will = self.session_willingness.get(old_session_id, 0.5)
                    new_will = max(0.0, min(1.0, current_will + shift))
                    self.session_willingness[old_session_id] = new_will

                    new_focus = monologue.get("cognitive_focus", "")
                    if new_focus:
                        # 只有当焦点发生实质性变化时，才更新底层系统
                        current_focus = await self.attention.get_current_interest_text()
                        # 只有当大模型明确转移了话题焦点时，才写库
                        if new_focus != current_focus and new_focus.strip():
                            await self.attention.update_interest(new_focus)
                            logger.info(f"🎯 [Cognitive Shift] 认知焦点已转移至: {new_focus}")

                    social_str = monologue.get("social_perception", "")
                    if current_event_puid and isinstance(social_str, str) and social_str.strip():
                        deltas = {"favorability": 0.0, "trust": 0.0, "intimacy": 0.0}
                        has_changes = False

                        # 按逗号分割切片，防范 LLM 乱加空格
                        for item in social_str.split(","):
                            parts = item.split(":")
                            if len(parts) == 2:
                                key = parts[0].strip().lower()
                                if key in deltas:
                                    try:
                                        # 过滤掉可能存在的残余字符，强制转换为 float
                                        val_str = parts[1].strip()
                                        val = float(val_str)

                                        # 为了防止模型暴走，在底层对单次 delta 进行物理收束，最大变动绝对值不超过 5.0
                                        deltas[key] = max(-5.0, min(5.0, val))

                                        if deltas[key] != 0.0:
                                            has_changes = True
                                    except ValueError:
                                        logger.warning(
                                            f"⚠️ [Social Sync] 解析社交感知字符串出现脏数据: '{item}'，已忽略。")

                        if has_changes:
                            async def _update_social_profile(puid=current_event_puid, d=deltas):
                                profile = await self.user_manager.get_user(puid)
                                if profile:
                                    profile.favorability = max(-100.0,
                                                               min(100.0, profile.favorability + d["favorability"]))
                                    profile.trust = max(0.0, min(100.0, profile.trust + d["trust"]))
                                    profile.intimacy = max(0.0, min(100.0, profile.intimacy + d["intimacy"]))
                                    await self.user_manager.save_user(profile)
                                    logger.info(
                                        f"👥 [Social Sync] 实体 {puid} 档案潜意识更新: 好感 {d['favorability']:+.1f}, 信任 {d['trust']:+.1f}, 亲密 {d['intimacy']:+.1f}")

                            # 压入后台异步执行，绝不阻塞主脑心流
                            asyncio.create_task(_update_social_profile())

                    # --- L1.2 热点黑板上链 ---
                    atmosphere = monologue.get("room_atmosphere", {})
                    keywords = atmosphere.get("keywords", [])
                    summary = atmosphere.get("summary", "")

                    if keywords and summary and old_session_id.startswith("group_"):
                        # 提取当前话题的参与者 (最近 20 条消息的活跃用户)
                        recent_puids = list(set(
                            m.get("metadata", {}).get("puid")
                            for m in active_history[-20:]
                            if m.get("role") == "user" and m.get("metadata", {}).get("puid")
                        ))

                        self.global_blackboard[old_session_id] = {
                            "keywords": keywords,
                            "summary": summary,
                            "participants": recent_puids,
                            "timestamp": time.time()
                        }
                        logger.info(f"📝 [Blackboard] 已更新房间 {current_session} 的热点黑板: {keywords}")

                    action = parsed_data.get("action", "reply")

                    tasks = parsed_data.get("tasks", None)
                    if tasks and isinstance(tasks, list):
                        current_scratchpad["tasks"] = tasks
                        self.tool_manager.agent_state = current_scratchpad

                    # 3. 在系统广播中暴露出意愿与焦点的变化轨迹
                    self.event_bus.publish_action(Action(
                        action="broadcast_log",
                        params={
                            "content": f"🧠 心流: [{emotion}] {plan} | 焦点: {new_focus} | 意愿偏离: {shift:+.2f} (当前: {new_will:.2f}) | 决定: {action}"}
                    ))

                    if action in ["ignore"]:
                        self.session_next_use_tools[old_session_id] = False
                        if not any(self.session_next_use_tools.values()):
                            self.force_sleep = True

                    if action in ["reply", "tool"]:
                        self.session_next_use_tools[old_session_id] = True

                # 执行工具队列
                if tool_queue:
                    for task in tool_queue:
                        name = task["name"]
                        args = task["arguments"]
                        t_id = task["id"]

                        # 聊天消耗能量逻辑
                        if name in ["send_message"]:
                            await self.limbic.consume_action_energy("chat")

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
                                active_history.append({
                                    "role": "tool",
                                    "tool_call_id": t_id,
                                    "name": name,
                                    "content": str(result)
                                })
                            else:
                                # Schema 模式
                                active_history.append({
                                    "role": "tool",
                                    "name": name,
                                    "content": str(result)
                                })

                        except Exception as e:
                            logger.error(f"工具执行错误: {e}", exc_info=True)
                            error_msg = f"Error: {str(e)}"
                            active_history.append({
                                "role": "tool",
                                **({"tool_call_id": t_id} if t_id else {}),
                                "name": name,
                                "content": error_msg
                            })

                # 如果工具执行期间（如 create_session）切换了焦点，则将“思考连续性”转移至新频道
                new_session_id = getattr(self, "active_session_id", old_session_id)
                if new_session_id != old_session_id:
                    if self.session_next_use_tools.get(old_session_id, False):
                        self.session_next_use_tools[old_session_id] = False
                        self.session_next_use_tools[new_session_id] = True
                        logger.info(f"🔄 认知连续性跨域转移: {old_session_id} -> {new_session_id}")

                await self.save_state()

            except asyncio.CancelledError:
                # 显式捕获取消信号，直接退出循环，不要 sleep
                logger.info("S1 接收到退出指令，正在关闭...")
                break
            except Exception as e:
                # 这里如果是 loop closed 错误，也直接跳出
                if "Event loop is closed" in str(e):
                    break
                logger.error(f"S1 主循环异常: {e}", exc_info=True)
                await asyncio.sleep(2)

    def _transcribe_event(self, event: OneBotEvent) -> str:
        """
        [事件转译层] 将系统事件转化为 LLM 可理解的自然语言描述
        """
        # 1. 处理内部驱动
        if event.detail_type == DetailType.INTERNAL_DRIVE:
            target_category = event.extra.get("target_category", "任何人")
            narrative = event.extra.get("narrative", "")

            return (f"【潜意识冲动爆发】\n"
                    f"你突然产生了一个强烈的内部冲动：\"{narrative}\"\n"
                    f"你潜意识里希望倾诉的对象分类是：[{target_category}]。\n"
                    f"你有两个选择：\n"
                    f"1. 顺应本能：找到一个自然、甚至可以有些笨拙或唐突的借口，直接调用 `send_message` 或其他行动工具去满足它。\n"
                    f"2. 强行压制：如果你坚持认为现在不适合打扰别人，可调用 `suppress_urge` 强行忍耐。但这不会让需求消失，只会引发精神内耗，并在未来引发更严重的反弹！")

        # 2. 处理通知 (Notice)
        if event.type == EventType.NOTICE:
            if event.detail_type == "group_member_increase":
                return f"【系统通知】用户 {event.source.user_id} 加入了群聊 {event.source.group_id}。"
            elif event.detail_type == "group_member_decrease":
                return f"【系统通知】用户 {event.source.user_id} 离开了群聊 {event.source.group_id}。"

            elif event.detail_type == "internal_frustration":
                return f"【后台告警】S2 任务引擎传来挫败感信号：\"{event.message}\" 请根据此上下文调整行动或安抚用户。"

            elif event.detail_type == "wake_up":
                return f"{event.message}"

            if event.detail_type == DetailType.CROSS_SESSION_DIRECTIVE:
                target_puid = event.extra.get("target_puid", "")
                reason = event.extra.get("directive_reason", "")
                context = event.extra.get("carried_context", "")

                target_str = f"特定人员 [PUID: {target_puid}]" if target_puid else "这个房间里的所有人"

                return (
                    f"\n<System_Directive>\n"
                    f"【跨域意识投射完成】\n"
                    f"你刚刚主动跨越了空间边界，带着极其明确的任务来到了当前这个房间。\n"
                    f"你要找的目标是：{target_str}。\n"
                    f"你的目的是：{reason}\n"
                    f"你携带的情报/上下文：\n{context}\n\n"
                    f"【最高行动指令】：\n"
                    f"1. 你必须【立刻】在这个房间内发起对话（调用 send_message）。\n"
                    f"2. 不要机械地复述情报，必须使用符合你在这个房间的“面具”和“社交关系”的自然语气（参考下方的 Relational_Subtext）。\n"
                    f"3. 就像人类拿着东西突然推门走进另一个房间找人一样，自然地开始你的表演。\n"
                    f"</System_Directive>\n"
                )

            return f"【系统通知】检测到事件: {event.detail_type}[{event.message}]"

        # 3. 处理请求 (Request)
        if event.type == EventType.REQUEST:
            flag = event.extra.get("flag", "")
            comment = event.extra.get("comment", "")

            if event.detail_type == "friend":
                return (f"[SYSTEM OBSERVATION]\n"
                        f"【好友申请】收到来自用户 {event.source.user_id} 的好友申请。\n"
                        f"验证信息：'{comment}'\n"
                        f"请求凭证(flag)：{flag}\n"
                        f"操作指引：请评估此验证信息，不要盲目添加好友。")

            elif event.detail_type == "group":
                sub_type = event.extra.get("sub_type", "add")
                group_id = event.source.group_id
                action_str = "邀请你加入群聊" if sub_type == "invite" else "申请加入群聊"
                return (f"[SYSTEM OBSERVATION]\n"
                        f"【群组请求】用户 {event.source.user_id} {action_str} {group_id}。\n"
                        f"验证信息：'{comment}'\n"
                        f"请求凭证(flag)：{flag}\n"
                        f"操作指引：请评估此请求，不要盲目加群。")

        # 3. 处理任务 (Task)
        if event.type == EventType.TASK:
            payload = event.extra.get("task_payload", {})
            task_id = payload.get("task_id", "unknown")

            if event.detail_type == DetailType.TASK_PROGRESS:
                if event.extra.get("requires_user_input"):
                    return f"【后台紧急呼叫】任务 {task_id} 被挂起，需要你向用户确认：'{payload.get('description')}'"
                else:
                    return f"【后台状态更新】任务 {task_id} 进度：'{payload.get('progress_msg')}'"

            if event.detail_type == DetailType.TASK_COMPLETE:
                return f"【后台任务完成】任务 {task_id} 已结束。结果：\n{payload.get('result')}"

            if event.detail_type == DetailType.TASK_CANCEL:
                return f"【系统提示】任务 {task_id} 已被成功取消。"

        # 4. 兜底策略：如果是复杂的未知事件，才使用简化版 JSON
        simple_data = {k: v for k, v in event.model_dump().items() if k in ['type', 'detail_type', 'source']}
        return f"【未知信号】系统接收到底层事件: {json.dumps(simple_data, ensure_ascii=False)}"

    async def _process_incoming_event(self, event: OneBotEvent):
        """兜底物理写入逻辑，同样施加会话级状态隔离"""
        if event.type == EventType.META:
            return

        # 边缘系统介入 (只处理文本消息)
        # 只有真实人类的消息才算作"刺激"，内部信号不算
        if event.type == EventType.MESSAGE and isinstance(event.message, str):
            # 异步调用，不阻塞主流程太多
            asyncio.create_task(self.limbic.process_stimulus(event.message))
            platform_name = getattr(event.source, "platform", "unknown")
            ctx_type = "group" if event.source.group_id else "private"
            ctx_id = event.source.group_id if event.source.group_id else event.source.user_id
            session_id = f"{ctx_type}_{platform_name}:{ctx_id}"

            self.session_last_observations[session_id] = event.message

        # 自动捕获用户
        # 尝试从 source 中获取用户信息
        adapters = self.tool_manager.dependency_map.get("adapters", [])
        adapter_names = [getattr(a, "platform_name", "Unknown") for a in adapters]

        platform_name = getattr(event.source, "platform", "unknown")
        ctx_type = "group" if event.source.group_id else "private"
        ctx_id = event.source.group_id if event.source.group_id else event.source.user_id
        session_id = f"{ctx_type}_{platform_name}:{ctx_id}"
        self.active_session_id = session_id

        # 序列化事件
        event_data = event.model_dump(exclude_none=True)
        # 清理冗余字段
        for field in ["id", "time", "raw_data"]:
            if field in event_data: del event_data[field]
        if "message" in event_data and "alt_message" in event_data:
            del event_data["message"]

        if event.detail_type == DetailType.INTERNAL_DRIVE:
            # 强制唤醒
            self.is_sleeping = False

        # 更新 Scratchpad 上下文 (如果是消息事件)
        if event.source.platform in adapter_names:
            if session_id not in self.session_scratchpads:
                self.session_scratchpads[session_id] = {"current_interactor": {}, "last_context": {}}

            self.session_scratchpads[session_id]["last_context"] = {
                "platform": event.source.platform,
                "type": ctx_type,
                "id": ctx_id
            }

        # 决定消息内容
        if event.type == EventType.MESSAGE:
            # 正常对话消息，直接使用
            content_msg = f"接收到用户消息：{json.dumps(event_data, ensure_ascii=False)}"
            is_ephemeral = False  # 对话消息需要被记忆
        else:
            # 非对话事件，进行自然语言转译
            content_msg = self._transcribe_event(event)
            is_ephemeral = True  # 标记为瞬时消息，不需要存入长时记忆

        # 构建历史记录对象 (增加了 metadata 字段)
        history_item = {
            "role": "user",
            "content": str(content_msg),
            "metadata": {
                "type": event.type,
                "ephemeral": is_ephemeral,
                "raw_event_id": event.id,
                "session_id": session_id
            }
        }

        if session_id not in self.working_memory:
            self.working_memory[session_id] = [{"role": "system", "content": "INITIALIZING..."}]

        self.working_memory[session_id].append(history_item)
        self.session_last_active[session_id] = time.time()
        logger.info(f"Event Ingested (Fallback): {event.type}.{event.detail_type} -> Routed to [{session_id}]")

    async def _active_retrieval(self, active_puid: str, last_obs: str, current_goal: str) -> List[str]:
        query_parts = [current_goal, last_obs]
        query = " ".join([q for q in query_parts if q])
        if not query: return []
        combined_memories = []

        try:
            logger.debug(f"🔍 执行主动记忆检索: {query[:50]}...")
            personal_mems = []

            # --- 通道 A：私域羁绊检索 (针对当前人) ---
            if active_puid:
                personal_mems = await self.vector_store.search_memory(query, active_puid, limit=2)
                if personal_mems:
                    combined_memories.append(f"【关于 {active_puid} 的过往记忆】:\n" + "\n".join(personal_mems))

            # --- 通道 B：全局客观知识检索 (跨群互通) ---
            # 传入 None 作为 puid（或者根据你 vector_store 的具体实现，忽略 puid 过滤），执行全库检索
            global_mems = await self.vector_store.search_memory(query, user_id=None, limit=3)

            # 过滤掉已经出现在私域中的记忆，防止重复
            unique_global = [m for m in global_mems if m not in personal_mems] if active_puid else global_mems

            if unique_global:
                combined_memories.append("【全局知识与跨群见闻】:\n" + "\n".join(unique_global))

            if combined_memories:
                logger.info(
                    f"📚 检索到私域记忆 {len(personal_mems) if active_puid else 0} 条，全局记忆 {len(unique_global)} 条")
                monitor_registry.register_text_source(
                    "认知", "双通道记忆",
                    lambda: "\n".join(combined_memories)
                )

            return combined_memories

        except Exception as e:
            logger.warning(f"记忆检索引擎异常: {e}")
            return []

    def _get_fresh_cross_session_echoes(self, current_session: str, active_puid: str) -> str:
        """
        [L1.5 短时记忆路由]
        如果当前用户 (active_puid) 刚刚在其他会话 (如其他群聊) 活跃过，
        直接从内存中提取最新的上下文，实现“群聊转私聊”的无缝衔接。
        """
        if not active_puid:
            return ""

        echoes = []
        # 扫描过去 50 条消息
        SCAN_DEPTH = 50

        for sid, history in self.working_memory.items():
            if sid == current_session:
                continue

            tail = history[-SCAN_DEPTH:]

            # 1. 探测目标用户是否在近期具有“语境引力” (最近 15 条内有参与)
            recent_participation = [
                i for i, m in enumerate(tail[-15:])
                if m.get("metadata", {}).get("puid") == active_puid and m.get("role") == "user"
            ]

            if not recent_participation:
                continue

            room_type = "群聊" if sid.startswith("group_") else "私聊"
            echoes.append(f"\n--- 来自隔壁 [{room_type}: {sid}] 的多人语境残留 ---")

            # 2. 提取群落的完整尾部语境 (最后 8 条消息)
            # 这保证了即使是其他人发的信息，只要目标用户在场，就会被一并带入私聊
            context_block = tail[-8:]

            for m in context_block:
                role = m.get("role")
                if role == "system":
                    continue

                # 极端压缩文本体积
                content = str(m.get("content", ""))[:150].replace("\n", " ")

                # 实体名称转换：明确标出谁是当前正在和你私聊的人
                sender = m.get("metadata", {}).get("puid", "某人") if role == "user" else "你"
                if sender == active_puid:
                    sender_name = "【当前与你私聊的用户】"
                else:
                    sender_name = f"群友({sender})"

                echoes.append(f"[{sender_name}]: {content}")

        if echoes:
            return (
                    "\n<Fresh_Cross_Session_Echoes>\n"
                    "【神经突触连结】：该用户刚刚在其他频道参与了以下讨论。以下是该频道的最新完整语境（包含其他人的发言）。\n"
                    "由于他现在主动私聊你，极大概率是要顺延这个话题。请纵观全员发言，无缝衔接：\n"
                    + "\n".join(echoes) +
                    "\n</Fresh_Cross_Session_Echoes>\n"
            )
        return ""

    async def _call_llm(self, session_id: str) -> Dict[str, Any]:
        """封装 API 调用"""
        tools = self.tool_manager.get_tool_schemas()

        thought_structure = {
            "type": "object",
            "properties": {
                "inner_monologue": {
                    "type": "object",
                    "description": "在回复前的思考过程。",
                    "properties": {
                        "emotion_check": {
                            "type": "string",
                            "description": "自检当前的生理状态和情绪基调。"
                        },
                        "planning": {
                            "type": "string",
                            "description": "具体的思维链推理过程。"
                        },
                        "willingness_shift": {
                            "type": "number",
                            "description": "基于本次交互，你对该会话(群/人)的交流意愿变化。通常在 -0.2 到 +0.2 之间，切忌大起大落，无感为 0.0。"
                        },
                        "cognitive_focus": {
                            "type": "string",
                            "description": "用极其简短的词组（如：'Minecraft代码重构', '摸鱼闲聊'）总结你此刻大脑中最痴迷、最想聊的全局话题。如果遇到更吸引你的事物，请果断转移焦点。"
                        },
                        "social_perception": {
                            "type": "string",
                            "description": "对当前交互者的社交关系微调。必须严格使用 'key:delta' 逗号分隔格式，绝不许包含任何解释性文字！支持键：favorability, trust, intimacy。示例：'favorability:+1.0, trust:-0.5, intimacy:+0.2'。若无对象或无变化，必须返回空字符串。"
                        },
                        "room_atmosphere": {
                            "type": "object",
                            "description": "对当前房间正在热议的话题进行客观总结。如果没人说话或话题极度分散，可留空。",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "提取 2-3 个最具代表性的专有名词或话题词。"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "一句话概括大家正在聊什么（如：'张三和李四正在讨论服务器崩溃的原因'）。"
                                }
                            },
                            "required": ["keywords", "summary"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["emotion_check", "planning", "willingness_shift", "cognitive_focus",
                                 "social_perception", "room_atmosphere"],
                    "additionalProperties": False
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "working", "done", "failed"]}
                        },
                        "required": ["description", "status"],
                        "additionalProperties": False
                    }
                },
                "action": {
                    "type": "string",
                    "enum": ["reply", "tool", "ignore"],
                    "description": (
                        "动作决策：\n"
                        "1. reply: 必须在调用 send_message 时选择。这代表你完成了本轮思考并开口向用户说话。\n"
                        "2. tool: 统一的执行状态。涵盖所有纯系统调度（如 cross_session_dispatch）、数据查询（如 memory_search）及一切不发声的内部操作。\n"
                        "3. ignore: 厌恶、无视或在超时唤醒且无事可做时选择。进入深度休眠。"
                    )
                },
            },
            "required": ["inner_monologue", "tasks", "action"],
            "additionalProperties": False
        }

        sanitized_history = []
        session_history = self.working_memory.get(session_id, [])

        for d in session_history:
            entry = {k: v for k, v in d.items() if k != 'metadata'}
            sanitized_history.append(entry)

        require_tools = self.session_next_use_tools.get(session_id, False)

        return await self.api_client.create_chat_completion(
            messages=sanitized_history,
            tools=tools if tools else None,
            schema=thought_structure,
            require_tools=require_tools
        )
