import asyncio
import json
import logging
import time
import traceback
from typing import List, Dict, Any

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, Action
from core.kernel.prompt import PromptManager
from core.memory.hippocampus import Hippocampus
from core.memory.vector_store import VectorStore
from core.tool_manager.aggregator import ToolManager

logger = logging.getLogger(__name__)


class AutonomousAgent:
    def __init__(self, config: Config, event_bus: EventBus, database: Database):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.api_client = GenericAPIClient(config)
        self.prompt_manager = PromptManager(config)
        self.last_interaction_time = None

        # 初始化记忆组件
        self.hippocampus = Hippocampus(config, self.api_client, database)
        self.vector_store = VectorStore(database, self.api_client)

        # 初始化任务调度器
        self.scheduler = AsyncIOScheduler()

        # 记录唤醒任务的 ID
        self.wakeup_job_id = None

        # 睡眠状态标记
        self.is_sleeping = False

        # --- 内部状态 ---
        self.history: List[Dict[str, Any]] = []
        self.scratchpad: Dict[str, Any] = {
            "goal": "System Standby",
            "progress": "All systems initialized",
            "next_action": "Await user instructions"
        }

        # --- 初始化工具管理器 (注入依赖) ---
        self.tool_manager = ToolManager(
            config=config,
            event_bus=event_bus,
            api_client=self.api_client,
            database=database,
            agent_state=self.scratchpad
        )

        # 注入 Scheduler 和 Agent 自身
        self.tool_manager.set_scheduler(self.scheduler)
        self.tool_manager.set_agent(self)

        # 消息缓冲区 (处理中断)
        self.incoming_events: asyncio.Queue = asyncio.Queue()
        self.event_bus.subscribe_event(self._enqueue_event)

    async def _enqueue_event(self, event: OneBotEvent):
        """回调：将总线事件放入缓冲区"""
        await self.incoming_events.put(event)

    async def run_autonomous_loop(self):
        """
        [Core Loop] 无限自主循环
        不需要用户先说话，系统启动即开始思考。
        """
        logger.info("Agent 核心循环已启动...")

        # 启动调度器
        self.scheduler.start()
        logger.info("任务调度器已启动")

        # 异步加载工具 (包括 MCP)
        logger.info("正在加载工具组件...")
        await self.tool_manager.initialize()

        # 3. 初始 Prompt 注入检索到的记忆 (RAG)
        # 简单起见，这里先检索 "Context" 相关的
        mems = await self.vector_store.search_memory("重要信息", "admin_console")
        mem_context = "\n".join(mems) if mems else "暂无记忆"

        # 1. 注入初始系统上下文
        system_prompt = self.prompt_manager.get_system_prompt()
        self.history.append({"role": "system", "content": system_prompt + f"\n\n# Memory Context\n{mem_context}"})

        # 2. 启动海马体后台任务
        asyncio.create_task(self.hippocampus.start())

        # 3. 注入启动信号
        self.history.append({
            "role": "user",
            "content": f"系统启动完成。\n"
                       f"指令：检查你的工具并核对系统状态。"
        })

        while True:
            try:
                # --- A. 感知阶段 (Perception) ---
                event = None

                # 根据状态决定等待策略
                if self.is_sleeping:
                    # 如果处于休眠状态，完全阻塞等待，直到有新事件（用户输入 或 Timer唤醒）
                    # 只有收到事件后，Agent 才会“醒来”
                    event = await self.incoming_events.get()
                    self.is_sleeping = False  # 收到事件，解除休眠
                    logger.info(f"⏰ Agent 结束休眠，收到事件: {event.type}")
                else:
                    # 如果处于活跃状态，使用短超时轮询，保持自主思考能力
                    try:
                        event = await asyncio.wait_for(self.incoming_events.get(), timeout=1.0)
                        self.is_sleeping = False
                        if self.wakeup_job_id:
                            try:
                                self.scheduler.remove_job(self.wakeup_job_id)
                                logger.debug(f"已取消剩余的唤醒定时器: {self.wakeup_job_id}")
                            except JobLookupError:
                                pass
                            self.wakeup_job_id = None
                        logger.info(f"⏰ Agent 结束休眠，收到事件: {event.type}")
                    except asyncio.TimeoutError:
                        pass  # 无新消息，继续执行

                if event:
                    self._process_incoming_event(event)
                    self.last_interaction_time = time.time()

                # --- B. 思考与决策阶段 (Thought) ---
                # 更新 System Prompt 时间
                system_prompt = self.prompt_manager.get_system_prompt(self.last_interaction_time)

                # 每次循环更新 System Prompt (如果是每次都 append 会太长，建议替换 history[0])
                # 这里假设 history[0] 永远是 system prompt
                if self.history and self.history[0]["role"] == "system":
                    self.history[0]["content"] = system_prompt
                else:
                    self.history.insert(0, {"role": "system", "content": system_prompt})

                # 调用 LLM
                response_msg = await self._call_llm()

                # --- C. 行动阶段 (Action) ---
                content_str = response_msg.get("content", "")

                # 1. 解析 JSON
                try:
                    parsed_data = json.loads(content_str)
                    thought_content = parsed_data.get("thought", "")
                    # 获取工具列表，默认为空列表
                    tool_calls = parsed_data.get("tool_calls", [])

                except json.JSONDecodeError as e:
                    logger.error(f"❌ 模型输出格式错误: {e}")
                    self.history.append({"role": "assistant", "content": content_str})
                    return

                    # 2. 记录思考
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
                        name = tool_call["name"]
                        args = tool_call["arguments"]

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
                                "content": f"错误: {str(e)}"
                            })
                elif not thought_content:
                    logger.warning("LLM 返回空内容，强制休眠")
                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                await asyncio.sleep(5)  # 出错冷却

    def _process_incoming_event(self, event: OneBotEvent):
        """将外部事件格式化并插入上下文"""
        self.last_interaction_time = time.time()

        # 处理 Wake Up 通知
        if event.type == "notice" and event.detail_type == "wake_up":
            msg = f"系统 [Timer]: {event.message}"
            self.history.append({"role": "user", "content": msg})
            logger.info(f"上下文已更新: {msg} (唤醒)")

        elif event.type == "message":
            self.current_user_id = event.source.user_id
            msg = f"用户 [{event.source.user_id}]: {event.message}"
            self.history.append({"role": "user", "content": msg})
            logger.info(f"上下文已更新: {msg}")

    async def _call_llm(self) -> Dict[str, Any]:
        """封装 API 调用"""
        schemas = self.tool_manager.get_tool_schemas()

        # 定义强制思维 Schema (JSON Schema)
        thought_structure = {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Chain of Thought: Step-by-step reasoning, memory retrieval verification, and plan formulation."
                },
                "tool_calls": {
                    "type": "array",
                    "description": "A list of tools to call.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the function to call."
                            },
                            "arguments": {
                                "type": "object",
                                "description": "The arguments for the function."
                            }
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["thought", "tool_calls"],
            "additionalProperties": False
        }

        # 调用 API，同时传入 tools 和 schema
        response = await self.api_client.create_chat_completion(
            messages=self.history,
            tools=schemas if schemas else None,
            schema=thought_structure
        )

        return response["choices"][0]["message"]
