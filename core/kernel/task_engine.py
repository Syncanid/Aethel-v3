# core/kernel/task_engine.py
import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional

import aiofiles

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, DetailType, EventType
from core.kernel.task_registry import global_task_registry
from core.tool_manager.aggregator import ToolManager

logger = logging.getLogger(__name__)


class TaskEngine:
    """
    System 2 (理性层/执行引擎)
    由原 AutonomousAgent 阉割而来。
    剥离了：Limbic(情绪), Middleware(中间件), Attention(门控), Hippocampus(海马体长时记忆)。
    保留了：长循环 (Long Loop), Scratchpad (状态维护), ToolManager (工具调用), 本地 System Prompt。
    """

    def __init__(self, config: Config, event_bus: EventBus, database: Database):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.api_client = GenericAPIClient(config)

        # --- 内部状态 (针对当前正在执行的任务) ---
        self.history: List[Dict[str, Any]] = []
        self.scratchpad: Dict[str, Any] = {
            "current_task_id": "",
            "current_goal": "",
            "subtasks": [],
            "progress_summary": "等待分配任务..."
        }
        self.last_response_used_tools = True

        # --- 初始化工具管理器 ---
        self.tool_manager = ToolManager(
            tools_dir="tools/System2",
            config=config,
            event_bus=event_bus,
            api_client=self.api_client,
            database=database,
            agent_state=self.scratchpad
        )

        # 依赖注入
        self.tool_manager.add_dependency("task_engine", self)

        # --- 事件队列与订阅 ---
        self.incoming_events: asyncio.Queue = asyncio.Queue()
        self.event_bus.subscribe_event(self._enqueue_event)

        # 任务控制
        self._running_task_coro: Optional[asyncio.Task] = None
        self.is_busy = False

    async def _enqueue_event(self, event: OneBotEvent):
        """回调：过滤并把 TASK 事件放入队列"""
        # 判断如果是我们要的任务事件才塞入
        if getattr(event, "type", "") == EventType.TASK or getattr(event, "type", "") == "task":
            await self.incoming_events.put(event)

    async def _on_task_update(self, event: OneBotEvent):
        """[新增] 处理来自 System 1 的实时信息补充"""
        payload_data = event.extra.get('task_payload')
        if not payload_data: return

        task_id = payload_data.get('task_id') if isinstance(payload_data, dict) else payload_data.task_id
        info = payload_data.get('description') if isinstance(payload_data, dict) else payload_data.description

        # 如果这个任务恰好正在当前 Engine 中运行
        if self.is_busy and self.scratchpad.get("current_task_id") == task_id:
            logger.info(f"📥 [System 2] 任务 {task_id} 收到实时补充信息: {info}")
            # 强行把用户的新指示塞入重循环的 LLM 历史中
            self.history.append({
                "role": "user",
                "content": f"【SYSTEM INTERRUPT: 用户实时补充信息】\n{info}\n请在后续的规划中考虑此信息。"
            })
            # 同时更新一下 Scratchpad 留档
            self.scratchpad["progress_summary"] = f"收到补充信息，正在重新评估..."

    async def run_engine_loop(self):
        """
        重循环守护进程，监听任务派发。
        """
        logger.info("⚙️ Task Engine (System 2) 已启动...")
        await self.tool_manager.initialize()

        while True:
            try:
                event = await self.incoming_events.get()
                detail_type = getattr(event, "detail_type", "")

                # --- 1. 派发任务 ---
                if detail_type == DetailType.TASK_DISPATCH or detail_type == "task_dispatch":
                    payload = getattr(event, "extra", {}).get("task_payload", {})
                    if not payload:
                        continue

                    if self.is_busy:
                        logger.warning("Task Engine 当前正在执行任务，已忽略新派发请求。")
                        # 实际生产中可以做排队，这里简化为丢弃或发回失败
                        continue

                    task_id = payload.get("task_id")
                    description = payload.get("description")
                    params = payload.get("parameters", {})

                    # 拉起长循环协程
                    self._running_task_coro = asyncio.create_task(
                        self._execute_long_loop(task_id, description, params, event.source)
                    )

                # --- 2. 取消任务 ---
                elif detail_type == DetailType.TASK_CANCEL or detail_type == "task_cancel":
                    payload = getattr(event, "extra", {}).get("task_payload", {})
                    task_id = payload.get("task_id")

                    if self.is_busy and self.scratchpad.get("current_task_id") == task_id:
                        if self._running_task_coro:
                            self._running_task_coro.cancel()
                            logger.info(f"🛑 [System 2] 收到取消指令，任务 {task_id} 已中止。")
                            await global_task_registry.cancel_task(task_id)
                            self.is_busy = False

                # --- 3. 接收 S1 传来的实时信息 ---
                elif detail_type == DetailType.TASK_UPDATE or detail_type == "task_update":
                    payload = getattr(event, "extra", {}).get("task_payload", {})
                    task_id = payload.get("task_id")
                    info = payload.get("description")  # S1 将补充信息塞在这个字段了

                    if self.is_busy and self.scratchpad.get("current_task_id") == task_id:
                        logger.info(f"📥 [System 2] 收到实时补充信息: {info}")
                        # 物理外挂：把用户的新指示塞入重循环的 LLM 历史中
                        self.history.append({
                            "role": "user",
                            "content": f"【SYSTEM INTERRUPT: 用户实时补充信息】\n{info}\n请在后续的规划中考虑此信息。"
                        })
                        self.scratchpad["progress_summary"] = f"收到补充信息，正在重新评估..."
                        await global_task_registry.update_progress(task_id, self.scratchpad["progress_summary"])

            except Exception as e:
                logger.error(f"Task Engine 监听循环异常: {e}", exc_info=True)

    async def _execute_long_loop(self, task_id: str, description: str, params: dict, source):
        """
        [Core Loop] 从原 agent.py 阉割提取的长循环。
        """
        self.is_busy = True
        try:
            # 1. 状态重置
            self.history.clear()
            self.scratchpad.update({
                "current_task_id": task_id,
                "current_goal": description,
                "subtasks": [],
                "progress_summary": "已接收指令，准备分析执行..."
            })
            self.tool_manager.agent_state = self.scratchpad
            self.last_response_used_tools = True
            final_result = "Unknown"

            # 更新全局看板
            await global_task_registry.register_task(task_id, description)

            # 2. 读取本地 System 2 专属 Prompt
            try:
                async with aiofiles.open("data/prompts/system2_prompt.md", "r", encoding="utf-8") as f:
                    base_system_prompt = await f.read()
            except FileNotFoundError:
                base_system_prompt = "You are Aethel's Task Engine. Execute the requested tasks accurately."

            # 3. 注入系统和首次指令
            self.history.append({"role": "system", "content": base_system_prompt})
            self.history.append({
                "role": "user",
                "content": f"【任务派发】\n目标: {description}\n上下文参数: {json.dumps(params, ensure_ascii=False)}\n请通过工具分步执行，并输出最终结论。"
            })

            # ================= 长循环开始 =================
            while True:
                # --- A. 更新 System Prompt 与状态同步 ---
                scratchpad_dump = json.dumps(self.scratchpad, indent=2, ensure_ascii=False)
                final_system_prompt = (
                    f"{base_system_prompt}\n\n"
                    f"## Scratchpad\n"
                    f"这是你当前内部状态，必须维护和更新：\n{scratchpad_dump}"
                )
                self.history["content"] = final_system_prompt

                # 将目前 scratchpad 里的进度同步给外部全局看版
                current_progress = self.scratchpad.get("progress_summary", f"执行中)")
                await global_task_registry.update_progress(task_id, current_progress)

                # --- B. 思考 (Call LLM) ---
                response_msg = await self._call_llm()

                content_str = response_msg.get("content", "")
                if isinstance(content_str, str):
                    content_str = content_str.replace("```json", "").replace("```", "").strip()
                else:
                    content_str = "{}"

                native_tool_calls = response_msg.get("tool_calls", [])

                # 回填历史记录
                if native_tool_calls:
                    msg_entry = response_msg.copy()
                    if "content" not in msg_entry or msg_entry["content"] is None:
                        msg_entry["content"] = ""
                    self.history.append(msg_entry)
                else:
                    self.history.append({"role": "assistant", "content": content_str})

                # --- C. 行动 (Parse & Execute Tools) ---
                tool_execution_queue = []

                # 1. 解析 Schema JSON (获取独白和工具)
                if content_str:
                    try:
                        parsed_data = json.loads(content_str)

                        # 同步 Scratchpad
                        new_scratchpad = parsed_data.get("scratchpad", None)
                        if new_scratchpad and isinstance(new_scratchpad, dict):
                            self.scratchpad.update(new_scratchpad)
                            self.tool_manager.agent_state = self.scratchpad

                        # 提取 Schema Tools
                        schema_calls = parsed_data.get("tool_calls", [])
                        for tc in schema_calls:
                            tool_execution_queue.append({
                                "name": tc.get("name"),
                                "args": tc.get("arguments"),
                                "id": None
                            })
                    except json.JSONDecodeError:
                        if not native_tool_calls:
                            self.history.append({"role": "user", "content": "SYSTEM ERROR: JSON Format Error."})
                            continue

                # 2. 提取 Native Tools
                if native_tool_calls:
                    for tc in native_tool_calls:
                        func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                        t_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        name = func.get("name") if isinstance(func, dict) else getattr(func, "name", "")
                        args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", "{}")

                        tool_execution_queue.append({"name": name, "args": args, "id": t_id})

                # 3. 执行工具
                if tool_execution_queue:
                    self.last_response_used_tools = True
                    for task in tool_execution_queue:
                        name = task["name"]
                        args = task["args"]
                        t_id = task["id"]

                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}

                        try:
                            await global_task_registry.update_progress(task_id, f"调用工具: {name}...")
                            # 调用 aggregator.py 里的 execute_tool
                            result = await self.tool_manager.execute_tool(name, args)
                        except Exception as e:
                            result = f"Error: {str(e)}"

                        if t_id:
                            self.history.append(
                                {"role": "tool", "tool_call_id": t_id, "name": name, "content": str(result)})
                        else:
                            self.history.append({"role": "tool", "name": name, "content": str(result)})
                else:
                    # 4. 没有调用工具，说明任务得出最终结论 -> 跳出循环
                    self.last_response_used_tools = False
                    logger.info(f"✅ [System 2] 任务已结束，结论已产生。")

                    final_result = content_str or "Task completed without detailed text."
                    await global_task_registry.complete_task(task_id, final_result)

                    # 发送回 EventBus 唤醒 S1
                    complete_event = OneBotEvent(
                        type=EventType.TASK,
                        detail_type=DetailType.TASK_COMPLETE,
                        source=source,
                        extra={"task_payload": {"task_id": task_id, "result": final_result, "description": description}}
                    )
                    self.event_bus.publish_event(complete_event)
                    break

        except asyncio.CancelledError:
            pass  # 外部打断
        except Exception as e:
            final_result = f"System Crash: {str(e)}"
            logger.error(f"❌ [System 2] 任务执行崩溃: {e}", exc_info=True)
            await global_task_registry.complete_task(task_id, final_result)
        finally:
            # 任务结束，打包并持久化黑匣子记录
            record_dir = "data/task_records"
            os.makedirs(record_dir, exist_ok=True)

            record_data = {
                "task_id": task_id,
                "goal": description,
                "parameters": params,
                "final_scratchpad": self.scratchpad,
                "final_result": final_result,
                "execution_history": self.history
            }

            try:
                filepath = os.path.join(record_dir, f"{task_id}.json")
                async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(record_data, ensure_ascii=False, indent=2))
                logger.info(f"📦 [System 2] 任务 {task_id} 历史记录已打包归档至: {filepath}")
            except Exception as e:
                logger.error(f"任务归档失败: {e}")

            self.is_busy = False
            self._running_task_coro = None

    async def _call_llm(self) -> Dict[str, Any]:
        """
        从 agent.py 完全克隆过来的 LLM 强制 Schema 调用。
        去掉了不需要的情绪(emotion_check)，保留了 planning 和 scratchpad。
        """
        tools = self.tool_manager.get_tool_schemas()
        use_schema_tools = self.config.get("llm.use_schema_tool_calls", True)
        arg_mode = self.config.get("llm.tool_call_arg_mode", "object")

        properties = {
            "inner_monologue": {
                "type": "object",
                "properties": {
                    "planning": {"type": "string", "description": "接下来的执行和推理计划。"}
                },
                "required": ["planning"],
                "additionalProperties": False
            },
            "scratchpad": {
                "type": "object",
                "properties": {
                    "current_goal": {"type": "string"},
                    "subtasks": {
                        "type": "array",
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
                    "progress_summary": {"type": "string", "description": "精简的一句话进度报告，将被用户看到"}
                },
                "required": ["current_goal", "subtasks", "progress_summary"],
                "additionalProperties": False
            }
        }
        required_fields = ["inner_monologue", "scratchpad"]

        if use_schema_tools:
            arg_schema = {"type": "object"} if arg_mode == "object" else {"type": "string"}
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

        thought_structure = {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False
        }

        # 清洗掉内部 metadata 防止报错
        sanitized_history = [
            {k: v for k, v in d.items() if k != 'metadata'}
            for d in self.history
        ]

        # 调用 GenericAPIClient
        response = await self.api_client.create_chat_completion(
            messages=sanitized_history,
            tools=tools if tools else None,
            schema=thought_structure,
            tool_choice="auto" if use_schema_tools else ("auto" if self.last_response_used_tools else "required")
        )
        return response["choices"]["message"]
