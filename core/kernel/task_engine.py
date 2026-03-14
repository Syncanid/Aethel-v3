# core/kernel/task_engine.py
import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import aiofiles

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, DetailType, EventType, Action
from core.kernel.task_registry import global_task_registry
from core.limbic.manager import LimbicManager
from core.memory.infinite_context import InfiniteContextManager
from core.tool_manager.aggregator import ToolManager
from core.utilities import calculate_tokens

logger = logging.getLogger(__name__)


class DummySource:
    """用于恢复重启前的 Event Source 对象"""

    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self):
        """伪装成 Pydantic 模型，输出字典供 JSON 序列化"""
        return self.__dict__


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
        self.limbic = LimbicManager(config, database, event_bus, self.api_client)

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

        self.context_manager = InfiniteContextManager(config, self.api_client)
        self.last_response_content = ""

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
        """处理来自 System 1 的实时信息补充"""
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

    async def save_state(self):
        """统一持久化 S2 状态与正在执行的任务进度现场"""
        if not self.is_busy:
            state = {"is_busy": False}
        else:
            source_dict = {}
            if hasattr(self, "_current_task_source") and self._current_task_source:
                # 兼容原生的 Pydantic Model 和恢复出的 DummySource
                if hasattr(self._current_task_source, 'model_dump'):
                    source_dict = self._current_task_source.model_dump()
                elif hasattr(self._current_task_source, '__dict__'):
                    source_dict = self._current_task_source.__dict__
                else:
                    source_dict = dict(self._current_task_source)

            state = {
                "is_busy": self.is_busy,
                "history": self.history,
                "scratchpad": self.scratchpad,
                "last_response_used_tools": self.last_response_used_tools,
                "last_response_content": self.last_response_content,
                "current_task_info": {
                    "task_id": getattr(self, "_current_task_id", ""),
                    "description": getattr(self, "_current_task_desc", ""),
                    "params": getattr(self, "_current_task_params", {}),
                    "source": source_dict
                },
                "timestamp": time.time()
            }

        try:
            json_str = json.dumps(state, ensure_ascii=False)
            async with self.database.get_connection() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO neuro_states (user_id, data_json, last_update) VALUES (?, ?, ?)",
                    ("UNIFIED_S2_STATE", json_str, time.time())
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"S2 状态保存失败: {e}", exc_info=True)

    async def load_state(self) -> bool:
        """恢复 S2 完整执行现场，实现任务断点续传"""
        try:
            async with self.database.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT data_json FROM neuro_states WHERE user_id=?",
                    ("UNIFIED_S2_STATE",)
                )
                row = await cursor.fetchone()
                if row:
                    state = json.loads(row[0])
                    self.is_busy = state.get("is_busy", False)

                    if self.is_busy:
                        self.history = state.get("history", [])
                        self.scratchpad.update(state.get("scratchpad", {}))
                        self.last_response_used_tools = state.get("last_response_used_tools", True)
                        self.last_response_content = state.get("last_response_content", "")

                        info = state.get("current_task_info", {})
                        self._current_task_id = info.get("task_id")
                        self._current_task_desc = info.get("description")
                        self._current_task_params = info.get("params")

                        source_dict = info.get("source", {})
                        self._current_task_source = DummySource(source_dict) if source_dict else None

                        logger.info(f"🔄 发现中断的 S2 任务 [{self._current_task_id}]，准备恢复现场...")
                        return True
        except Exception as e:
            logger.error(f"S2 状态恢复失败: {e}", exc_info=True)
        return False

    def _prune_context(self):
        """[应急防爆] 强制上下文修剪，防止 TaskEngine 内存溢出"""
        TOKEN_LIMIT_APPROX = self.config.get("llm.model_context", 16384)
        SAFE_LIMIT = TOKEN_LIMIT_APPROX - 200

        current_tokens = sum(calculate_tokens(msg.get("content", "")) for msg in self.history)

        while len(self.history) > 6 and current_tokens > SAFE_LIMIT:
            candidate_idx = 1
            msg_to_remove = self.history[candidate_idx]

            # 兼容工具调用链删除
            is_tool_call_msg = (msg_to_remove.get("role") == "assistant" and
                                (msg_to_remove.get("tool_calls") or msg_to_remove.get("function_call")))

            count_to_remove = 1
            if is_tool_call_msg:
                scan_idx = candidate_idx + 1
                while scan_idx < len(self.history) and self.history[scan_idx].get("role") == "tool":
                    count_to_remove += 1
                    scan_idx += 1

            for _ in range(count_to_remove):
                if len(self.history) > 1:
                    removed = self.history.pop(candidate_idx)
                    current_tokens -= calculate_tokens(removed.get("content", ""))

            logger.warning(f"✂️ [System 2 应急防御] Token超出安全限制，已强制切割 {count_to_remove} 条历史消息。")

    async def run_engine_loop(self):
        """
        重循环守护进程，监听任务派发。
        """
        logger.info("⚙️ Task Engine (System 2) 已启动...")
        await self.tool_manager.initialize()

        if await self.load_state():
            self._running_task_coro = asyncio.create_task(
                self._execute_long_loop(
                    self._current_task_id,
                    self._current_task_desc,
                    self._current_task_params,
                    self._current_task_source,
                    is_resume=True  # 标记为断点恢复
                )
            )

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

    async def _execute_long_loop(self, task_id: str, description: str, params: dict, source, is_resume: bool = False):
        """
        [Core Loop] 重循环守护任务。
        """
        self.is_busy = True
        self._current_task_id = task_id
        self._current_task_desc = description
        self._current_task_params = params
        self._current_task_source = source

        final_result = "Unknown"
        try:
            if not is_resume:
                # 1. 全新启动：重置并注册任务
                self.history.clear()
                self.scratchpad.update({
                    "current_task_id": task_id,
                    "current_goal": description,
                    "task_status": "normal",
                    "subtasks": [],
                    "progress_summary": "已接收指令，准备分析执行..."
                })
                self.tool_manager.agent_state = self.scratchpad
                self.last_response_used_tools = True
                await global_task_registry.register_task(task_id, description)

                # 读取基础 prompt
                try:
                    async with aiofiles.open("data/prompts/system2_prompt.md", "r", encoding="utf-8") as f:
                        base_system_prompt = await f.read()
                except FileNotFoundError:
                    base_system_prompt = "You are Aethel's Task Engine."

                self.history.append({"role": "system", "content": base_system_prompt})
                self.history.append({
                    "role": "user",
                    "content": f"【任务派发】\n目标: {description}\n上下文参数: {json.dumps(params, ensure_ascii=False)}\n请通过工具分步执行，并输出最终结论。"
                })

                await self.save_state()  # 保存全新任务的第一帧
            else:
                # 2. 恢复启动：直接沿用缓存中的记录
                self.tool_manager.agent_state = self.scratchpad

                try:
                    async with aiofiles.open("data/prompts/system2_prompt.md", "r", encoding="utf-8") as f:
                        base_system_prompt = await f.read()
                except FileNotFoundError:
                    base_system_prompt = "You are Aethel's Task Engine."

                # 更新 System prompt 防止代码改变
                self.history[0]["content"] = base_system_prompt

                # 告知 Agent 发生了灾难恢复
                self.history.append({
                    "role": "user",
                    "content": "【系统事件】系统刚刚经历了一次重启。你之前执行到一半的任务进度、历史和记忆已被完全恢复。请基于上面的记忆继续执行当前任务。"
                })
                await self.save_state()

            error_count = 0  # 追踪工具报错

            # ================= 长循环开始 =================
            while True:
                # 触发脱水压缩与防爆
                await self.context_manager.compress_if_needed(self.history)
                self._prune_context()

                # --- A. 更新 System Prompt 与状态同步 ---
                scratchpad_dump = json.dumps(self.scratchpad, indent=2, ensure_ascii=False)
                final_system_prompt = (
                    f"{base_system_prompt}\n\n"
                    f"## Scratchpad\n"
                    f"这是你当前内部状态，必须维护和更新：\n{scratchpad_dump}"
                )
                self.history[0]["content"] = final_system_prompt

                # 将目前 scratchpad 里的进度同步给外部全局看版
                current_progress = self.scratchpad.get("progress_summary", f"执行中)")
                await global_task_registry.update_progress(task_id, current_progress)

                try:
                    async with aiofiles.open("data/messages_in_memory_s2.json", "w", encoding="utf-8") as f:
                        await f.write(json.dumps(self.history, ensure_ascii=False, indent=4))

                    async with aiofiles.open("data/prompt_in_memory_s2.txt", "w", encoding="utf-8") as f:
                        await f.write(final_system_prompt)
                except Exception as e:
                    logger.warning(f"Failed to write S2 debug logs: {e}")

                # --- B. 思考 (Call LLM) ---
                response_msg = await self._call_llm()
                await self.limbic.consume_action_energy("complex_reasoning")

                content_str = response_msg.get("content", "")
                if isinstance(content_str, str):
                    content_str = content_str.replace("```json", "").replace("```", "").strip()
                else:
                    content_str = "{}"

                native_tool_calls = response_msg.get("tool_calls", [])

                # 致命死锁检测 (如果完全一致且没调用原生工具，直接打断)
                if content_str and content_str == self.last_response_content and not native_tool_calls:
                    logger.error("⚠️ [System 2] 检测到严重死锁。")
                    self.history.append({
                        "role": "user",
                        "content": "【SYSTEM ERROR - DEADLOCK DETECTED】系统检测到你输出了与上一次完全一致的内容，且未执行任何有效动作！这会导致无限死循环！请立即改变规划思路，如果方法行不通，请调用 `conclude_task` 汇报失败，绝不许死磕！"
                    })
                    self.last_response_content = ""
                    continue # 直接跳入下一轮让模型反思

                self.last_response_content = content_str

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

                        # 提取并广播 S2 的思考过程
                        monologue = parsed_data.get("inner_monologue", {})
                        plan = monologue.get("planning", "正在思考下一步...")
                        self.event_bus.publish_action(Action(
                            action="broadcast_log",
                            params={"content": f"⚙️ [S2 深度思考] {plan}"}
                        ))

                        # 同步 Scratchpad
                        new_scratchpad = parsed_data.get("scratchpad", None)

                        if new_scratchpad and isinstance(new_scratchpad, dict):
                            old_status = self.scratchpad.get("task_status", "normal")

                            self.scratchpad.update(new_scratchpad)
                            self.tool_manager.agent_state = self.scratchpad

                            new_status = self.scratchpad.get("task_status", "normal")

                            if new_status in ["stuck", "frustrated"] and old_status not in ["stuck", "frustrated"]:
                                logger.warning(f"🧠 [System 2] 认知状态变为 {new_status}，触发情绪泄露！")
                                frust_event = OneBotEvent(
                                    type=EventType.NOTICE,
                                    detail_type="internal_frustration",
                                    source=source,
                                    message=f"后台任务遇到死胡同了！当前进度：{self.scratchpad.get('progress_summary')}。这让你感到非常烦躁和挫败！你可以主动向用户发一句牢骚，或者直接向用户求助。",
                                    extra={"task_id": task_id, "frustration_level": 2}
                                )
                                self.event_bus.publish_event(frust_event)

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

                # 3. 执行工具与跳出机制
                is_finished = False

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

                        if name == "conclude_task":
                            final_result = args.get("result", "未提供结论")
                            status = args.get("status", "success")
                            logger.info(f"✅ [System 2] 任务主动宣布结束。结论: {final_result[:50]}...")

                            # 回填历史保证格式完备
                            if t_id:
                                self.history.append({"role": "tool", "tool_call_id": t_id, "name": name,
                                                     "content": f"Task Concluded with status: {status}"})
                            else:
                                self.history.append(
                                    {"role": "tool", "name": name, "content": f"Task Concluded with status: {status}"})

                            is_finished = True
                            break

                        # 正常工具执行
                        try:
                            self.event_bus.publish_action(Action(
                                action="broadcast_log",
                                params={"content": f"🛠️ [S2 任务引擎] 调用工具: {name}({args})"}
                            ))

                            await global_task_registry.update_progress(task_id, f"调用工具: {name}...")
                            result = await self.tool_manager.execute_tool(name, args)
                            error_count = 0  # 成功执行则重置挫败感
                        except Exception as e:
                            result = f"Error: {str(e)}"
                            error_count += 1  # 记录连续失败次数

                            # 失败达到阈值，触发情绪泄露给 S1
                            if error_count >= 2:
                                logger.warning(f"🔧 [System 2] 工具 {name} 连续失败，触发挫败感泄露！")
                                frust_event = OneBotEvent(
                                    type=EventType.NOTICE,
                                    detail_type="internal_frustration",
                                    source=source,
                                    message=f"后台任务遇到大麻烦了！执行 {name} 连续报错：{str(e)[:50]}... 这让你感到非常烦躁和挫败！",
                                    extra={"task_id": task_id, "frustration_level": error_count}
                                )
                                self.event_bus.publish_event(frust_event)

                        if t_id:
                            self.history.append(
                                {"role": "tool", "tool_call_id": t_id, "name": name, "content": str(result)})
                        else:
                            self.history.append({"role": "tool", "name": name, "content": str(result)})
                else:
                    self.last_response_used_tools = False

                await self.save_state()

                if is_finished:
                    await global_task_registry.complete_task(task_id, final_result)

                    # 发送回 EventBus 唤醒 S1
                    complete_event = OneBotEvent(
                        type=EventType.TASK,
                        detail_type=DetailType.TASK_COMPLETE,
                        source=source,
                        extra={"task_payload": {"task_id": task_id, "result": final_result, "description": description}}
                    )
                    self.event_bus.publish_event(complete_event)
                    self.is_busy = False
                    await self.save_state()  # 任务结束，清空繁忙状态
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

            self.tool_manager.unmount_skill_tools()

            self.is_busy = False
            await self.save_state()  # 确保清理现场
            self._running_task_coro = None

    async def _call_llm(self) -> Dict[str, Any]:
        """
        从 agent.py 完全克隆过来的 LLM 强制 Schema 调用。
        去掉了不需要的情绪(emotion_check)，保留了 planning 和 scratchpad。
        """
        tools = self.tool_manager.get_tool_schemas()
        if tools is None:
            tools = []

        tools.append({
            "type": "function",
            "function": {
                "name": "conclude_task",
                "description": "当任务已经得出最终结论，或者确认彻底失败无法继续时，必须调用此工具来结束任务进程。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "任务的最终执行结果、结论或失败原因汇总。此内容将直接作为最终报告。"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure"],
                            "description": "任务最终定性状态。"
                        }
                    },
                    "required": ["result", "status"],
                    "additionalProperties": False
                }
            }
        })

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
                    "task_status": {
                        "type": "string",
                        "enum": ["normal", "working", "stuck", "frustrated", "completed"],
                        "description": "当前任务的认知状态。如果思路受阻、方法无效、或迟迟没有进展，请务必将其修改为 stuck 或 frustrated。"
                    },
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
                "required": ["current_goal", "task_status", "subtasks", "progress_summary"],
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
        return response["choices"][0]["message"]
