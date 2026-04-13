# core/kernel/task_engine.py
import asyncio
import json
import logging
import os
import platform
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

    async def save_checkpoint(self, is_stable: bool = False):
        """
        统一持久化 S2 状态与正在执行的任务进度现场
        实施基于状态机的快照堆栈。仅在关键节点建立可回滚的 Stable Checkpoint。
        """
        if not self.is_busy:
            return

        task_id = getattr(self, "_current_task_id", "UNKNOWN")

        # 序列化当前现场源
        source_dict = {}
        if hasattr(self, "_current_task_source") and self._current_task_source:
            if hasattr(self._current_task_source, 'model_dump'):
                source_dict = self._current_task_source.model_dump()
            elif hasattr(self._current_task_source, '__dict__'):
                source_dict = self._current_task_source.__dict__
            else:
                source_dict = dict(self._current_task_source)

        state_payload = {
            "history": self.history,
            "scratchpad": self.scratchpad,
            "last_response_used_tools": self.last_response_used_tools,
            "last_response_content": self.last_response_content,
            "current_task_info": {
                "task_id": task_id,
                "description": getattr(self, "_current_task_desc", ""),
                "params": getattr(self, "_current_task_params", {}),
                "source": source_dict
            }
        }

        json_str = json.dumps(state_payload, ensure_ascii=False)
        current_time = time.time()

        try:
            async with self.database.get_connection() as conn:
                # 1. 插入新快照
                await conn.execute(
                    """
                    INSERT INTO s2_checkpoints (task_id, timestamp, is_stable, data_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, current_time, is_stable, json_str)
                )

                # 2. 状态修剪 (Pruning)：为了防止 I/O 与存储爆炸，每个任务最多保留最新的 5 个稳态快照和 1 个最新动态快照
                await conn.execute(
                    """
                    DELETE
                    FROM s2_checkpoints
                    WHERE task_id = ?
                      AND checkpoint_id NOT IN (SELECT checkpoint_id
                                                FROM s2_checkpoints
                                                WHERE task_id = ?
                                                ORDER BY checkpoint_id DESC
                        LIMIT 6
                        )
                    """,
                    (task_id, task_id)
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"S2 快照保存失败 (Task: {task_id}): {e}", exc_info=True)

    async def load_latest_checkpoint(self) -> bool:
        """
        系统灾难恢复时，拉取该任务最新的 Checkpoint
        """
        try:
            async with self.database.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT data_json FROM s2_checkpoints ORDER BY timestamp DESC LIMIT 1"
                )
                row = await cursor.fetchone()
                if row:
                    state = json.loads(row[0])

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

                    self.is_busy = True
                    logger.info(f"🔄 S2 现场已从最新快照恢复 [{self._current_task_id}]")
                    return True
        except Exception as e:
            logger.error(f"S2 快照恢复失败: {e}", exc_info=True)
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

        if await self.load_latest_checkpoint():
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

        self.task_status = {
            "finished": False,
            "final_result": "Unknown",
            "status": "Unknown",
            "skill": False
        }

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

                await self.save_checkpoint(is_stable=True)
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
                    "content": "【系统事件】系统刚刚经历了一次重启或主动回滚。你之前的任务进度、历史和记忆已被全盘恢复。请基于上面的记忆继续执行当前任务。"
                })
                await self.save_checkpoint(is_stable=True)

            # ================= 长循环开始 =================
            while True:
                # 触发脱水压缩与防爆
                await self.context_manager.compress_if_needed(self.history)
                self._prune_context()

                # --- A. 更新 System Prompt 与状态同步 ---
                env_info = self._get_environment_context(task_id)

                scratchpad_dump = json.dumps(self.scratchpad, indent=2, ensure_ascii=False)
                final_system_prompt = (
                    f"{base_system_prompt}\n\n"
                    f"## Environment Context\n"
                    f"你当前处于以下执行环境中：\n"
                    f"{env_info}\n\n"
                    f"## Scratchpad\n"
                    f"这是你当前内部状态，必须维护和更新：\n{scratchpad_dump}"
                )
                self.history[0]["content"] = final_system_prompt

                # 将目前 scratchpad 里的进度同步给外部全局看版
                current_progress = self.scratchpad.get("progress_summary", "执行中...")
                await global_task_registry.update_progress(task_id, current_progress)

                try:
                    async with aiofiles.open("data/messages_in_memory_s2.json", "w", encoding="utf-8") as f:
                        await f.write(json.dumps(self.history, ensure_ascii=False, indent=4))

                    async with aiofiles.open("data/prompt_in_memory_s2.txt", "w", encoding="utf-8") as f:
                        await f.write(final_system_prompt)
                except Exception as e:
                    logger.warning(f"Failed to write S2 debug logs: {e}")

                # --- B. 思考 (Call LLM) ---
                response_data = await self._call_llm()
                await self.limbic.consume_action_energy("complex_reasoning")

                # 解析响应
                parsed_data = response_data.get("content", {})
                tool_queue = response_data.get("tool_calls", [])
                raw_receive = response_data.get("raw_receive", {})

                # 回填历史记录
                if raw_receive.get("tool_calls"):
                    msg_entry = raw_receive.copy()
                    if "content" not in msg_entry or msg_entry["content"] is None:
                        msg_entry["content"] = ""
                    self.history.append(msg_entry)
                else:
                    formatted_content = (json.dumps(parsed_data, ensure_ascii=False)
                                         if isinstance(parsed_data, dict) else str(parsed_data))
                    self.history.append({"role": "assistant", "content": formatted_content})

                # [死锁检测] 将结构化数据序列化后比对
                content_for_deadlock = (json.dumps(parsed_data, sort_keys=True)
                                        if isinstance(parsed_data, dict) else str(parsed_data))
                if content_for_deadlock and content_for_deadlock == self.last_response_content and not tool_queue:
                    logger.warning("⚠️ [System 2] 检测到严重死锁。")
                    self.history.append({
                        "role": "user",
                        "content": "【SYSTEM ERROR - DEADLOCK DETECTED】系统检测到你输出了与上一次完全一致的内容，且未执行任何有效动作！这会导致无限死循环！请立即改变规划思路，如果方法行不通，请调用 `conclude_task` 汇报失败，绝不许死磕！"
                    })
                    continue  # 直接跳入下一轮让模型反思

                self.last_response_content = content_for_deadlock

                # --- C. 行动 (Parse & Execute Tools) ---
                has_valid_scratchpad = False

                if isinstance(parsed_data, dict):
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
                        has_valid_scratchpad = True  # 标记合法解析
                        old_status = self.scratchpad.get("task_status", "normal")

                        self.scratchpad.update(new_scratchpad)
                        self.tool_manager.agent_state = self.scratchpad

                        new_status = self.scratchpad.get("task_status", "normal")

                        # 认知状态恶化时触发情绪泄露
                        if new_status in ["stuck", "frustrated"] and old_status not in ["stuck", "frustrated"]:
                            logger.warning(f"🧠 [System 2] 认知状态恶化为 {new_status}，触发情绪泄露！")
                            frust_event = OneBotEvent(
                                type=EventType.NOTICE,
                                detail_type="internal_frustration",
                                source=source,
                                message=f"后台任务陷入僵局！当前进度：{self.scratchpad.get('progress_summary')}。这让你感到非常受挫！可以向用户发一句牢骚，或者直接求助。",
                                extra={"task_id": task_id, "frustration_level": 3}
                            )
                            self.event_bus.publish_event(frust_event)

                # 执行工具列表
                if tool_queue:
                    self.last_response_used_tools = True
                    for task in tool_queue:
                        name = task["name"]
                        args = task["arguments"]
                        t_id = task["id"]

                        # 正常工具执行
                        try:
                            self.event_bus.publish_action(Action(
                                action="broadcast_log",
                                params={"content": f"🛠️ [S2 任务引擎] 调用工具: {name}({args})"}
                            ))

                            await global_task_registry.update_progress(task_id, f"调用工具: {name}...")

                            # 执行工具，内部包含了重试与防抖
                            result = await self.tool_manager.execute_tool(name, args)

                        except Exception as e:
                            # 这里只捕获 ToolManager 自身的致命崩溃
                            result = f"【SYSTEM CRASH】引擎调用栈致命错误: {str(e)}"
                            logger.error(f"🔧 [System 2] 工具中间件崩溃: {e}")

                        # 将安全的返回结果记录进历史
                        if t_id:
                            self.history.append(
                                {"role": "tool", "tool_call_id": t_id, "name": name, "content": str(result)})
                        else:
                            self.history.append({"role": "tool", "name": name, "content": str(result)})
                else:
                    self.last_response_used_tools = False

                await self.save_checkpoint(is_stable=has_valid_scratchpad)

                if self.task_status["finished"]:
                    break

        except asyncio.CancelledError:
            pass  # 外部打断
        except Exception as e:
            final_result = f"System Crash: {str(e)}"
            logger.error(f"❌ [System 2] 任务执行崩溃: {e}", exc_info=True)
            await global_task_registry.complete_task(task_id, final_result)

            complete_event = OneBotEvent(
                type=EventType.TASK,
                detail_type=DetailType.TASK_COMPLETE,
                source=source,
                extra={"task_payload": {
                    "task_id": task_id,
                    "result": final_result,
                    "status": "failure",
                    "description": description
                }}
            )
            self.event_bus.publish_event(complete_event)
            await self.save_checkpoint(is_stable=False)
        finally:
            # 任务结束，打包并持久化黑匣子记录
            record_dir = "data/task_records"
            os.makedirs(record_dir, exist_ok=True)

            record_data = {
                "task_id": task_id,
                "goal": description,
                "parameters": params,
                "final_scratchpad": self.scratchpad,
                "final_result": self.task_status["final_result"],
                "execution_history": self.history
            }

            try:
                await global_task_registry.complete_task(task_id, self.task_status["final_result"])

                # 发送回 EventBus 唤醒 S1
                complete_event = OneBotEvent(
                    type=EventType.TASK,
                    detail_type=DetailType.TASK_COMPLETE,
                    source=source,
                    extra={"task_payload": {
                        "task_id": task_id,
                        "result": self.task_status["final_result"],
                        "status": self.task_status["status"],
                        "description": description
                    }}
                )
                self.event_bus.publish_event(complete_event)

                filepath = os.path.join(record_dir, f"{task_id}.json")
                async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(record_data, ensure_ascii=False, indent=2))
                logger.info(f"📦 [System 2] 任务 {task_id} 历史记录已打包归档至: {filepath}")

                if self.task_status["status"] == "success" and self.task_status["skill"]:
                    logger.info("🧬 [System 2] 任务成功且要求演化，正在发布内部提取 Action...")
                    self.event_bus.publish_action(Action(
                        action="extract_skill",
                        params={"task_id": task_id},
                        target_platform="internal"
                    ))
            except Exception as e:
                logger.error(f"任务归档失败: {e}")

            from tools.System2.terminal_ops import TERMINAL_SESSIONS
            if task_id in TERMINAL_SESSIONS:
                logger.info(f"🧹 [System 2] 自动清理任务 {task_id} 残留的终端会话。")
                session = TERMINAL_SESSIONS.pop(task_id)
                try:
                    if session["type"] == "ssh":
                        session["process"].terminate()
                        session["conn"].close()
                    else:
                        session["process"].terminate()
                except Exception:
                    pass

            self.tool_manager.unmount_skill_tools()
            self.is_busy = False
            await self.save_checkpoint(is_stable=False)
            self._running_task_coro = None

    async def rollback_to_last_stable(self, new_reason: str) -> str:
        """
        主动状态回滚机制。
        寻找当前任务倒数第一个或第二个稳定快照，并覆盖当前现场。
        在覆盖当前污染历史前，提取所有已知的失败教训，防止模型在同一节点无限震荡。
        """
        task_id = getattr(self, "_current_task_id", "UNKNOWN")

        # 1. 在当前被污染的 history 被销毁前，打捞该节点上所有既往的“回滚记录”
        previous_lessons = []
        for msg in self.history:
            if msg.get("role") == "tool" and msg.get("name") == "revert_to_checkpoint":
                # 提取历史中已经积累的失败警告
                content = msg.get("content", "")
                if "失败原因：" in content:
                    extracted = content.split("失败原因：")[-1].split("\n")[0].strip()
                    previous_lessons.append(extracted)

        try:
            async with self.database.get_connection() as conn:
                # 倒序查找最近的 stable 快照。由于最新的 stable 可能就是刚触发错误的上一刻，
                # 为了确保真正退回到分岔路口，通常需要提取 LIMIT 1
                cursor = await conn.execute(
                    """
                    SELECT data_json
                    FROM s2_checkpoints
                    WHERE task_id = ?
                      AND is_stable = 1
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (task_id,)
                )
                row = await cursor.fetchone()

                if row:
                    state = json.loads(row[0])

                    # 2. 物理覆盖：销毁当前污染时间线
                    self.history = state.get("history", [])
                    self.scratchpad.update(state.get("scratchpad", {}))
                    self.last_response_used_tools = False

                    # 3. 合并新旧教训，构建“记忆残留”黑匣子
                    all_lessons = previous_lessons + [new_reason]

                    lessons_text = "\n".join([f"- 尝试 {i + 1} 失败原因：{r}" for i, r in enumerate(all_lessons)])

                    # 构造强制注入的 Prompt 返回值
                    combined_warning = (
                        f"【SYSTEM INTERVENTION - TIMELINE REVERTED】\n"
                        f"系统已将状态回滚到发生错误之前的安全节点。\n"
                        f"⚠️ 注意：在此分岔口，你已经在平行的废弃时间线中遭遇了 {len(all_lessons)} 次严重失败，教训如下：\n"
                        f"{lessons_text}\n\n"
                        f"最高指令：在接下来的 planning 中，你【必须彻底放弃】上述所有尝试过的思路！如果所有可能路径均已封死，请立即调用 ask_system1_for_help 或 conclude_task(status='failure')，严禁再次重试上述逻辑！"
                    )
                    logger.warning(f"⏪ [System 2] 认知回退完成，携带 {len(all_lessons)} 条残存记忆。")
                    return combined_warning
                else:
                    return "【SYSTEM ERROR】无稳定状态可供回滚。"
        except Exception as e:
            logger.error(f"S2 回滚异常: {e}", exc_info=True)
            return f"【SYSTEM CRASH】回滚过程崩溃: {str(e)}"

    def _get_environment_context(self, task_id: str) -> str:
        """
        [感知引擎] 实时提取当前任务挂载的终端环境元数据。
        """
        native_os = platform.system()
        native_node = platform.node()

        # 默认基础信息
        context = [
            f"- 宿主系统: {native_os} ({os.name})",
            f"- 宿主节点: {native_node}"
        ]

        # 检查是否有活跃的终端会话
        from tools.System2.terminal_ops import TERMINAL_SESSIONS
        session = TERMINAL_SESSIONS.get(task_id)
        if not session:
            shell_type = "PowerShell" if native_os == "Windows" else "Bash"
            context.append(f"- 当前终端: 未初始化 (将默认使用 {shell_type})")
        else:
            s_type = session.get("type", "unknown")
            if s_type == "ssh":
                conn = session.get("conn")
                # 提取 SSH 连接对象信息
                peer_info = f"{conn._username}@{conn._host}:{conn._port}" if conn else "Unknown SSH"
                context.append(f"- 执行模式: **REMOTE SSH SESSION**")
                context.append(f"- 连接对象: `{peer_info}`")
                context.append("- 语法约束: 必须使用目标机器的 Shell 语法 (通常为 Bash)")
            else:
                shell_type = "PowerShell" if native_os == "Windows" else "Bash"
                context.append(f"- 执行模式: **LOCAL TERMINAL**")
                context.append(f"- 活跃 Shell: {shell_type}")
                if native_os == "Windows":
                    context.append("- 警告: 当前为 PowerShell 环境，禁用 Bash 专属操作符，复杂命令请用 `cmd /c` 桥接。")

        return "\n".join(context)

    async def _call_llm(self) -> Dict[str, Any]:
        """
        S2 LLM 调用
        """
        tools = self.tool_manager.get_tool_schemas()
        if tools is None:
            tools = []

        thought_structure = {
            "type": "object",
            "properties": {
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
                            "description": "当前任务的认知状态。如果思路受阻、方法无效、或迟迟没有进展，请将其修改为 stuck 或 frustrated。"
                        },
                        "path_viability": {
                            "type": "string",
                            "enum": ["high", "medium", "dead_end"],
                            "description": "评估当前执行路径的可行性。如果连续遇到环境报错、找不到元素或逻辑不通，必须将其设为 dead_end。"
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
                        "progress_summary": {"type": "string", "description": "精简的一句话进度报告，将被用户看到"},
                    },
                    "required": ["current_goal", "path_viability", "task_status", "subtasks", "progress_summary"],
                    "additionalProperties": False
                }
            },
            "required": ["inner_monologue", "scratchpad"],
            "additionalProperties": False
        }

        # 清洗掉内部 metadata 防止报错
        sanitized_history = [
            {k: v for k, v in d.items() if k != 'metadata'}
            for d in self.history
        ]

        # 调用 GenericAPIClient
        return await self.api_client.create_chat_completion(
            messages=sanitized_history,
            tools=tools if tools else None,
            schema=thought_structure,
            require_tools=not self.last_response_used_tools
        )

    @property
    def current_task_id(self):
        return self._current_task_id
