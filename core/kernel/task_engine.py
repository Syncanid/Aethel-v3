# core/kernel/task_engine.py
import asyncio
import copy
import json
import logging
import os
import platform
import random
import time
from typing import List, Dict, Any

import aiofiles

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, DetailType, EventType, Action, EventSource
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
    System 2 (理性层/执行引擎) - 并发调度器版本
    职责：监听事件总线，为每个独立任务拉起沙盒协程，提供无状态的服务函数。
    """

    def __init__(self, config: Config, event_bus: EventBus, database: Database):
        self.config = config
        self.event_bus = event_bus
        self.database = database
        self.api_client = GenericAPIClient(config)
        self.limbic = LimbicManager(config, database, event_bus, self.api_client)

        self.context_manager = InfiniteContextManager(config, self.api_client)

        # 事件队列与订阅
        self.incoming_events = None
        self.event_bus.subscribe_event(self._enqueue_event)

        # 并发任务追踪池
        self.active_tasks: Dict[str, asyncio.Task] = {}
        # 实时消息信箱 (用于 S1 强行向运行中的 S2 注入指令)
        self.task_mailboxes: Dict[str, asyncio.Queue] = {}

    async def _enqueue_event(self, event: OneBotEvent):
        """回调：过滤并把 TASK 事件放入队列"""
        # 判断如果是我们要的任务事件才塞入
        if getattr(event, "type", "") == EventType.TASK or getattr(event, "type", "") == "task":
            # 惰性初始化，确保 Queue 绝对绑定在当前激活的事件循环上
            if self.incoming_events is None:
                self.incoming_events = asyncio.Queue()
            await self.incoming_events.put(event)

    async def save_checkpoint(self, task_id: str, history: list, scratchpad: dict, params: dict, source_dict: dict,
                              last_response_used_tools: bool, last_response_content: str, description: str,
                              is_stable: bool = False):
        """无状态持久化：保存特定任务现场快照"""
        state_payload = {
            "history": history,
            "scratchpad": scratchpad,
            "last_response_used_tools": last_response_used_tools,
            "last_response_content": last_response_content,
            "current_task_info": {
                "task_id": task_id,
                "description": description,
                "params": params,
                "source": source_dict
            }
        }

        json_str = json.dumps(state_payload, ensure_ascii=False)
        current_time = time.time()

        try:
            async with self.database.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO s2_checkpoints (task_id, timestamp, is_stable, data_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, current_time, is_stable, json_str)
                )

                # 状态修剪，保留最新6个
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

    def _prune_context(self, history: list):
        """[应急防爆] 强制上下文修剪"""
        TOKEN_LIMIT_APPROX = self.config.get("llm.model_context", 16384)
        SAFE_LIMIT = TOKEN_LIMIT_APPROX - 1000

        current_tokens = sum(calculate_tokens(msg.get("content", "")) for msg in history)

        while len(history) > 6 and current_tokens > SAFE_LIMIT:
            candidate_idx = 1
            msg_to_remove = history[candidate_idx]

            # 兼容工具调用链删除
            is_tool_call_msg = (msg_to_remove.get("role") == "assistant" and
                                (msg_to_remove.get("tool_calls") or msg_to_remove.get("function_call")))

            count_to_remove = 1
            if is_tool_call_msg:
                scan_idx = candidate_idx + 1
                while scan_idx < len(history) and history[scan_idx].get("role") == "tool":
                    count_to_remove += 1
                    scan_idx += 1

            for _ in range(count_to_remove):
                if len(history) > 1:
                    removed = history.pop(candidate_idx)
                    current_tokens -= calculate_tokens(removed.get("content", ""))

            logger.warning(f"✂️ [System 2 应急防御] Token超出安全限制，已强制切割 {count_to_remove} 条历史消息。")

    async def run_engine_loop(self):
        """
        重循环并发调度引擎
        """
        logger.info("⚙️ Task Engine (System 2) 已启动...")

        if self.incoming_events is None:
            self.incoming_events = asyncio.Queue()

        await self._recover_suspended_tasks()

        while True:
            try:
                event = await self.incoming_events.get()
                detail_type = getattr(event, "detail_type", "")

                # 1. 派发新任务
                if detail_type in [DetailType.TASK_DISPATCH, "task_dispatch"]:
                    payload = getattr(event, "extra", {}).get("task_payload", {})
                    if not payload:
                        continue

                    task_id = payload.get("task_id")
                    description = payload.get("description")
                    params = copy.deepcopy(payload.get("parameters", {}))

                    if task_id in self.active_tasks:
                        logger.warning(f"任务 {task_id} 已经在调度池中，拒绝重复拉起。")
                        continue

                    # 为任务初始化信箱
                    self.task_mailboxes[task_id] = asyncio.Queue()

                    # 拉起独立沙盒协程
                    task_coro = asyncio.create_task(
                        self._execute_long_loop(task_id, description, params, event.source)
                    )
                    self.active_tasks[task_id] = task_coro

                    # 协程结束后自动清理追踪与信箱
                    def _cleanup(t, tid=task_id):
                        self.active_tasks.pop(tid, None)
                        self.task_mailboxes.pop(tid, None)

                    task_coro.add_done_callback(_cleanup)

                # 2. 强行中止任务
                elif detail_type in [DetailType.TASK_CANCEL, "task_cancel"]:
                    payload = getattr(event, "extra", {}).get("task_payload", {})
                    task_id = payload.get("task_id")

                    if task_id in self.active_tasks:
                        logger.info(f"🛑 [System 2] 收到取消指令，强行中止工作协程 {task_id}。")
                        self.active_tasks[task_id].cancel()
                        await global_task_registry.cancel_task(task_id)

                # 3. S1 注入实时信息
                elif detail_type in [DetailType.TASK_UPDATE, "task_update"]:
                    payload = getattr(event, "extra", {}).get("task_payload", {})
                    task_id = payload.get("task_id")
                    info = payload.get("description")

                    if task_id in self.active_tasks and task_id in self.task_mailboxes:
                        logger.info(f"📥 [System 2] 向任务 {task_id} 沙盒投递实时补充信息...")
                        await self.task_mailboxes[task_id].put(info)

            except RuntimeError as e:
                if "different event loop" in str(e) or "Event loop is closed" in str(e):
                    logger.warning("🛑 [System 2] 检测到事件循环关闭，调度器主动退出。")
                    break
                logger.error(f"Task Engine 运行时异常: {e}", exc_info=True)
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("🛑 [System 2] 收到系统取消信号，关闭调度器...")
                for t in self.active_tasks.values():
                    t.cancel()
                break

            except Exception as e:
                logger.error(f"Task Engine 监听循环异常: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _execute_long_loop(self, task_id: str, description: str, params: dict, source, is_resume: bool = False):
        """
        [隔离沙盒] 每个任务独立的执行环境，完全切断状态共享。
        """
        # ================= 沙盒状态初始化 =================
        local_history: List[Dict[str, Any]] = []
        local_scratchpad: Dict[str, Any] = {
            "current_task_id": task_id,
            "current_goal": description,
            "task_status": "normal",
            "subtasks": [],
            "progress_summary": "已接收指令，准备分析执行..."
        }
        local_last_response_content = ""
        local_last_response_used_tools = True

        task_status_tracker = {
            "finished": False,
            "final_result": "Unknown",
            "status": "Unknown",
            "skill": False
        }

        source_dict = {}

        if is_resume:
            logger.info(f"🔄 [Sandbox:{task_id}] 正在从 Checkpoint 墓碑中提取记忆...")
            try:
                async with self.database.get_connection() as conn:
                    # 获取时间线最末端的快照
                    cursor = await conn.execute(
                        "SELECT data_json FROM s2_checkpoints WHERE task_id = ? ORDER BY timestamp DESC LIMIT 1",
                        (task_id,)
                    )
                    row = await cursor.fetchone()
                    if not row:
                        raise ValueError("找不到对应的 Checkpoint 物理数据。")

                    state_payload = json.loads(row[0])

                    # 1. 严格覆盖局部变量
                    local_history = state_payload.get("history", [])
                    local_scratchpad = state_payload.get("scratchpad", {})
                    local_last_response_used_tools = state_payload.get("last_response_used_tools", True)
                    local_last_response_content = state_payload.get("last_response_content", "")

                    # 2. 恢复宏观元数据
                    task_info = state_payload.get("current_task_info", {})
                    description = task_info.get("description", "恢复的任务")
                    params = task_info.get("params", {})
                    source_dict = task_info.get("source", {})
                    origin_session = params.get("origin_session")

                    if origin_session:
                        # 确保重启后，S1 依然能收到结果
                        await global_task_registry.subscribe_session(task_id, origin_session)

                    # 3. 反序列化 EventSource
                    source = DummySource(source_dict) if source_dict else None

                    # 4. 认知矫正：注入系统级强制提示，防止模型出现幻觉
                    local_history.append({
                        "role": "user",
                        "content": (
                            "【SYSTEM RECOVERY: 灾难恢复协议已激活】\n"
                            "严重警告：系统刚才发生了物理级的崩溃或被强制重启！\n"
                            "你当前的大脑状态是基于系统崩溃前最后一秒的快照恢复的。\n"
                            "请立即阅读你的 Scratchpad 确认之前的目标，并审查 history 确认你最后一步执行了什么操作。\n"
                            "如果你当时正在执行耗时工具，它大概率已经被打断了，请根据现状自行决定是重新调用还是继续后续计划。"
                        )
                    })
                    logger.info(f"✅ [Sandbox:{task_id}] 现场复水完毕，恢复目标: {description}")

            except Exception as e:
                logger.error(f"❌ [Sandbox:{task_id}] 快照损坏或恢复失败，放弃复活: {e}")
                # 销毁损坏的 Checkpoint 防止无限重启死循环
                async with self.database.get_connection() as conn:
                    await conn.execute("DELETE FROM s2_checkpoints WHERE task_id = ?", (task_id,))
                    await conn.commit()
                return  # 安全退出协程
        else:
            # 全新任务的初始化逻辑 (阶段一中的逻辑)
            local_scratchpad = {
                "current_task_id": task_id,
                "current_goal": description,
                "task_status": "normal",
                "subtasks": [],
                "progress_summary": "已接收指令，准备分析执行..."
            }
            if source:
                if hasattr(source, 'model_dump'):
                    source_dict = source.model_dump()
                elif hasattr(source, '__dict__'):
                    source_dict = source.__dict__
                else:
                    source_dict = dict(source)

            await global_task_registry.register_task(task_id, description)

        # 实例化完全独立的 ToolManager 并挂载局部状态
        local_tool_manager = ToolManager(
            tools_dir="tools/System2",
            config=self.config,
            event_bus=self.event_bus,
            api_client=self.api_client,
            database=self.database,
            agent_state=local_scratchpad
        )
        local_tool_manager.add_dependency("task_engine", self)

        # 从引擎外层提取并向沙盒注入全局依赖
        if hasattr(self, "daemon_manager"):
            local_tool_manager.add_dependency("daemon_manager", self.daemon_manager)

        await local_tool_manager.initialize()

        safe_task_id = str(task_id).replace(":", "_").replace("-", "")
        os.makedirs("data/s2_debug", exist_ok=True)

        try:
            if not is_resume:
                await global_task_registry.register_task(task_id, description)
                origin_session = params.get("origin_session")
                if origin_session:
                    await global_task_registry.subscribe_session(task_id, origin_session)

                # 读取基础 prompt
                try:
                    async with aiofiles.open("data/prompts/system2_prompt.md", "r", encoding="utf-8") as f:
                        base_system_prompt = await f.read()
                except FileNotFoundError:
                    base_system_prompt = "You are Aethel's Task Engine."

                local_history.append({"role": "system", "content": base_system_prompt})
                local_history.append({
                    "role": "user",
                    "content": f"【任务派发】\n目标: {description}\n上下文参数: {json.dumps(params, ensure_ascii=False)}\n请通过工具分步执行，并输出最终结论。"
                })

                await self.save_checkpoint(task_id, local_history, local_scratchpad, params, source_dict,
                                           local_last_response_used_tools, local_last_response_content, description,
                                           is_stable=True)
            else:
                # 预留给灾难恢复加载状态的位置
                pass

            # ================= 长循环开始 =================
            while True:
                # 检查信箱并注入实时中断信息
                mailbox = self.task_mailboxes.get(task_id)
                while mailbox and not mailbox.empty():
                    interrupt_info = mailbox.get_nowait()
                    logger.info(f"📥 [Sandbox:{task_id}] 消费补充信息: {interrupt_info}")
                    local_history.append({
                        "role": "user",
                        "content": f"【SYSTEM INTERRUPT: 用户实时补充信息】\n{interrupt_info}\n请在后续的规划中考虑此信息。"
                    })
                    local_scratchpad["progress_summary"] = f"收到补充信息，重新评估中..."

                # 脱水压缩与防爆
                await self.context_manager.compress_if_needed(local_history)
                self._prune_context(local_history)

                # --- A. 更新 System Prompt 与状态同步 ---
                env_info = self._get_environment_context(task_id)
                scratchpad_dump = json.dumps(local_scratchpad, indent=2, ensure_ascii=False)

                # 动态获取 Prompt 避免文件更改未生效
                try:
                    async with aiofiles.open("data/prompts/system2_prompt.md", "r", encoding="utf-8") as f:
                        base_system_prompt = await f.read()
                except FileNotFoundError:
                    base_system_prompt = "You are Aethel's Task Engine."

                final_system_prompt = (
                    f"{base_system_prompt}\n\n"
                    f"## Environment Context\n"
                    f"你当前处于以下执行环境中：\n{env_info}\n\n"
                    f"## Scratchpad\n"
                    f"这是你当前内部状态，必须维护和更新：\n{scratchpad_dump}"
                )
                local_history[0]["content"] = final_system_prompt

                # 将目前 scratchpad 里的进度同步给外部全局看版
                current_progress = local_scratchpad.get("progress_summary", "执行中...")
                await global_task_registry.update_progress(task_id, current_progress)

                try:
                    async with aiofiles.open(f"data/s2_debug/history_{safe_task_id}.json", "w", encoding="utf-8") as f:
                        await f.write(json.dumps(local_history, ensure_ascii=False, indent=4))
                    async with aiofiles.open(f"data/s2_debug/prompt_{safe_task_id}.txt", "w", encoding="utf-8") as f:
                        await f.write(final_system_prompt)
                except Exception as e:
                    logger.warning(f"Failed to write S2 sandbox logs for {task_id}: {e}")

                # --- B. 思考 (Call LLM) ---
                response_data = await self._call_llm(local_history, local_tool_manager,
                                                     not local_last_response_used_tools)
                await self.limbic.consume_action_energy("complex_reasoning")

                # 解析响应
                parsed_data = response_data.get("content", {})
                tool_queue = response_data.get("tool_calls", [])
                raw_receive = response_data.get("raw_receive", {})

                # 回填记录
                if raw_receive.get("tool_calls"):
                    msg_entry = raw_receive.copy()
                    if "content" not in msg_entry or msg_entry["content"] is None:
                        msg_entry["content"] = ""
                    local_history.append(msg_entry)
                else:
                    formatted_content = json.dumps(parsed_data, ensure_ascii=False) \
                        if isinstance(parsed_data, dict) else str(parsed_data)
                    local_history.append({"role": "assistant", "content": formatted_content})

                # [死锁检测]
                content_for_deadlock = json.dumps(parsed_data, sort_keys=True) \
                    if isinstance(parsed_data, dict) else str(parsed_data)
                if content_for_deadlock and content_for_deadlock == local_last_response_content and not tool_queue:
                    logger.warning(f"⚠️ [Sandbox:{task_id}] 检测到死锁。")
                    local_history.append({
                        "role": "user",
                        "content": "【SYSTEM ERROR - DEADLOCK DETECTED】系统检测到你输出了与上一次完全一致的内容！请立即改变规划思路，如果方法行不通，请调用 conclude_task 汇报失败，绝不许死磕！"
                    })
                    continue

                local_last_response_content = content_for_deadlock
                has_valid_scratchpad = False

                if isinstance(parsed_data, dict):
                    # 提取并广播 S2 的思考过程
                    monologue = parsed_data.get("inner_monologue", {})
                    plan = monologue.get("planning", "正在思考下一步...")
                    self.event_bus.publish_action(Action(
                        action="broadcast_log",
                        params={"content": f"⚙️ [S2 深度思考:{task_id}] {plan}"}
                    ))

                    # 同步 Scratchpad
                    new_scratchpad = parsed_data.get("scratchpad", None)
                    if new_scratchpad and isinstance(new_scratchpad, dict):
                        has_valid_scratchpad = True
                        old_status = local_scratchpad.get("task_status", "normal")
                        local_scratchpad.update(new_scratchpad)

                        if local_scratchpad.get("task_status") in ["stuck", "frustrated"] and old_status not in [
                            "stuck", "frustrated"]:
                            logger.warning(f"🧠 [Sandbox:{task_id}] 认知恶化，触发情绪泄露！")
                            self.event_bus.publish_event(OneBotEvent(
                                type=EventType.NOTICE,
                                detail_type="internal_frustration",
                                source=source,
                                message=f"后台任务陷入僵局！当前进度：{local_scratchpad.get('progress_summary')}。这让你感到非常受挫！",
                                extra={"task_id": task_id, "frustration_level": 3}
                            ))

                # --- C. 行动 (Execute Tools) ---
                if tool_queue:
                    local_last_response_used_tools = True
                    for task_op in tool_queue:
                        name = task_op["name"]
                        args = task_op["arguments"]
                        t_id = task_op["id"]

                        # 正常工具执行
                        try:
                            self.event_bus.publish_action(Action(
                                action="broadcast_log",
                                params={"content": f"🛠️ [S2 任务引擎] 调用工具: {name}({args})"}
                            ))
                            await global_task_registry.update_progress(task_id, f"调用工具: {name}...")
                            result = await local_tool_manager.execute_tool(name, args)
                        except Exception as e:
                            # 这里只捕获 ToolManager 自身的致命崩溃
                            result = f"【SYSTEM CRASH】引擎调用栈致命错误: {str(e)}"
                            logger.error(f"🔧 [Sandbox:{task_id}] 工具中间件崩溃: {e}")

                        # 将安全的返回结果记录进历史
                        if t_id:
                            local_history.append(
                                {"role": "tool", "tool_call_id": t_id, "name": name, "content": str(result)})
                        else:
                            local_history.append({"role": "tool", "name": name, "content": str(result)})
                else:
                    local_last_response_used_tools = False

                await self.save_checkpoint(task_id, local_history, local_scratchpad, params, source_dict,
                                           local_last_response_used_tools, local_last_response_content, description,
                                           is_stable=has_valid_scratchpad)

                # ==========================================
                # 任务结束条件检测机制
                # ==========================================
                is_concluded = False

                # 1. 主路拦截：检测是否显式调用了 conclude_task 工具
                if tool_queue:
                    for task_op in tool_queue:
                        if task_op["name"] == "conclude_task":
                            is_concluded = True
                            try:
                                # 尝试从工具调用的 arguments 中提取最终状态和总结
                                args_data = task_op["arguments"]
                                args_dict = json.loads(args_data) if isinstance(args_data, str) else args_data

                                task_status_tracker["status"] = args_dict.get("status", "success")
                                task_status_tracker["final_result"] = args_dict.get("result", "任务通过工具完结，但未提供总结。")
                                task_status_tracker["skill"] = args_dict.get("generate_skill", False)
                            except Exception as e:
                                logger.warning(f"⚠️ [Sandbox:{task_id}] 解析 conclude_task 参数失败: {e}")
                                task_status_tracker["status"] = "success"
                                task_status_tracker["final_result"] = "任务完结 (参数提取降级)。"
                                task_status_tracker["skill"] = False

                            task_status_tracker["finished"] = True
                            break  # 只要调用了完结工具，立刻停止遍历后续工具

                # 2. 认知兜底：如果模型未调用工具，但暂存板状态已变更为终态，且没有挂起其他工具
                if not is_concluded and not tool_queue:
                    current_status = local_scratchpad.get("task_status", "normal")
                    if current_status == "completed":
                        logger.warning(f"⚠️ [Sandbox:{task_id}] 模型未调用 conclude_task，但暂存板已标记为 {current_status}，触发兜底退出。")
                        is_concluded = True
                        task_status_tracker["finished"] = True
                        task_status_tracker["status"] = "success"
                        task_status_tracker["final_result"] = local_scratchpad.get("progress_summary", f"触发 {current_status} 兜底完结。")
                        task_status_tracker["skill"] = False

                # 如果命中任一完结条件，打破沙盒主循环
                if is_concluded:
                    logger.info(f"🏁 [Sandbox:{task_id}] 长循环正常终止。最终状态: {task_status_tracker['status']}")
                    break

        except asyncio.CancelledError:
            task_status_tracker["status"] = "cancelled"
            logger.info(f"🛑 [Sandbox:{task_id}] 协程已被强行中止。")
        except Exception as e:
            final_result = f"System Crash: {str(e)}"
            logger.error(f"❌ [Sandbox:{task_id}] 任务崩溃: {e}", exc_info=True)
            task_status_tracker.update({"finished": True, "final_result": final_result, "status": "failure"})

            await global_task_registry.complete_task(task_id, final_result)
            await self._broadcast_task_event(task_id, DetailType.TASK_COMPLETE, final_result, "failure", description)
        finally:
            # 安全归档黑匣子与环境清理
            record_dir = "data/task_records"
            os.makedirs(record_dir, exist_ok=True)
            record_data = {
                "task_id": task_id,
                "goal": description,
                "parameters": params,
                "final_scratchpad": local_scratchpad,
                "final_result": task_status_tracker["final_result"],
                "execution_history": local_history
            }

            try:
                if task_status_tracker["status"] not in ["failure", "cancelled"]:
                    await global_task_registry.complete_task(task_id, task_status_tracker["final_result"])
                    await self._broadcast_task_event(task_id, DetailType.TASK_COMPLETE,
                                                     task_status_tracker["final_result"], task_status_tracker["status"],
                                                     description)

                async with aiofiles.open(os.path.join(record_dir, f"{safe_task_id}.json"), "w", encoding="utf-8") as f:
                    await f.write(json.dumps(record_data, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.error(f"任务归档失败: {e}")

            from tools.System2.terminal_ops import TERMINAL_SESSIONS
            if task_id in TERMINAL_SESSIONS:
                logger.info(f"🧹 [Sandbox:{task_id}] 清理终端会话。")
                session = TERMINAL_SESSIONS.pop(task_id)
                try:
                    session["process"].terminate()
                    if session["type"] == "ssh": session["conn"].close()
                except Exception:
                    pass

            async with self.database.get_connection() as conn:
                await conn.execute("DELETE FROM s2_checkpoints WHERE task_id = ?", (task_id,))
                await conn.commit()

            local_tool_manager.unmount_skill_tools()

    async def _recover_suspended_tasks(self):
        """
        [灾难恢复核心] 扫描底层事务日志，并发拉起所有遭遇物理中断的沙盒。
        """
        logger.info("🔍 [System 2] 正在校验 Checkpoint 事务日志，启动灾难恢复扫描...")
        try:
            async with self.database.get_connection() as conn:
                cursor = await conn.execute("SELECT DISTINCT task_id FROM s2_checkpoints")
                rows = await cursor.fetchall()
                suspended_tasks = [r[0] for r in rows]

            if not suspended_tasks:
                logger.info("✅ [System 2] 事务日志干净，未发现悬空任务。")
                return

            logger.warning(f"⚠️ [System 2] 发现 {len(suspended_tasks)} 个意外断电/崩溃的悬空任务！正在执行并发复活...")

            for task_id in suspended_tasks:
                if task_id in self.active_tasks:
                    continue

                # 为复活的任务重新建立异步信箱
                self.task_mailboxes[task_id] = asyncio.Queue()

                # 拉起沙盒协程，激活 is_resume=True 标记
                # 初始参数置空，具体上下文由沙盒在内部从数据库反序列化
                task_coro = asyncio.create_task(
                    self._execute_long_loop(task_id, "", {}, None, is_resume=True)
                )
                self.active_tasks[task_id] = task_coro

                def _cleanup(t, tid=task_id):
                    self.active_tasks.pop(tid, None)
                    self.task_mailboxes.pop(tid, None)

                task_coro.add_done_callback(_cleanup)

                # 错峰拉起，防止瞬间 I/O 爆炸或 LLM 并发限流
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"S2 灾难恢复引擎致命异常: {e}", exc_info=True)

    async def rollback_to_last_stable(self, task_id: str, local_history: list, local_scratchpad: dict,
                                      new_reason: str) -> str:
        """从隔离库中执行单任务回滚逻辑，并将修正信息注入传入的 local_history"""
        previous_lessons = []
        for msg in local_history:
            if msg.get("role") == "tool" and msg.get("name") == "revert_to_checkpoint":
                # 提取历史中已经积累的失败警告
                content = msg.get("content", "")
                if "失败原因：" in content:
                    previous_lessons.append(content.split("失败原因：")[-1].split("\n")[0].strip())

        try:
            async with self.database.get_connection() as conn:
                # 倒序查找最近的 stable 快照。由于最新的 stable 可能就是刚触发错误的上一刻，
                # 为了确保真正退回到分岔路口，通常需要提取 LIMIT 1
                cursor = await conn.execute(
                    "SELECT data_json FROM s2_checkpoints WHERE task_id = ? AND is_stable = 1 ORDER BY timestamp DESC LIMIT 1",
                    (task_id,)
                )
                row = await cursor.fetchone()

                if row:
                    state = json.loads(row[0])
                    local_history.clear()
                    local_history.extend(state.get("history", []))
                    local_scratchpad.clear()
                    local_scratchpad.update(state.get("scratchpad", {}))

                    all_lessons = previous_lessons + [new_reason]
                    lessons_text = "\n".join([f"- 尝试 {i + 1} 失败原因：{r}" for i, r in enumerate(all_lessons)])

                    logger.warning(f"⏪ [Sandbox:{task_id}] 认知回退完成，携带 {len(all_lessons)} 条残存记忆。")
                    return (f"【SYSTEM INTERVENTION - TIMELINE REVERTED】\n系统已将状态回滚到发生错误之前的安全节点。\n"
                            f"⚠️ 注意：在此分岔口，你已经在平行的废弃时间线中遭遇了 {len(all_lessons)} 次严重失败，教训如下：\n{lessons_text}\n\n"
                            f"最高指令：放弃上述思路！如果死胡同请调用 ask_system1_for_help 或 conclude_task。")
                return "【SYSTEM ERROR】无稳定状态可供回滚。"
        except Exception as e:
            logger.error(f"S2 回滚异常: {e}", exc_info=True)
            return f"【SYSTEM CRASH】回滚过程崩溃: {str(e)}"

    def _get_environment_context(self, task_id: str) -> str:
        """获取任务沙盒特定的环境元数据"""
        native_os = platform.system()
        native_node = platform.node()
        current_work_dir = os.getcwd()

        # 默认基础信息
        context = [
            f"- 宿主系统: {native_os} ({os.name})",
            f"- 宿主节点: {native_node}",
            f"- 宿主执行路径: `{current_work_dir}`",
            f"- 当前终端工具运行模式:",
        ]

        # 检查是否有活跃的终端会话
        from tools.System2.terminal_ops import TERMINAL_SESSIONS
        session = TERMINAL_SESSIONS.get(task_id)
        if not session:
            shell_type = "PowerShell" if native_os == "Windows" else "Bash"
            context.append(f"- 未初始化 (将默认使用 {shell_type})")
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

    async def _broadcast_task_event(self, task_id: str, detail_type: str, result: str, status: str, description: str):
        """多路原子事件广播"""
        subscribers = await global_task_registry.get_subscribers(task_id)
        if not subscribers:
            subscribers = ["private_internal:daemon"]

        for index, session_id in enumerate(subscribers):
            if index > 0: await asyncio.sleep(random.uniform(0.1, 1.0))

            # 动态坐标解析
            target_source = EventSource(platform="system")

            try:
                # 严格校验并解析 session_id (例如: private_onebot:12345)
                if "_" in session_id and ":" in session_id:
                    ctx_type, puid = session_id.split('_', 1)
                    plat, ctx_id = puid.split(':', 1)

                    target_source.platform = plat
                    if ctx_type == "group":
                        target_source.group_id = ctx_id
                    else:
                        target_source.user_id = ctx_id
                else:
                    # 格式不符时的降级处理
                    target_source.user_id = session_id
            except Exception as e:
                logger.warning(f"⚠️ 解析订阅者 session_id [{session_id}] 失败: {e}")

            # 推送事件
            self.event_bus.publish_event(OneBotEvent(
                type=EventType.TASK, detail_type=detail_type, source=target_source,
                extra={"task_payload": {"task_id": task_id, "result": result, "status": status,
                                        "description": description}}
            ))

    async def _call_llm(self, local_history: list, local_tool_manager: ToolManager, require_tools: bool) -> Dict[
        str, Any]:
        """纯函数化 LLM 投递"""
        tools = local_tool_manager.get_tool_schemas() or []
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

        sanitized_history = [{k: v for k, v in d.items() if k != 'metadata'} for d in local_history]

        # 调用 GenericAPIClient
        return await self.api_client.create_chat_completion(
            messages=sanitized_history,
            tools=tools if tools else None,
            schema=thought_structure,
            require_tools=require_tools
        )
