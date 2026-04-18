import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskRecord(BaseModel):
    """后台任务的状态记录"""
    task_id: str
    description: str  # 任务的原始描述
    status: TaskStatus = TaskStatus.PENDING
    progress_msg: str = "准备执行..."  # 实时进度描述
    result: Optional[str] = None
    subscribed_sessions: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class TaskRegistry:
    """
    任务状态共享看板 (Blackboard)
    用于 System 1 (轻循环) 和 System 2 (重循环) 的状态同步
    """

    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()  # 保证高并发下的异步安全

    async def subscribe_session(self, task_id: str, session_id: str):
        """将特定会话加入任务的事件广播监听池"""
        async with self._lock:
            if task_id in self._tasks:
                if session_id and session_id not in self._tasks[task_id].subscribed_sessions:
                    self._tasks[task_id].subscribed_sessions.append(session_id)

    async def get_subscribers(self, task_id: str) -> List[str]:
        """获取该任务的所有订阅会话 ID"""
        async with self._lock:
            if task_id in self._tasks:
                return list(self._tasks[task_id].subscribed_sessions)
            return []

    async def register_task(self, task_id: str, description: str) -> TaskRecord:
        """注册一个新任务"""
        async with self._lock:
            task = TaskRecord(task_id=task_id, description=description)
            self._tasks[task_id] = task
            return task

    async def update_progress(self, task_id: str, progress_msg: str):
        """更新任务的进度文本"""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].progress_msg = progress_msg
                self._tasks[task_id].status = TaskStatus.RUNNING
                self._tasks[task_id].updated_at = time.time()

    async def complete_task(self, task_id: str, result: str):
        """标记任务完成并附带结果摘要"""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.COMPLETED
                self._tasks[task_id].result = result
                self._tasks[task_id].updated_at = time.time()

    async def cancel_task(self, task_id: str):
        """取消任务"""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.CANCELLED
                self._tasks[task_id].updated_at = time.time()

    async def get_active_tasks(self) -> List[TaskRecord]:
        """获取所有未完成的活跃任务"""
        async with self._lock:
            return [
                task for task in self._tasks.values()
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]

    async def cleanup_finished_tasks(self, max_age_seconds: int = 300):
        """清理过期的已完成/取消任务，防止内存泄漏或幻觉"""
        async with self._lock:
            current_time = time.time()
            keys_to_delete = [
                task_id for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED)
                   and (current_time - task.updated_at) > max_age_seconds
            ]
            for k in keys_to_delete:
                del self._tasks[k]

    async def render_for_prompt(self) -> str:
        """
        为 System 1 生成注入 Prompt 的上下文文本。
        如果当前没有后台任务，返回空字符串。
        """
        active_tasks = await self.get_active_tasks()
        if not active_tasks:
            return ""

        xml_blocks = ["<Background_Tasks>"]
        for task in active_tasks:
            xml_blocks.append(f"  <Task id=\"{task.task_id}\">")
            xml_blocks.append(f"    <Description>{task.description}</Description>")
            xml_blocks.append(f"    <Status>{task.status.value}</Status>")
            xml_blocks.append(f"    <Progress>{task.progress_msg}</Progress>")
            xml_blocks.append("  </Task>")
        xml_blocks.append("</Background_Tasks>")

        return "\n".join(xml_blocks)


global_task_registry = TaskRegistry()
