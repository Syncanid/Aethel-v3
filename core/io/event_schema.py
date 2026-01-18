import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel, Field


# --- 枚举定义 ---

class EventType(str, Enum):
    META = "meta"
    MESSAGE = "message"
    NOTICE = "notice"
    REQUEST = "request"


class DetailType(str, Enum):
    # Meta
    HEARTBEAT = "heartbeat"
    CONNECT = "connect"
    STATUS_UPDATE = "status_update"
    INTERNAL_DRIVE = "internal_drive"

    # Message
    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"
    # Notice
    MEMBER_INCREASE = "group_member_increase"
    MEMBER_DECREASE = "group_member_decrease"
    # Request
    FRIEND = "friend"


class ActionStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


# --- 基础模型 ---

class EventSource(BaseModel):
    """事件来源描述"""
    platform: str  # 平台名称，如 'console', 'qq', 'telegram', 'internal'
    user_id: Optional[str] = None  # 用户ID
    group_id: Optional[str] = None
    channel_id: Optional[str] = None


class OneBotEvent(BaseModel):
    """
    OneBot v12 标准事件模型
    所有进入 Aethel 核心的输入都必须被转换为此格式。
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time: float = Field(default_factory=time.time)
    type: EventType
    detail_type: str
    sub_type: str = ""

    # 事件源
    source: EventSource

    # 消息内容 (仅 Message 类型有效)
    message: Union[str, List[Dict[str, Any]]] = ""  # 支持纯文本或消息段列表
    alt_message: str = ""  # 纯文本替代表示

    # 原始数据 (用于调试或回溯)
    raw_data: Dict[str, Any] = {}

    # [扩展] 预留给情感引擎、中间件的额外字段
    # 例如: extra['sentiment'] = 'positive', extra['urgency'] = 10
    extra: Dict[str, Any] = {}

    class Config:
        use_enum_values = True


# --- 动作模型 (Output) ---

class Action(BaseModel):
    """
    系统发出的动作指令
    """
    action: str  # 例如 'send_message', 'delete_msg'
    params: Dict[str, Any] = {}
    echo: Optional[str] = None  # 用于请求响应匹配

    # 指定动作执行的平台/适配器，None表示广播或自动路由
    target_platform: Optional[str] = None
    target_self_id: Optional[str] = None


class ActionResponse(BaseModel):
    """动作执行结果"""
    status: ActionStatus
    retcode: int = 0
    data: Dict[str, Any] = {}
    message: str = ""
    echo: Optional[str] = None
