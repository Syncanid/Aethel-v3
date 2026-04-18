# core/io/adapters/onebot_v11.py
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import websockets

from core.io.adapters.base import BaseAdapter
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventType, DetailType, EventSource, Action, ActionStatus, ActionResponse
from core.utilities import get_log_filename

logger = logging.getLogger(__name__)


class OneBotV11Adapter(BaseAdapter):
    """
    OneBot v11 (原 CQHTTP) 协议适配器
    支持 NapCat, LLOneBot, go-cqhttp 等实现
    """

    def __init__(self, event_bus: EventBus, config: Any):
        super().__init__(event_bus)
        # 获取配置，支持字典或对象属性访问
        self.config = config
        self.ws_url = config.get("system.onebot_url", "ws://127.0.0.1:3001")
        self.token = config.get("system.onebot_token", None)

        if not logger.handlers:
            log_file = "data/logs/Onebot_" + datetime.now().strftime('%Y-%m-%d')
            handler = logging.FileHandler(get_log_filename(log_file), encoding='utf-8')
            # logger.setLevel(logging.DEBUG)
            # handler.setFormatter(logging.Formatter(
            #     "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
            #     datefmt="%H:%M:%S"
            # ))
            # logger.addHandler(handler)
            # logger.propagate = False

            websockets_logger = logging.getLogger("websockets.client")
            websockets_logger.setLevel(logging.DEBUG)
            websockets_logger.propagate = False
            websockets_logger.addHandler(handler)

        self.websocket = None
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._running = False

    @property
    def platform_name(self) -> str:
        return "onebot"

    async def run(self):
        """启动 WebSocket 连接循环"""
        self._running = True
        logger.info(f"OneBot 适配器启动，目标地址: {self.ws_url}")

        while self._running:
            try:
                headers = {}
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"

                async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                    self.websocket = ws
                    logger.info("OneBot WebSocket 已连接")

                    # 触发一次连接事件
                    self._emit_lifecycle_event(DetailType.CONNECT)

                    async for message in ws:
                        await self._handle_raw_message(message)

            except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
                logger.warning(f"OneBot 连接断开或失败 ({e})，5秒后重试...")
                self.websocket = None
                # 清理悬挂的请求
                for future in self._pending_requests.values():
                    if not future.done():
                        future.cancel()
                self._pending_requests.clear()

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"OneBot 适配器发生未捕获异常: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _handle_raw_message(self, raw_msg: str):
        """分发处理 WS 消息"""
        try:
            data = json.loads(raw_msg)

            # 1. API 响应 (Echo 匹配)
            if "echo" in data:
                echo_id = data["echo"]
                if echo_id in self._pending_requests:
                    future = self._pending_requests.pop(echo_id)
                    if not future.done():
                        future.set_result(data)
                return

            # 2. 事件推送
            post_type = data.get("post_type")
            if post_type == "message":
                await self._process_message_event(data)
            elif post_type == "request":
                await self._process_request_event(data)
            elif post_type == "meta_event":
                if data.get("meta_event_type") == "heartbeat":
                    # 心跳包通常忽略，或者用来更新状态
                    pass
            elif post_type == "notice":
                # 处理群成员增减等通知
                pass

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"处理 OneBot 消息失败: {e}", exc_info=True)

    async def _process_message_event(self, data: Dict[str, Any]):
        """将 OneBot 消息转换为 v3 标准事件"""

        # 提取基础信息
        msg_type = data.get("message_type")  # group / private
        sub_type = data.get("sub_type", "normal")

        # 构建事件源
        user_id = str(data.get("user_id", ""))
        group_id = str(data.get("group_id", "")) if msg_type == "group" else None

        if user_id == str(self.config.get("bot_self_id")):
            return

        source = EventSource(
            platform=self.platform_name,
            user_id=user_id,
            group_id=group_id
        )

        # 映射 DetailType
        detail_type = DetailType.GROUP if msg_type == "group" else DetailType.PRIVATE

        # 处理消息内容 (简单解析文本 + 保留原始结构)
        raw_message = data.get("message", "")
        alt_text = ""
        image_list = []

        # 如果是 list 格式 (OneBot v11 Array)
        if isinstance(raw_message, list):
            for seg in raw_message:
                t = seg.get("type")
                d = seg.get("data", {})
                if t == "text":
                    alt_text += d.get("text", "")
                elif t == "image":
                    alt_text += "[图片]"
                    # 提取 URL 或本地文件路径
                    img_url = d.get("url") or d.get("file")
                    if img_url:
                        image_list.append(img_url)
                elif t == "face":
                    alt_text += "[表情]"
                elif t == "at":
                    alt_text += f"@{d.get('qq', 'User')} "
                elif t == "json":
                    alt_text += "[卡片消息]"
                else:
                    alt_text += f"[{t}]"
        else:
            # 如果是 string 格式 (CQ码)，直接作为 alt_text
            alt_text = str(raw_message)

        # 构造 v3 事件
        event = OneBotEvent(
            type=EventType.MESSAGE,
            detail_type=detail_type,
            sub_type=sub_type,
            source=source,
            message=raw_message,  # 保留原始结构给 Agent 分析
            alt_message=alt_text,  # 纯文本供简单处理
            raw_data=data,
            extra={
                "sender_info": data.get("sender", {}),
                "msg_id": data.get("message_id"),
                "images": image_list
            }
        )

        self.event_bus.publish_event(event)

    async def _process_request_event(self, data: Dict[str, Any]):
        """将 OneBot 请求消息转换为 v3 标准事件"""
        request_type = data.get("request_type")
        user_id = str(data.get("user_id", ""))

        source = EventSource(
            platform=self.platform_name,
            user_id=user_id,
            group_id=str(data.get("group_id", "")) if request_type == "group" else None
        )

        event = OneBotEvent(
            type=EventType.REQUEST,
            detail_type=request_type,  # "friend" 或 "group"
            source=source,
            message=data.get("comment", ""),  # 将验证消息作为主体
            raw_data=data,
            extra={
                "flag": data.get("flag", ""),
                "comment": data.get("comment", ""),
                "sub_type": data.get("sub_type", "")  # "add" 或 "invite"
            }
        )
        self.event_bus.publish_event(event)

    async def handle_action(self, action: Action) -> Optional[ActionResponse]:
        """
        处理系统发出的 Action 并返回执行结果
        """
        # 1. 路由检查
        if action.target_platform and action.target_platform != self.platform_name:
            return None

        # 2. 参数映射
        api_name = ""
        params = {}

        if action.action == "send_message":
            # 智能判断发送目标
            msg = action.params.get("message")

            # 优先检查 Action 中是否显式指定了 target
            target_group = str(action.params.get("group_id")) if action.params.get("group_id") else None
            target_user = str(action.params.get("user_id")) if action.params.get("user_id") else None

            if target_group:
                resp = await self.call_api("get_group_list")
                data = resp['data']
                groups = []
                for group in data:
                    groups.append(str(group["group_id"]))
                if target_group not in groups:
                    return ActionResponse(status=ActionStatus.FAILED, message="未知的群聊，请检查群聊列表")
                api_name = "send_group_msg"
                params = {"group_id": target_group, "message": msg}
            elif target_user:
                resp = await self.call_api("get_friend_list")
                data = resp['data']
                friends = []
                self_id = int(self.config.get("system.bot_self_id", 0)) if self.config else 0
                for friend in data:
                    if friend["user_id"] == self_id:
                        continue
                    friends.append(str(friend["user_id"]))
                if target_user not in friends:
                    print(friends)
                    return ActionResponse(status=ActionStatus.FAILED, message="未知的用户，请检查好友列表")
                api_name = "send_private_msg"
                params = {"user_id": target_user, "message": msg}
            else:
                return ActionResponse(status=ActionStatus.FAILED, message="Missing group_id or user_id")

        elif action.action == "delete_msg":
            api_name = "delete_msg"
            params = {"message_id": action.params.get("message_id")}

        if not api_name:
            # 不是此适配器支持的标准动作
            return None

        # 3. 执行调用与结果封装
        try:
            # 3. 发送请求并等待 Echo
            resp = await self.call_api(api_name, params)

            if not resp:
                return ActionResponse(status=ActionStatus.FAILED, message="Network Timeout (No Response)")

            if resp.get("status") == "ok" and resp.get("retcode") == 0:
                return ActionResponse(
                    status=ActionStatus.OK,
                    data=resp.get("data", {}),
                    message="Success"
                )
            else:
                # 提取错误信息
                err_msg = resp.get("msg") or resp.get("wording") or f"Retcode: {resp.get('retcode')}"
                return ActionResponse(
                    status=ActionStatus.FAILED,
                    message=f"OneBot API Error: {err_msg}",
                    data=resp
                )

        except Exception as e:
            logger.error(f"Action {action.action} 执行异常: {e}", exc_info=True)
            return ActionResponse(status=ActionStatus.FAILED, message=f"Adapter Exception: {str(e)}")

    async def call_api(self, action: str, params: Dict = None, timeout: float = 20.0) -> Any:
        """
        [Public] 调用 API 并等待响应 (Echo 机制)
        """
        if not self.websocket:
            logger.warning(f"WebSocket 未连接，无法调用 API: {action}")
            return {"status": "failed", "retcode": -1, "msg": "WebSocket Disconnected"}

        params = params or {}
        echo_id = str(uuid.uuid4())
        payload = {
            "action": action,
            "params": params,
            "echo": echo_id
        }

        # 创建 Future 等待响应
        future = asyncio.Future()
        self._pending_requests[echo_id] = future

        try:
            await self.websocket.send(json.dumps(payload))
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"API 请求超时: {action}")
            await self._pending_requests.pop(echo_id, None)
            return {"status": "failed", "retcode": -1, "msg": "Timeout"}
        except Exception as e:
            logger.error(f"API 请求异常: {e}", exc_info=True)
            await self._pending_requests.pop(echo_id, None)
            return {"status": "failed", "retcode": -1, "msg": str(e)}

    def _emit_lifecycle_event(self, detail_type: DetailType):
        event = OneBotEvent(
            type=EventType.META,
            detail_type=detail_type,
            source=EventSource(platform=self.platform_name),
            message=f"OneBot adapter {detail_type}"
        )
        self.event_bus.publish_event(event)
