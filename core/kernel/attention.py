# core/kernel/attention.py
import json
import logging
from enum import Enum
from typing import Optional

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.io.event_schema import OneBotEvent, EventType, DetailType
from core.kernel.prompt import PromptManager

logger = logging.getLogger(__name__)


class ReactionType(str, Enum):
    REPLY = "reply"  # 必须回复 (直接交互/强相关)
    INTERJECT = "interject"  # 主动插话 (弱交互/感兴趣)
    OBSERVE = "observe"  # 静默观察 (无价值/不相关/插话阈值过高)
    IGNORE = "ignore"  # 完全忽略 (如黑名单/无关系统通知)


class AttentionFilter:
    def __init__(self, config: Config, prompt: PromptManager, api_client: GenericAPIClient):
        self.config = config
        self.api_client = api_client

        # 加载身份配置
        self.bot_self_id = str(config.get("system.bot_self_id", ""))
        self.nickname = prompt.role_data.get("identity", {}).get("name", "Aethel")

        # 动态阈值 (后续可接入 Limbic System 动态调整)
        self.interject_threshold = 0.75

    async def evaluate(self, event: OneBotEvent) -> ReactionType:
        """
        评估事件的重要性，决定响应策略。
        """
        # 1. 基础过滤：非消息事件通常由 Adapter 处理或转化为文本
        # 注意：内部驱动 (Internal Drive) 属于系统自发需求，必须响应
        if event.type != EventType.MESSAGE:
            if event.detail_type == DetailType.INTERNAL_DRIVE:
                return ReactionType.REPLY
            # 系统心跳、连接通知等 Meta 事件，默认静默
            return ReactionType.OBSERVE

        # 2. 硬规则检测 (Hard Rules) - 优先匹配强信号
        hard_reaction = self._check_hard_rules(event)
        if hard_reaction:
            logger.info(f"⚡ [Attention] 硬规则触发: {hard_reaction.value}")
            return hard_reaction

        # 3. 软规则检测 (Soft Semantics) - LLM 深度评估
        # 只有在硬规则未决定的情况下才调用 LLM，节省资源
        soft_reaction = await self._check_soft_rules(event)
        logger.info(f"🧠 [Attention] 语义评估: {soft_reaction.value}")
        return soft_reaction

    def _check_hard_rules(self, event: OneBotEvent) -> Optional[ReactionType]:
        """基于规则的快速匹配"""

        # 规则 1: 私聊消息 -> 必须回复
        if event.detail_type == DetailType.PRIVATE:
            return ReactionType.REPLY

        # 规则 2: 群聊中被 @ -> 必须回复
        # OneBot v11 的 CQ 码格式: [CQ:at,qq=123456]
        if self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in event.alt_message:
            return ReactionType.REPLY

        # 规则 3: 消息以 Bot 名字开头 (呼叫模式) -> 必须回复
        clean_msg = event.alt_message.strip()
        if self.nickname and clean_msg.startswith(self.nickname):
            return ReactionType.REPLY

        return None

    async def _check_soft_rules(self, event: OneBotEvent) -> ReactionType:
        """
        调用 LLM 评估消息的相关性与插话价值
        """
        content = event.alt_message
        sender = event.source.user_id

        if not content:
            return ReactionType.OBSERVE

        # 构造 Prompt
        prompt = f"""
你是一个群聊中的 AI 助手 ({self.nickname})。请作为“注意力过滤器”，评估以下用户消息，决定是否需要介入。

【当前消息】
发送者: {sender}
内容: "{content}"

【决策标准】
1. REPLY (回复): 
   - 用户在向你提问 (即使没 @ 你)。
   - 话题与你高度相关。
   - 检测到用户情绪激动，需要安抚。

2. INTERJECT (插话): 
   - 用户在聊其他话题，但你觉得非常有趣、有梗。
   - 你的专业知识能提供巨大帮助。
   - 注意：不要做一个烦人的插话者，只有高质量的插话才被允许。

3. OBSERVE (观察): 
   - 闲聊、无关话题。
   - 争吵、辱骂等负面内容。
   - 你插不上话，或者不需要你参与的内容。

请输出 JSON:
{{
    "decision": "REPLY" | "INTERJECT" | "OBSERVE",
    "reason": "简短理由",
    "confidence": 0.0~1.0
}}
"""
        try:
            # 使用 create_chat_completion 的 schema 模式强制结构化输出
            response = await self.api_client.create_chat_completion(
                messages=[{"role": "system", "content": prompt}],
                schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["REPLY", "INTERJECT", "OBSERVE"]},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["decision", "reason", "confidence"],
                    "additionalProperties": False
                }
            )

            content_str = response["choices"][0]["message"]["content"]
            result = json.loads(content_str)

            decision = result.get("decision", "OBSERVE")
            confidence = result.get("confidence", 0.0)

            # 决策逻辑
            if decision == "REPLY":
                return ReactionType.REPLY

            if decision == "INTERJECT":
                # 插话需要高置信度
                if confidence >= self.interject_threshold:
                    return ReactionType.INTERJECT
                else:
                    logger.info(f"🛑 [Attention] 抑制插话意图 (置信度 {confidence:.2f} < {self.interject_threshold})")
                    return ReactionType.OBSERVE

            return ReactionType.OBSERVE

        except Exception as e:
            logger.error(f"Attention LLM check failed: {e}")
            # 发生错误时保持安静，避免刷屏
            return ReactionType.OBSERVE
