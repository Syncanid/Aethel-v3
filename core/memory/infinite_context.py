# core/memory/infinite_context.py
import json
import logging
from typing import List, Dict

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.utilities import calculate_tokens

logger = logging.getLogger(__name__)


class InfiniteContextManager:
    def __init__(self, config: Config, api_client: GenericAPIClient):
        self.config = config
        self.api_client = api_client
        # 从配置读取阈值，默认上下文窗口的 75% 触发压缩
        self.max_tokens = self.config.get("llm.model_context", 16384)
        self.trigger_threshold = self.max_tokens * 0.75

        # 压缩提示词
        self.compress_prompt = """
你是一个顶级的“对话记忆脱水引擎”。你的任务是将下方提供的一段极其冗长的历史对话记录，压缩并重写为【少量、极度精炼的对话回合】（仅限 User 和 Assistant 的问答对）。

【压缩原则】
1. 极度精简：无情地删除所有客套话、寒暄、停顿词、重复确认和无关痛痒的闲聊。
2. 核心保留：必须保留所有客观事实、给出的最终结论、报错代码、用户设定的规则和关键逻辑。
3. 消化工具日志：原始记录中可能包含冗长的 tool (工具执行日志)。请将这些日志的核心结果转化为 Assistant 的自然陈述（例如：“我通过系统检查发现网络确实断了”）。
4. 连贯性：压缩后的消息依然要保持一问一答的流畅性，让读取这份记忆的 AI 仿佛经历了一次极其高效、直奔主题的对话。
5. 角色一致：不要改变说话人的原本立场，千万不要把系统提示或工具报错写成用户的发言。
"""

    async def compress_if_needed(self, history: List[Dict]) -> bool:
        """
        检查并执行平滑的对话流压缩。如果历史记录被修改了，返回 True。
        :param history: Agent 的历史记录引用 (会直接修改)
        """

        current_tokens = 0.0
        for msg in history:
            current_tokens += calculate_tokens(msg.get("content", ""))

        # 如果未达到阈值，跳过
        if current_tokens < self.trigger_threshold:
            return False

        logger.info(f"🔄 [InfiniteContext] Token ({int(current_tokens)}) 超过阈值，开始对话压缩...")

        # --- 策略分层 ---
        # 1. System Prompt (index 0): 始终保留
        # 2. 最近的消息 (例如最近 8 条): 始终保留，保持短期对话的绝对连贯性
        # 3. 中间层: 待压缩区域

        PRESERVE_COUNT = 8
        if len(history) <= PRESERVE_COUNT + 1:
            return False

        system_msg = history[0]
        recent_msgs = history[-PRESERVE_COUNT:]
        to_compress_msgs = history[1:-PRESERVE_COUNT]

        if not to_compress_msgs:
            return False

        # 构建给 LLM 压缩用的文本副本
        conversation_text = ""
        for msg in to_compress_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # 简化多模态内容为文本标记，避免 Token 爆炸
            if isinstance(content, list):
                content = "[Multimodal Content]"

            # 清洗工具调用标记，方便 LLM 理解
            if msg.get("name"):
                role = f"tool_result({msg['name']})"
            elif msg.get("tool_calls"):
                role = "assistant_calling_tool"
                content = f"执行了工具调用: {msg['tool_calls']}"

            conversation_text += f"[{role}]: {content}\n"

        # 调用 LLM 进行压缩
        try:
            compressed_msgs = await self._generate_summary(conversation_text)

            if not compressed_msgs:
                logger.warning("⚠️ [InfiniteContext] 压缩返回为空或解析失败，跳过本次压缩。")
                return False

            # --- 重组历史记录 ---
            # 新结构: [System] + [脱水后的精炼 User/Assistant 回合] + [Recent Messages]

            # 在第一条压缩消息前加个极小的免责声明，帮助 Agent 认知这是回忆
            if compressed_msgs and compressed_msgs[0].get("role") == "user":
                compressed_msgs[0]["content"] = f"【以下是脱水精简后的早期记忆】\n{compressed_msgs[0]['content']}"
            else:
                compressed_msgs.insert(0, {"role": "user", "content": "【系统提示：以下是脱水精简后的早期对话记忆】"})

            # 直接修改引用的列表
            history.clear()
            history.append(system_msg)
            history.extend(compressed_msgs)
            history.extend(recent_msgs)

            logger.info(
                f"✅ [InfiniteContext] 脱水完成。历史记录从 {len(to_compress_msgs) + len(recent_msgs) + 1} 条减少到 {len(history)} 条。")
            return True

        except Exception as e:
            logger.error(f"❌ [InfiniteContext] 压缩过程发生致命错误: {e}", exc_info=True)
            return False

    async def _generate_summary(self, text: str) -> List[Dict[str, str]]:
        """调用 API 生成结构化的精炼对话对"""
        messages = [
            {"role": "system", "content": self.compress_prompt},
            {"role": "user", "content": f"【待压缩的原始对话流水】\n{text}"}
        ]

        # 强制输出 Schema
        schema = {
            "type": "object",
            "properties": {
                "compressed_dialogue": {
                    "type": "array",
                    "description": "压缩后的精炼对话列表，必须按时间顺序排列",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant"]},
                            "content": {"type": "string", "description": "该角色精简后的发言内容"}
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["compressed_dialogue"],
            "additionalProperties": False
        }

        # 优先使用 small_model 提速
        response = await self.api_client.create_chat_completion(
            model=self.api_client.small_model,
            messages=messages,
            schema=schema
        )
        data = response.get("content", {})
        return data.get("compressed_dialogue", [])