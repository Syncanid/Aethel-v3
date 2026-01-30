# core/memory/infinite_context.py
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

        # 压缩提示词 (参考 ChatLuna)
        self.compress_prompt = """
你是 "Infinite Context"，一个对话式 AI 助手的记忆架构师。
你的任务是将提供的对话片段压缩成一份紧凑的“知识笔记”，仅保留对未来对话有必要的信息。

要求：
1. 将片段整理为清晰的主题。
2. 保留行动承诺、未解决的问题、决策、指令、用户偏好、情感基调变化和关键事实。
3. 移除闲聊、重复措辞或已过期的信息。
4. 使用按主题分组的简明要点。
5. 必须保留 XML 标签格式。

格式示例：
<infinite_context>
- 主题: <简短标题>
  - 细节: <关键事实、指令或待办任务>
</infinite_context>

如果之前的摘要存在（即输入中包含旧的 <infinite_context>），请将其与新对话合并更新。
如果没有任何重要信息保留，输出 "<infinite_context />"。
"""

    def _is_compressed_message(self, message: Dict) -> bool:
        """检查消息是否已经是压缩后的上下文"""
        content = message.get("content", "")
        return isinstance(content, str) and "<infinite_context>" in content

    async def compress_if_needed(self, history: List[Dict]) -> bool:
        """
        检查并执行压缩。如果历史记录被修改了，返回 True。
        :param history: Agent 的历史记录引用 (会直接修改)
        """

        current_tokens = 0.0
        for msg in history:
            current_tokens += calculate_tokens(msg.get("content"))

        # 如果未达到阈值，跳过
        if current_tokens < self.trigger_threshold:
            return False

        logger.info(f"🔄 [InfiniteContext] Token ({current_tokens}) 超过阈值，开始压缩...")

        # --- 策略分层 ---
        # 1. System Prompt (index 0): 始终保留
        # 2. 最近的消息 (例如最近 8 条): 始终保留，保持对话连贯性
        # 3. 中间层: 待压缩区域

        PRESERVE_COUNT = 8
        if len(history) <= PRESERVE_COUNT + 1:
            return False

        system_msg = history[0]
        recent_msgs = history[-PRESERVE_COUNT:]
        to_compress_msgs = history[1:-PRESERVE_COUNT]

        if not to_compress_msgs:
            return False

        # 检查是否已经有旧的摘要
        existing_summary = ""

        # 如果 to_compress_msgs 的第一条已经是 infinite_context，提取出来作为基础
        if self._is_compressed_message(to_compress_msgs[0]):
            existing_summary = to_compress_msgs[0]["content"]
            msgs_to_process = to_compress_msgs[1:]
        else:
            msgs_to_process = to_compress_msgs

        if not msgs_to_process:
            return False

        # 构建 LLM 输入
        conversation_text = ""
        if existing_summary:
            conversation_text += f"Existing Summary:\n{existing_summary}\n\n---\n\n"

        for msg in msgs_to_process:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # 简化多模态内容为文本标记，避免 Token 爆炸
            if isinstance(content, list):
                content = "[Multimodal Content]"
            conversation_text += f"[{role}]: {content}\n"

        # 调用 LLM 进行压缩
        try:
            new_summary = await self._generate_summary(conversation_text)

            if not new_summary or "<infinite_context" not in new_summary:
                logger.warning("[InfiniteContext] 压缩生成失败或格式错误，跳过本次压缩。")
                return False

            # --- 重组历史记录 ---
            # 新结构: [System] + [Infinite Context Summary] + [Recent Messages]

            summary_message = {
                "role": "system",  # 或者用 user，视模型遵循能力而定
                "content": f"以下是之前的对话记忆摘要，请基于此继续对话：\n{new_summary}"
            }

            # 直接修改引用的列表
            history.clear()
            history.append(system_msg)
            history.append(summary_message)
            history.extend(recent_msgs)

            logger.info(
                f"✅ [InfiniteContext] 压缩完成。历史记录从 {len(to_compress_msgs) + len(recent_msgs) + 1} 条减少到 {len(history)} 条。")
            return True

        except Exception as e:
            logger.error(f"[InfiniteContext] 压缩过程发生错误: {e}")
            return False

    async def _generate_summary(self, text: str) -> str:
        """调用 API 生成摘要"""
        messages = [
            {"role": "system", "content": self.compress_prompt},
            {"role": "user", "content": f"Conversation Fragment:\n{text}"}
        ]

        # 使用 create_chat_completion，不带 tools，纯文本生成
        response = await self.api_client.create_chat_completion(
            model=self.api_client.small_model,
            messages=messages,
        )

        return response["choices"][0]["message"]["content"]
