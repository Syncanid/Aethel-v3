# core/evolution/mimicry.py
import json
import logging
import os
from typing import List, Dict

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config

logger = logging.getLogger(__name__)


class SocialMimicry:
    def __init__(self, config: Config, api_client: GenericAPIClient):
        self.config = config
        self.api_client = api_client
        self.style_file = "data/style_config.json"
        self._ensure_style_file()

    def _ensure_style_file(self):
        if not os.path.exists(self.style_file):
            try:
                os.makedirs(os.path.dirname(self.style_file), exist_ok=True)
                with open(self.style_file, "w", encoding="utf-8") as f:
                    json.dump({"catchphrases": [], "emoji_style": [], "sentence_structure": []}, f)
            except Exception as e:
                logger.error(f"Failed to init style file: {e}", exc_info=True)

    async def evolve(self, recent_logs: List[Dict]):
        """
        [进化核心] 分析近期对话日志，提取群友的语言风格。
        通常在海马体归档周期(Dream Cycle)中被调用。
        """
        if not recent_logs:
            return

        # 过滤出 User 的发言，排除 Bot 自己和 System
        user_msgs = [msg.get("content", "") for msg in recent_logs if msg.get("role") == "user"]

        # 样本太少不进行分析，避免过拟合
        if len(user_msgs) < 5:
            return

        # 拼接最近的文本块
        text_block = "\n".join(user_msgs[-50:])

        prompt = """
你是社会语言学专家。请分析以下群聊日志中用户的语言风格，提取出具有代表性的特征，帮助 AI 更好地融入群体。

需提取：
1. High-frequency Slang (高频梗/黑话): 比如 "草", "乐", "绷不住了", "急了".
2. Emoji/Kaomoji Patterns (表情使用习惯): 比如 "🤣", "😭", "（", "xs".
3. Sentence Structure (句式特征): 比如 "倒装句", "省略主语", "重复叠词".

请输出 JSON:
{
    "catchphrases": ["string"],
    "emoji_style": ["string"],
    "sentence_structure": ["string"]
}
"""
        try:
            # 使用 Schema Mode 强制结构化输出
            response = await self.api_client.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text_block}
                ],
                schema={
                    "type": "object",
                    "properties": {
                        "catchphrases": {"type": "array", "items": {"type": "string"}},
                        "emoji_style": {"type": "array", "items": {"type": "string"}},
                        "sentence_structure": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["catchphrases", "emoji_style", "sentence_structure"],
                    "additionalProperties": False
                }
            )
            new_style = response.get("content", {})

            # 验证数据有效性
            if any(new_style.values()):
                self._save_style(new_style)
                logger.info(f"🧬 [Mimicry] 进化完成: 习得 {len(new_style.get('catchphrases', []))} 个新梗")

        except Exception as e:
            logger.error(f"Mimicry evolution failed: {e}", exc_info=True)

    def _save_style(self, data: Dict):
        """持久化风格配置"""
        try:
            with open(self.style_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save style config: {e}", exc_info=True)
