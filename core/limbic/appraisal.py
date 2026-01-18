# core/limbic/appraisal.py
import json
import logging
from typing import Dict, Any

from core.infrastructure.api_client import GenericAPIClient
from core.limbic.arch import NeuroState

logger = logging.getLogger(__name__)


class AppraisalSystem:
    """
    认知评估系统 (Appraisal):
    将自然语言事件转化为神经递质的刺激值。
    """

    def __init__(self, api_client: GenericAPIClient):
        self.api_client = api_client

    async def evaluate_event(self, event_content: str, current_state: NeuroState) -> Dict[str, Any]:
        """
        调用 LLM 评估事件。
        """
        # 构造 Prompt
        prompt = f"""
你是一个数字生命的边缘系统（潜意识）。
请根据当前状态和外部刺激，评估其对神经递质的影响。

【当前状态】
- 多巴胺 (动力): {current_state.dopamine:.2f}
- 皮质醇 (压力): {current_state.cortisol:.2f}
- 社交饱腹感: {current_state.social_satiety:.2f}

【外部刺激】
"{event_content}"

【评估规则】
1. Dopamine (多巴胺): 新奇、夸奖、有趣 -> 增加 (+0.1~0.3)；重复、无聊 -> 减少。
2. Cortisol (皮质醇): 攻击、错误、高压 -> 增加；安慰、安全 -> 减少。
3. Serotonin (血清素): 获得认同、完成任务 -> 增加。
4. Oxytocin (催产素): 亲密互动、陪伴 -> 增加。

请输出 JSON (刺激增量 -0.5 ~ +0.5):
{{
  "dopamine": 0.0,
  "cortisol": 0.0,
  "serotonin": 0.0,
  "oxytocin": 0.0,
  "reason": "简短分析"
}}
"""
        try:
            # 使用较快的模型或默认模型
            resp = await self.api_client.create_chat_completion(
                messages=[{"role": "system", "content": prompt}],
                schema={
                    "type": "object",
                    "properties": {
                        "dopamine": {"type": "number"},
                        "cortisol": {"type": "number"},
                        "serotonin": {"type": "number"},
                        "oxytocin": {"type": "number"},
                        "reason": {"type": "string"}
                    },
                    "required": ["dopamine", "cortisol", "reason"]
                }
            )

            content = resp["choices"][0]["message"]["content"]
            result = json.loads(content)
            logger.debug(f"🧠 [Appraisal] {result.get('reason')} | D:{result.get('dopamine'):+.2f}")
            return result

        except Exception as e:
            logger.error(f"Appraisal failed: {e}")
            return {}
