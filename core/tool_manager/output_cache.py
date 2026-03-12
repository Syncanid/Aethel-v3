# core/tool_manager/output_cache.py
import logging
import time
import uuid
from typing import Optional

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import get_config

logger = logging.getLogger(__name__)


class ToolOutputCache:
    """全局工具输出暂存池"""
    _cache = {}
    _api_client: Optional[GenericAPIClient] = None

    @classmethod
    def save(cls, content: str) -> str:
        """保存完整输出并返回凭证号 (Receipt ID)"""
        receipt_id = f"out_{uuid.uuid4().hex[:8]}"
        cls._cache[receipt_id] = {
            "content": content,
            "timestamp": time.time()
        }
        return receipt_id

    @classmethod
    def get(cls, receipt_id: str) -> Optional[str]:
        """根据凭证号获取原始输出"""
        if receipt_id in cls._cache:
            return cls._cache[receipt_id]["content"]
        return None

    @classmethod
    def _get_api_client(cls) -> GenericAPIClient:
        if not cls._api_client:
            cls._api_client = GenericAPIClient(get_config())
        return cls._api_client

    @classmethod
    async def refine_content(cls, raw_content: str, purpose: str) -> str:
        """
        调用 LLM，根据 Agent 的【目的】对长文本进行针对性提炼
        """
        api = cls._get_api_client()

        prompt = f"""
请你作为一个极度精准的“信息过滤器”。
以下是一个工具返回的【长篇原始输出】。请严格根据执行 Agent 的【调用目的】，从中提取并总结出唯一有用的信息。

【Agent 调用该工具的目的】: 
{purpose}

【工具原始输出】:
{raw_content}

【过滤要求】:
1. 直接输出提炼后的结果，不要任何废话前缀（如“根据你的目的...”）。
2. 无情地剔除与【目的】无关的所有细节！如果原始输出有 1000 字，但只有 1 句话满足目的，就只输出那 1 句话。
3. 如果仔细通读后，发现原始输出中【没有任何内容】能满足该目的，请明确回复：“原始输出中未包含能满足该目的的信息。”
"""
        try:
            # 优先使用 small_model 提速，因为只是做信息提取
            model = getattr(api, "small_model", api.model)
            message = await api.create_chat_completion_once(
                messages=prompt,
                system_prompt="你是一个冷酷、高效的信息过滤引擎。",
                model=model
            )

            content = message.get("content", "")
            # 清理可能存在的 thinking 标签
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()

            return content.strip()

        except Exception as e:
            logger.error(f"提取内容失败: {e}")
            return "【系统警告】提炼引擎报错，提炼失败。请使用 read_full_output 工具全量读取。"

    @classmethod
    async def process_tool_output(cls, raw_content: str, purpose: str, threshold: int = 200) -> tuple[str, str, str]:
        """
        [辅助函数] 一键完成：保存原文 -> 提炼内容 -> 拼装最终回复

        Args:
            raw_content: 工具产生的大量原始输出
            purpose: Agent 调用该工具的目的
            threshold: 触发提炼的字符数阈值。低于此值直接返回，节省 Token。

        Returns:
            tuple: (receipt_id, refined_result, final_response)
        """
        # 1. 长度熔断：如果输出本来就很短，不需要浪费 Token 去提炼
        if len(raw_content) < threshold:
            # 依然保存一下，以防万一
            receipt_id = cls.save(raw_content)
            final_response = f"{raw_content}\n\n---\n[系统提示]: 结果较短，已全量展示。(凭证: {receipt_id})"
            return receipt_id, raw_content, final_response

        # 2. 保存完整原始输出并获取凭证号
        receipt_id = cls.save(raw_content)

        # 3. 根据目的调用 LLM 提炼引擎
        refined_result = await cls.refine_content(raw_content, purpose)

        # 4. 拼装标准化带有“收据后门”的系统提示
        final_response = f"""{refined_result}

    ---
    [系统提示]: 以上输出已根据您的目的进行了精简过滤。
    原始长文本数据已暂存，凭证号为: {receipt_id}。
    如需查看全部细节，请调用 `read_full_output` 工具；
    若想换个目的重新提取，请调用 `refine_tool_output` 工具。"""

        return receipt_id, refined_result, final_response
