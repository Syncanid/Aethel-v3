import logging

import httpx

from core.infrastructure.config_loader import get_config, Config
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


@register()
async def web_search(config: Config, query: str) -> str:
    """
    使用 SearXNG 引擎进行联网搜索，获取实时信息。
    当用户询问当前新闻、具体事实、天气、或模型知识库之外的信息时使用此工具。

    Args:
        query: 搜索关键词，例如 "今天北京天气" 或 "最新的AI新闻"

    Returns:
        str: 搜索结果的摘要文本
    """
    base_url = config.get("searxng_base_url")
    if not base_url:
        return "错误: 未配置 searxng_base_url，请联系管理员检查 config。"

    # SearXNG 的标准 API 端点
    url = f"{base_url}/search"

    params = {
        "q": query,
        "format": "json",
        "language": "zh-CN",  # 默认中文优先，可根据需要调整
        "safesearch": 1  # 0=None, 1=Moderate, 2=Strict
    }

    logger.info(f"正在搜索: {query}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return f"搜索 '{query}' 未找到相关结果。"

        # 格式化前5条结果给 LLM
        # 包含 标题、链接 和 摘要
        formatted_results = []
        for i, item in enumerate(results[:5], 1):
            title = item.get("title", "无标题")
            link = item.get("url", "")
            content = item.get("content", "无内容")

            formatted_results.append(
                f"[{i}] {title}\n"
                f"    来源: {link}\n"
                f"    摘要: {content}\n"
            )

        return "\n".join(formatted_results)

    except httpx.HTTPStatusError as e:
        error_msg = f"搜索请求失败 (HTTP {e.response.status_code})"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"搜索执行出错: {str(e)}"
        logger.error(error_msg)
        return error_msg
