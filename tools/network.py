# tools/network.py
import logging
import json
import httpx
from bs4 import BeautifulSoup
from core.infrastructure.config_loader import Config
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


# 获取代理配置辅助函数
def _get_proxies(config: Config):
    if config.get("proxy.enabled"):
        return {
            "http://": config.get("proxy.http"),
            "https://": config.get("proxy.https")
        }
    return None


@register()
async def web_search(config: Config, query: str) -> str:
    """
    [Network] 使用 SearXNG 引擎进行联网搜索。

    Args:
        query: 搜索关键词。
    """
    base_url = config.get("searxng_base_url")
    if not base_url:
        return "错误: 未配置 searxng_base_url。"

    url = f"{base_url}/search"
    params = {"q": query, "format": "json", "language": "zh-CN", "safesearch": 1}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return f"未找到 '{query}' 的相关结果。"

        formatted = []
        for i, item in enumerate(results[:5], 1):
            formatted.append(
                f"[{i}] {item.get('title', '无标题')}\n"
                f"    URL: {item.get('url', '')}\n"
                f"    摘要: {item.get('content', '无内容')}\n"
            )
        return "\n".join(formatted)

    except Exception as e:
        return f"搜索出错: {str(e)}"


@register()
async def browse_website(url: str, config: Config) -> str:
    """
    [Network] 访问指定 URL 并提取网页正文内容。
    用于深入阅读搜索结果中的网页。

    Args:
        url: 目标网页地址。
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }

    try:
        async with httpx.AsyncClient(headers=headers, timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            # 使用 BeautifulSoup 清洗 HTML
            soup = BeautifulSoup(resp.content, 'html.parser')

            # 移除干扰元素
            for element in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript"]):
                element.decompose()

            text = soup.get_text(separator='\n')

            # 清理空行
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = '\n'.join(lines)

            return f"=== 网页内容: {url} ===\n{cleaned_text[:5000]}"  # 限制长度防止爆 Token

    except Exception as e:
        return f"无法访问网页 {url}: {e}"


@register()
async def send_http_request(
        method: str,
        url: str,
        headers: str = None,
        data: str = None,
        config: Config = None
) -> str:
    """
    [Network] 发送自定义 HTTP 请求 (GET, POST, PUT, DELETE)。
    用于测试 API 或与外部服务交互。

    Args:
        method: 请求方法 (GET, POST 等)。
        url: 请求地址。
        headers: JSON 格式的请求头字符串 (可选)。
        data: 请求体内容 (可选)。
    """
    method = method.upper()

    json_headers = {}
    if headers:
        try:
            json_headers = json.loads(headers)
        except:
            return "错误: headers 必须是有效的 JSON 字符串。"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.request(method, url, headers=json_headers, content=data)

            result = f"Status: {resp.status_code}\n"
            try:
                result += f"Body: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}"
            except:
                result += f"Body: {resp.text[:2000]}"
            return result
    except Exception as e:
        return f"HTTP 请求失败: {e}"
