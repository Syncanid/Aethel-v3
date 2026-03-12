# tools/network.py
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from core.infrastructure.config_loader import Config
from core.tool_manager.output_cache import ToolOutputCache
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
async def web_search(
        config: Config,
        query: str,
        safesearch: Optional[bool] = True,
) -> str:
    """
    [Network] 使用 SearXNG 引擎进行联网搜索。

    Args:
        query: 搜索关键词。
        safesearch: 是否启用安全搜索（默认开启）。
    """
    base_url = config.get("searxng_base_url")
    if not base_url:
        return "错误: 未配置 searxng_base_url。"

    url = f"{base_url}/search"
    params = {
        "q": query,
        "format": "json",
        "language": "zh-CN",
        "safesearch": 1 if safesearch else 0
    }

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
                f"    摘要: {item.get('content', '无')}\n"
            )
        return "\n".join(formatted)

    except Exception as e:
        return f"搜索出错: {str(e)}"


@register()
async def browse_website(
        url: str,
        purpose: str,
) -> str:
    """
    [Network] 访问指定 URL 并提取网页正文内容。
    用于深入阅读搜索结果中的网页。

    Args:
        url: 目标网页地址。
        purpose: 你为什么要搜索这个？你想从结果中得出什么结论？
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

            text = soup.get_text(separator="\n")

            # 清理空行
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = '\n'.join(lines)

            receipt_id, refined, final_response = await ToolOutputCache.process_tool_output(
                raw_content=f"=== 网页内容: {url} ===\n{cleaned_text}",
                purpose=purpose
            )

            return final_response

    except Exception as e:
        return f"无法访问网页 {url}: {e}"


@register()
async def send_http_request(
        method: str,
        url: str,
        headers: Optional[dict] = None,
        data: Optional[str] = None
) -> str:
    """
    [Network] 发送自定义 HTTP 请求 (GET, POST, PUT, DELETE)。
    用于测试 API 或与外部服务交互。

    Args:
        method: 请求方法 (GET, POST 等)。
        url: 请求地址。
        headers: 请求头字典 (可选)。
        data: 请求体内容 (可选)。
    """
    method = method.upper()

    # 直接处理传入的字典或字符串，增强鲁棒性
    json_headers = {}
    if headers:
        if isinstance(headers, dict):
            json_headers = headers
        elif isinstance(headers, str):
            try:
                json_headers = json.loads(headers)
            except:
                return "错误: headers 解析失败，请提供有效的字典或 JSON 字符串。"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.request(method, url, headers=json_headers, content=data)

            result = f"Status: {resp.status_code}\n"
            try:
                result += f"Body: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}"
            except:
                result += f"Body: {resp.text}"
            return result
    except Exception as e:
        return f"HTTP 请求失败: {e}"


@register()
async def download_file(
        config: Config,
        url: str,
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = 300.0,
        chunk_size: Optional[int] = 8192,
        overwrite: Optional[bool] = False
) -> str:
    """
    [Network] 下载文件并保存到本地磁盘。
    支持流式下载大文件、代理配置、自定义请求头和断点续传基础支持。

    Args:
        url: 要下载的文件 URL 地址。
        save_path: 保存目录路径（可选），默认为当前工作目录下的 'downloads' 文件夹。
        filename: 指定保存的文件名（可选），不指定则从 URL 或 Content-Disposition 自动提取。
        headers: 请求头字典（可选），如认证信息。
        timeout: 下载超时时间（秒），默认 300 秒（5 分钟），大文件可适当延长。
        chunk_size: 分块下载大小（字节），默认 8KB。
        overwrite: 是否覆盖已存在的同名文件，默认 False（跳过下载）。

    Returns:
        str: 下载结果信息，包含文件路径、大小、状态等。
    """
    from urllib.parse import urlparse, unquote

    # 解析请求头，直接合并字典
    json_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }

    if headers:
        if isinstance(headers, dict):
            json_headers.update(headers)
        elif isinstance(headers, str):
            try:
                json_headers.update(json.loads(headers))
            except json.JSONDecodeError:
                return "错误: headers 必须是有效的 JSON 字符串或字典。"

    # 准备保存路径
    if save_path is None:
        save_path = Path("downloads")
    else:
        save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    # 尝试从 URL 或响应头获取文件名
    parsed_url = urlparse(url)
    default_filename = unquote(Path(parsed_url.path).name) or "downloaded_file"

    try:
        async with httpx.AsyncClient(
                headers=json_headers,
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                proxy=_get_proxies(config)
        ) as client:
            # 先发送 HEAD 或 GET 请求获取文件信息（使用 stream 模式）
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()

                # 尝试从 Content-Disposition 获取文件名
                content_disposition = resp.headers.get("content-disposition", "")
                if filename is None and "filename=" in content_disposition:
                    import re
                    match = re.search(r'filename\*=UTF-8\'\'([^\s;]+)|filename=["\']?([^\s;\'"]+)', content_disposition)
                    if match:
                        filename = unquote(match.group(1) or match.group(2))

                if filename is None:
                    filename = default_filename

                # 确保文件名安全
                filename = Path(filename).name
                if not filename:
                    filename = "downloaded_file"

                file_path = save_path / filename

                # 检查文件是否已存在
                if file_path.exists() and not overwrite:
                    return f"文件已存在: {file_path}\n如需覆盖请设置 overwrite=True"

                # 获取文件大小（如果服务器提供）
                total_size = resp.headers.get("content-length")
                total_size = int(total_size) if total_size else None

                downloaded = 0
                # 使用临时文件避免下载中断产生残缺文件
                temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

                with open(temp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                # 下载完成重命名临时文件
                temp_path.rename(file_path)

                # 构建结果信息
                result = [
                    f"✓ 下载成功",
                    f"  源地址: {url}",
                    f"  保存路径: {file_path.resolve()}",
                    f"  文件大小: {_format_bytes(downloaded)}",
                ]
                if total_size and downloaded == total_size:
                    result.append(f"  状态: 完整下载")
                elif total_size:
                    result.append(f"  状态: 可能不完整 (期望: {_format_bytes(total_size)})")
                else:
                    result.append(f"  状态: 下载完成 (服务器未提供文件大小)")

                return "\n".join(result)

    except httpx.TimeoutException:
        return f"下载超时: 超过 {timeout} 秒未完成，请检查网络或增大 timeout 参数。"
    except httpx.HTTPStatusError as e:
        return f"HTTP 错误 {e.response.status_code}: {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"请求失败: {str(e)}"
    except PermissionError:
        return f"权限错误: 无法写入文件 {file_path}"
    except Exception as e:
        # 清理可能残留的临时文件
        if 'temp_path' in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        return f"下载异常: {type(e).__name__} - {str(e)}"


def _format_bytes(size: int) -> str:
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
