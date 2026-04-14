import base64
import logging
import mimetypes
import os
import urllib.parse
from typing import Any

import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

def get_log_filename(base_name):
    counter = 0
    filename = f"{base_name}.log"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}-{counter}.log"
    return filename

async def encode_image_to_data_uri(img_source: str) -> str:
    """
    [核心转换管线] 将异构图片地址统一转为大模型兼容的 Base64 Data URI
    """
    try:
        mime_type = "image/jpeg"  # 默认兜底
        image_data = b""

        # 1. 处理网络图片 (HTTP/HTTPS)
        if img_source.startswith("http://") or img_source.startswith("https://"):
            async with aiohttp.ClientSession() as session:
                # 设置合理的超时，防止大文件阻塞主循环
                async with session.get(img_source, timeout=10) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        mime_type = resp.headers.get("Content-Type", "image/jpeg")
                    else:
                        logger.error(f"图片下载失败 HTTP {resp.status}: {img_source}")
                        return ""

        # 2. 处理本地图片 (OneBot NapCat 常见的 file:/// 协议)
        else:
            # 剥离 file:// 协议头
            file_path = img_source.replace("file://", "", 1) if img_source.startswith("file://") else img_source

            # 处理 URL 编码的本地路径 (例如 Windows 下常见的 file:///C:/Users/%E4%B8%AD%E6%96%87...)
            if "%" in file_path:
                file_path = urllib.parse.unquote(file_path)

            # Windows 兼容性: 移除顶部的斜杠 (例如 /C:/Users/...)
            if os.name == 'nt' and file_path.startswith('/') and ':' in file_path:
                file_path = file_path[1:]

            if not os.path.exists(file_path):
                logger.warning(f"本地图片不存在或无权限读取: {file_path}")
                return ""

            # 动态推断正确的 MIME 类型
            guessed_mime, _ = mimetypes.guess_type(file_path)
            if guessed_mime:
                mime_type = guessed_mime

            async with aiofiles.open(file_path, "rb") as f:
                image_data = await f.read()

        if not image_data:
            return ""

        # 3. 封装为严格的标准规范 Data URI
        b64_str = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime_type};base64,{b64_str}"

    except Exception as e:
        logger.error(f"[多模态异常] 图片转 Base64 失败: {img_source}, 错误: {e}")
        return ""

def calculate_tokens(content: Any) -> int:
    """
    [辅助方法] 计算单个内容块的 Token 估算值
    逻辑源自 v1 planner.py，区分中英文和图片
    """
    # 定义中英文的字符-Token比例（根据实测数据校准）
    CHARS_PER_CHINESE_TOKEN = 1.0  # 中文：286 字 / 177 Token ≈ 1.6 字/Token
    CHARS_PER_ENGLISH_TOKEN = 4.0  # 英文
    SCREENSHOT_TOKEN_COST_APPROX = 30  # 1080p 截图的 Token 估算值

    estimated = 0.0

    if isinstance(content, list):  # 处理多模态内容（列表）
        for part in content:
            if part.get("type") == "text":
                text = str(part.get("text", ""))
                english_chars = sum(1 for char in text if ord(char) < 128)
                chinese_chars = len(text) - english_chars
                estimated += (english_chars / CHARS_PER_ENGLISH_TOKEN) + (chinese_chars / CHARS_PER_CHINESE_TOKEN)
            elif part.get("type") == "image_url" or part.get("type") == "image_base64":
                estimated += SCREENSHOT_TOKEN_COST_APPROX
    elif isinstance(content, str):
        text = content
        english_chars = sum(1 for char in text if ord(char) < 128)
        chinese_chars = len(text) - english_chars
        estimated += (english_chars / CHARS_PER_ENGLISH_TOKEN) + (chinese_chars / CHARS_PER_CHINESE_TOKEN)

    return estimated
