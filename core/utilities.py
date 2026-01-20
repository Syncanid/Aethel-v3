import os
from typing import Any


def get_log_filename(base_name):
    counter = 0
    filename = f"{base_name}.log"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_name}-{counter}.log"
    return filename


def calculate_tokens(content: Any) -> float:
    """
    [辅助方法] 计算单个内容块的 Token 估算值
    逻辑源自 v1 planner.py，区分中英文和图片
    """
    # 定义中英文的字符-Token比例（根据实测数据校准）
    CHARS_PER_CHINESE_TOKEN = 1.0  # 中文：286 字 / 177 Token ≈ 1.6 字/Token
    CHARS_PER_ENGLISH_TOKEN = 5.0  # 英文：1009 字 / 183 Token ≈ 5.5 字/Token
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
