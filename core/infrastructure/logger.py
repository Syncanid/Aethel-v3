import logging
import os
import sys
from datetime import datetime

import colorlog

from core.infrastructure.config_loader import Config
from core.utilities import get_log_filename


def setup_logger(config: Config):
    """初始化全局日志系统"""

    # 1. 确定日志级别
    level_str = config.get("system.log_level", "INFO").upper()
    log_level = getattr(logging, level_str, logging.INFO)

    # 2. 准备日志目录
    log_dir = config.get("storage.log_dir", "data/logs")
    os.makedirs(log_dir, exist_ok=True)
    base_name = f"{log_dir}/Aethel_Trinity_{datetime.now().strftime('%Y-%m-%d')}"

    # 3. 配置 Root Logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除旧的 handlers (防止重复)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. 控制台 Handler (带颜色)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)
    console_fmt = "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(name)s:%(lineno)d - %(message)s"
    console_handler.setFormatter(colorlog.ColoredFormatter(
        console_fmt,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        datefmt="%H:%M:%S",
        reset=True,
        style='%'
    ))
    logger.addHandler(console_handler)

    # 5. 文件 Handler (普通文本)
    file_handler = logging.FileHandler(get_log_filename(base_name), encoding='utf-8')
    file_handler.setLevel(log_level)
    file_fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
    file_handler.setFormatter(logging.Formatter(file_fmt))
    logger.addHandler(file_handler)

    # 抑制部分嘈杂的库日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)

    logging.getLogger("core").info(f"日志系统已启动，级别: {level_str}")
