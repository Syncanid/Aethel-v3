import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def load(self, config_path: str = "data/config.yaml"):
        """加载 YAML 配置文件"""
        if not os.path.exists(config_path):
            # 如果配置不存在，创建一个默认模板或报错
            logger.warning(f"配置文件 {config_path} 未找到")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"配置已加载: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}", exc_info=True)
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，支持点号索引，例如 get("llm.model_name")
        """
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value.get(k)
                if value is None:
                    return default
            return value
        except AttributeError:
            return default

    @property
    def all(self) -> Dict[str, Any]:
        return self._config


# 全局单例辅助函数
def get_config() -> Config:
    return Config()
