# core/infrastructure/config_loader.py
import logging
import os
import shutil
from typing import Any, Dict

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


class Config:
    _instance = None
    _config: Dict[str, Any] = {}
    _yaml_engine: YAML = None
    _config_path: str = "data/config.yaml"
    _default_path: str = "data/default-config.yaml"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # 初始化 Round-Trip 引擎，深度保留注释和物理排版
            cls._yaml_engine = YAML(typ='rt')
            cls._yaml_engine.preserve_quotes = True
            # 设置标准的 2 空格缩进
            cls._yaml_engine.indent(mapping=2, sequence=4, offset=2)
        return cls._instance

    def load(self, config_path: str = "data/config.yaml"):
        """
        加载 YAML 配置文件。
        如果主配置文件不存在，则尝试从默认模板复制。
        """
        self._config_path = config_path

        # 1. 检查主配置是否存在
        if not os.path.exists(self._config_path):
            logger.warning(f"主配置文件 {self._config_path} 未找到，尝试检查默认模板...")

            # 2. 检查默认模板是否存在
            if os.path.exists(self._default_path):
                try:
                    # 确保数据目录存在
                    os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
                    # 执行物理复制
                    shutil.copy2(self._default_path, self._config_path)
                    logger.info(f"✨ 已成功从 {self._default_path} 复制并创建了新的配置文件。")
                except Exception as e:
                    logger.error(f"复制默认配置失败: {e}")
                    return
            else:
                logger.error(f"严重错误: 找不到主配置 {self._config_path}，且默认模板 {self._default_path} 也不存在！")
                return

        # 3. 执行加载逻辑
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                # 解析出带有注释元数据的 CommentedMap 结构
                self._config = self._yaml_engine.load(f)

            if self._config is None:
                self._config = {}

            logger.info(f"配置已加载: {self._config_path}")
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
                # 兼容 ruamel 的 CommentedMap
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception:
            return default

    def set(self, key: str, value: Any):
        """
        动态设置配置项 (支持 'a.b.c' 格式的键路径)
        如果节点缺失，将自动补全嵌套的字典结构。
        """
        keys = key.split('.')
        d = self._config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                # 动态构建层级
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def save(self):
        """
        将内存中的配置脏数据原子化地覆写回磁盘。
        """
        try:
            # 提取所有顶层根节点
            root_keys = list(self._config.keys())

            for i, key in enumerate(root_keys):
                # 跳过第一个节点，从第二个节点开始处理
                if i > 0:
                    # 获取该节点现有的注释信息
                    # ca (comment attribute) 存储了节点前后的所有词法信息
                    # 这里的 '\n' 会被 ruamel 处理为物理上的空行
                    # 如果该节点之前已经有注释，它会将空行插入在注释之上
                    self._config.yaml_set_comment_before_after_key(key, before='\n')

            with open(self._config_path, "w", encoding="utf-8") as f:
                self._yaml_engine.dump(self._config, f)
            logger.info(f"🔧 配置已安全落盘至 {self._config_path}")
        except Exception as e:
            logger.error(f"配置文件物理落盘失败: {e}", exc_info=True)

    @property
    def all(self) -> Dict[str, Any]:
        return self._config


# 全局单例辅助函数
def get_config() -> Config:
    return Config()
