# core/infrastructure/api_client.py
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import httpx
from openai import AsyncOpenAI

from core.infrastructure.config_loader import Config
from core.utilities import get_log_filename

logger = logging.getLogger(__name__)


class GenericAPIClient:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.get("llm.api_base_url")
        self.api_key = config.get("llm.api_key")
        self.model = config.get("llm.model_name")
        self.small_model = config.get("llm.small_model", self.model)
        self.embedding_model = config.get("llm.embedding_model_name")
        self.enable_reasoning = config.get("llm.enable_reasoning", False)

        if not logger.handlers:
            log_file = "data/logs/Openai_" + datetime.now().strftime('%Y-%m-%d')
            handler = logging.FileHandler(get_log_filename(log_file), encoding='utf-8')
            # logger.setLevel(logging.DEBUG)
            # handler.setFormatter(logging.Formatter(
            #     "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
            #     datefmt="%H:%M:%S"
            # ))
            # logger.addHandler(handler)
            # logger.propagate = False

            websockets_logger = logging.getLogger("websockets.client")
            websockets_logger.setLevel(logging.DEBUG)
            websockets_logger.propagate = False
            websockets_logger.addHandler(handler)

        self.client: Optional[AsyncOpenAI] = None
        self._setup_proxy()

    def _setup_proxy(self):
        self.proxy = None
        if self.config.get("proxy.enabled_llm"):
            self.proxy = self.config.get("proxy.https") or self.config.get("proxy.http")

    def _get_client(self) -> AsyncOpenAI:
        """获取或初始化 AsyncOpenAI 客户端"""
        if self.client is None or self.client.is_closed():
            # 调大超时时间，防止思考时间过长导致断连
            timeout = httpx.Timeout(120.0, connect=10.0, read=120.0)

            # 如果配置了代理，则使用自定义的 httpx.AsyncClient
            http_client = None
            if self.proxy:
                http_client = httpx.AsyncClient(proxy=self.proxy, timeout=timeout)
            else:
                http_client = httpx.AsyncClient(timeout=timeout)

            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client
            )
        return self.client

    async def close(self):
        """关闭客户端资源"""
        if self.client:
            await self.client.close()

    async def create_chat_completion(self, messages: list, model: str = None, tools: list = None,
                                     schema: Dict = None, tool_choice: str = "auto") -> Dict:
        """核心 LLM 调用方法"""
        model = model or self.model
        client = self._get_client()

        has_user = any(msg.get("role") == "user" for msg in messages)
        if not has_user:
            # 倒序查找最后一个 system 消息的索引
            last_system_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "system":
                    last_system_idx = i
                    break

            if last_system_idx != -1:
                logger.warning("未在 messages 中发现 user 角色，将最后一个 system 消息重写为 user 角色以兼容本地模型。")
                messages[last_system_idx]["role"] = "user"
            else:
                # 极端异常情况：既没有 user 也没有 system
                error_msg = "LLM API 异常调用：messages 中既没有 user 角色，也没有 system 角色可供重写，请求被强制终止。"
                logger.error(error_msg)
                logger.error(f"异常的 messages 负载: {json.dumps(messages, ensure_ascii=False)}")
                raise ValueError(error_msg)  # 直接抛出异常，触发 traceback 阻断运行

        payload = {
            "model": model,
            "messages": messages
        }

        if self.enable_reasoning is not None:
            payload["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_reasoning}
            }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": True
                }
            }

        try:
            # 使用 openai 库发送请求
            response = await client.chat.completions.create(**payload)
            result = response.model_dump()

            for choice in result.get("choices", []):
                message = choice.get("message", {})
                if message:
                    reasoning_content = message.get("reasoning_content")
                    content = message.get("content")

                    # 确保 content 是字符串类型，处理 null
                    content_str = content if content is not None else ""

                    if reasoning_content:
                        if content_str.strip():
                            # 场景1：两者都有值，说明是正常的深度思考模型
                            message["content"] = f"<think>\n{reasoning_content}\n</think>\n{content_str}"
                        else:
                            # 场景2：content为空，但reasoning_content有值，说明触发了 LM Studio 错位 Bug
                            message["content"] = reasoning_content
                    else:
                        # 场景3：普通模型，没有思考字段
                        message["content"] = content_str

            return result

        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}", exc_info=True)
            logger.debug(json.dumps(payload, ensure_ascii=False))

            # 发生错误时，强制关闭并重置 client，以便下一次请求重建连接
            if self.client:
                await self.client.close()
            self.client = None

            raise

    async def create_chat_completion_once(self, messages: str, system_prompt: str = None, model: str = None,
                                          schema: Dict = None) -> dict:
        msg = []
        if system_prompt:
            msg.append({
                "role": "system",
                "content": system_prompt
            })
        msg.append({
            "role": "user",
            "content": messages
        })
        ret = await self.create_chat_completion(messages=msg, model=model, schema=schema)
        return ret["choices"][0]["message"]

    async def create_embedding(self, text: str) -> list:
        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            # 提取向量数据
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding API 调用失败: {e}", exc_info=True)
            # Embedding 出错也同样重置连接
            if self.client:
                await self.client.close()
            self.client = None
            return []
