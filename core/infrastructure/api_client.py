# core/infrastructure/api_client.py
import json
import logging
from typing import Dict, Optional

import aiohttp

from core.infrastructure.config_loader import Config

logger = logging.getLogger(__name__)


class GenericAPIClient:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.get("llm.api_base_url")
        self.api_key = config.get("llm.api_key")
        self.model = config.get("llm.model_name")
        self.small_model = config.get("llm.small_model", self.model)
        self.embedding_model = config.get("llm.embedding_model_name")

        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_proxy()

    def _setup_proxy(self):
        self.proxy = None
        if self.config.get("proxy.enabled_llm"):
            self.proxy = self.config.get("proxy.https") or self.config.get("proxy.http")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            # 调大超时时间，防止思考时间过长导致断连
            timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=120)
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=timeout
            )
        return self.session

    async def close(self):
        if self.session:
            await self.session.close()

    async def create_chat_completion(self, messages: list, model: str = None, tools: list = None,
                                     schema: Dict = None, tool_choice: str = "auto") -> Dict:
        """核心 LLM 调用方法"""
        model = model or self.model
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages
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
            session = await self._get_session()
            async with session.post(endpoint, json=payload, proxy=self.proxy) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            logger.debug(json.dumps(payload, ensure_ascii=False))

            # 发生错误时，强制关闭并重置 session
            # 这样下一次重试时会创建一个全新的连接，避免 WinError 10053 复用死连接
            if self.session:
                await self.session.close()
            self.session = None

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
        endpoint = f"{self.base_url}/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": text
        }
        try:
            session = await self._get_session()
            async with session.post(endpoint, json=payload, proxy=self.proxy) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Embedding API 调用失败: {e}")
            # Embedding 出错也同样重置连接
            if self.session:
                await self.session.close()
            self.session = None
            return []
