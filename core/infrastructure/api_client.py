# core/infrastructure/api_client.py
import copy
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import aiofiles
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
        self.use_prompt_tools = self.config.get("llm.use_prompt_tools", False)
        self.use_schema_tools = self.config.get("llm.use_schema_tool_calls", True)
        self.arg_mode = self.config.get("llm.tool_call_arg_mode", "object")

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

            httpcore_logger = logging.getLogger("httpcore")
            httpcore_logger.setLevel(logging.DEBUG)
            httpcore_logger.propagate = False
            httpcore_logger.addHandler(handler)

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

    def _compress_tool_schemas(self, tools: list) -> str:
        """
        将低信息密度的 JSON Tool Schema 压缩为高密度伪代码。
        """
        if not tools:
            return ""

        lines = [
            "\n[SYSTEM DIRECTIVE: AVAILABLE TOOLS]\n你可以使用以下工具，请在返回的 JSON 的 tool_calls 节点中按需调用："]
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "").replace("\n", " ")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            param_strs = []
            for p_name, p_attr in params.items():
                p_type = p_attr.get("type", "any")
                is_req = "" if p_name in required else "?"
                p_desc = p_attr.get("description", "")

                if p_desc:
                    param_strs.append(f"{p_name}{is_req}: {p_type} /* {p_desc} */")
                else:
                    param_strs.append(f"{p_name}{is_req}: {p_type}")

            params_joined = ",\n    ".join(param_strs)

            if len(param_strs) > 1:
                signature = f"- {name}(\n    {params_joined}\n  )"
            else:
                signature = f"- {name}({params_joined})"

            lines.append(f"{signature}\n  用途: {desc}")

        return "\n".join(lines)

    async def create_chat_completion(self, messages: list, model: str = None, tools: list = None,
                                     schema: Dict = None, require_tools: bool = False) -> Dict:
        """核心 LLM 调用方法"""
        model = model or self.model
        client = self._get_client()

        tool_choice_val = "auto"

        if self.use_prompt_tools and not self.use_schema_tools:
            logger.warning("配置冲突: use_prompt_tools 必须在 use_schema_tool_calls 开启时才有效。已自动强制开启 Schema 工具模式。")
            self.use_schema_tools = True

        # 1. Prompt Tools 模式干预
        if self.use_prompt_tools and tools and schema:
            logger.debug("API_Client: 启用 Prompt Tools，执行注入...")
            compressed_tools_text = self._compress_tool_schemas(tools)
            messages = copy.deepcopy(messages)

            system_idx = next((i for i, msg in enumerate(messages) if msg.get("role") == "system"), -1)
            if system_idx != -1:
                messages[system_idx]["content"] += f"\n{compressed_tools_text}"
            else:
                messages.insert(0, {"role": "system", "content": compressed_tools_text})

        # 2. Schema Tools 模式干预
        if self.use_schema_tools and tools and schema:
            logger.debug("API_Client: 启用 Schema Tool Calls，自动重构参数约束...")
            schema = copy.deepcopy(schema)

            arg_schema = {"type": "object"} if self.arg_mode == "object" else {
                "type": "string",
                "description": "工具的参数对象，JSON格式。"
            }

            if "properties" not in schema:
                schema["properties"] = {}

            tool_calls_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": arg_schema
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False
                }
            }

            if require_tools:
                # 强制要求 key 存在
                if "required" not in schema:
                    schema["required"] = []
                if "tool_calls" not in schema["required"]:
                    schema["required"].append("tool_calls")

                # 强制要求数组内至少包含 1 个元素，消灭 [] 空调用的合法性
                tool_calls_schema["minItems"] = 1

            schema["properties"]["tool_calls"] = tool_calls_schema

            tools = None
            tool_choice_val = None

        # 3. 原生 Tools 模式处理
        elif tools and not self.use_schema_tools:
            if require_tools:
                tool_choice_val = "required"

        # --- 兼容性检查与构建 Payload ---
        has_user = any(msg.get("role") == "user" for msg in messages)
        if not has_user:
            last_system_idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "system"),
                                   -1)
            if last_system_idx != -1:
                logger.warning("未在 messages 中发现 user 角色，将最后一个 system 消息重写为 user 角色以兼容本地模型。")
                messages[last_system_idx]["role"] = "user"
            else:
                # 极端异常情况：既没有 user 也没有 system
                error_msg = "LLM API 异常调用：messages 中既没有 user 也没有 system，请求被终止。"
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
            payload["tool_choice"] = tool_choice_val
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": True
                }
            }

        # --- 执行请求与洗稿 ---
        try:
            response = await client.chat.completions.create(**payload)
            result = response.model_dump()

            message = result.get("choices", [{}])[0].get("message", {})
            reasoning_content = message.get("reasoning_content")
            content_str = message.get("content") or ""
            native_tool_calls = message.get("tool_calls") or []

            # 底层错位容错
            if not content_str.strip() and not native_tool_calls and reasoning_content:
                message["content"] = reasoning_content
                content_str = reasoning_content
            else:
                message["content"] = content_str

            parsed_content = content_str
            unified_tool_calls = []

            # JSON 解析容错
            if schema and content_str:
                cleaned_str = content_str.strip()
                if "<think>" in cleaned_str and "</think>" in cleaned_str:
                    # 强行截断思考过程，只取最终输出的 JSON
                    cleaned_str = cleaned_str.split("</think>")[-1].strip()

                cleaned_str = cleaned_str.replace("```json", "").replace("```", "").strip()

                try:
                    parsed_content = json.loads(cleaned_str)
                except json.JSONDecodeError:
                    logger.warning(f"JSON Parsing failed for schema mode. Content: {content_str}")

            # 提取并归一化 Tool Calls
            if native_tool_calls:
                for tc in native_tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    # 尝试将原生字符串 args 转回 dict
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass

                    unified_tool_calls.append({
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "arguments": args
                    })
            elif isinstance(parsed_content, dict) and "tool_calls" in parsed_content:
                schema_calls = parsed_content.get("tool_calls", [])
                for tc in schema_calls:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass

                    unified_tool_calls.append({
                        "id": None,
                        "name": tc.get("name"),
                        "arguments": args
                    })

            # 最终统一返回格式
            return {
                "content": parsed_content,
                "tool_calls": unified_tool_calls,
                "raw_receive": message
            }

        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}", exc_info=True)
            logger.info("Payload: " + json.dumps(payload, ensure_ascii=False))

            # 发生错误时，强制关闭并重置 client，以便下一次请求重建连接
            if self.client:
                await self.client.close()
            self.client = None
            raise

    async def create_chat_completion_once(self, messages: str, system_prompt: str = None, model: str = None,
                                          schema: Dict = None, tools: list = None, require_tools: bool = False) -> dict:
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

        return await self.create_chat_completion(
            messages=msg, model=model, schema=schema, tools=tools, require_tools=require_tools
        )

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
