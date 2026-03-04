# core/tool_manager/mcp_client.py
import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, name: str, command: str, args: List[str], env: Dict[str, str] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()

        self.process: Optional[asyncio.subprocess.Process] = None
        self._msg_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._tools_cache: List[Dict[str, Any]] = []

    async def start(self):
        """启动 MCP Server 子进程"""
        logger.info(f"正在启动 MCP 服务器 [{self.name}]: {self.command} {self.args}")
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env
            )
            # 启动读取循环
            asyncio.create_task(self._read_stdout())
            asyncio.create_task(self._read_stderr())

            # 初始化协议
            await self._initialize()

            # 获取工具列表
            await self._refresh_tools()

        except Exception as e:
            logger.error(f"MCP服务器 [{self.name}] 启动失败: {e}", exc_info=True)
            raise

    async def _initialize(self):
        """发送 initialize 请求"""
        res = await self.send_request("initialize", {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {"name": "Aethel-Trinity", "version": "3.0.0"}
        })
        logger.debug(f"MCP [{self.name}] 初始化: {res}")

        # 发送 initialized 通知
        await self.send_notification("notifications/initialized", {})

    async def _refresh_tools(self):
        """获取工具列表"""
        res = await self.send_request("tools/list", {})
        self._tools_cache = res.get("tools", [])
        logger.info(f"MCP [{self.name}] 已加载 {len(self._tools_cache)} 个工具.")

    def get_tools(self) -> List[Dict[str, Any]]:
        """返回 OpenAI 格式的工具定义"""
        schemas = []
        for tool in self._tools_cache:
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"{self.name}__{tool['name']}",  # 命名空间隔离
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {})
                }
            })
        return schemas

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用工具"""
        # tool_name 格式为 server__tool
        real_tool_name = tool_name.split("__", 1)[1]

        res = await self.send_request("tools/call", {
            "name": real_tool_name,
            "arguments": arguments
        })

        # MCP 返回 content 列表
        content_list = res.get("content", [])
        text_results = [item["text"] for item in content_list if item["type"] == "text"]
        return "\n".join(text_results)

    async def send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """发送 JSON-RPC 请求"""
        self._msg_id += 1
        curr_id = self._msg_id
        future = asyncio.Future()
        self._pending_requests[curr_id] = future

        payload = {
            "jsonrpc": "2.0",
            "id": curr_id,
            "method": method,
            "params": params
        }

        data = json.dumps(payload).encode() + b"\n"
        self.process.stdin.write(data)
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            del self._pending_requests[curr_id]
            raise Exception(f"MCP 请求超时: {method}")

    async def send_notification(self, method: str, params: Dict[str, Any]):
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        data = json.dumps(payload).encode() + b"\n"
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def _read_stdout(self):
        """读取子进程输出并处理 JSON-RPC"""
        while True:
            line = await self.process.stdout.readline()
            if not line: break

            try:
                msg = json.loads(line.decode())
                if "id" in msg and msg["id"] in self._pending_requests:
                    # 响应
                    future = self._pending_requests.pop(msg["id"])
                    if "error" in msg:
                        future.set_exception(Exception(msg["error"]["message"]))
                    else:
                        future.set_result(msg.get("result"))
                else:
                    # 通知或请求 (暂不处理 Server 发来的请求)
                    pass
            except json.JSONDecodeError:
                logger.warning(f"MCP [{self.name}] 非JSON输出: {line}")

    async def _read_stderr(self):
        """读取子进程错误流"""
        while True:
            line = await self.process.stderr.readline()
            if not line: break
            logger.warning(f"MCP [{self.name}] STDERR: {line.decode().strip()}")
