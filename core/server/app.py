# core/server/app.py (完整更新)
import asyncio
import logging
import json
import uvicorn
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from core.infrastructure.config_loader import Config
from core.io.event_bus import EventBus
from core.io.event_schema import Action, OneBotEvent, EventType, DetailType, EventSource
from core.kernel.agent import AutonomousAgent

logger = logging.getLogger("WebServer")


class WebServer:
    def __init__(self, config: Config, event_bus: EventBus, agent: AutonomousAgent):
        self.config = config
        self.bus = event_bus
        self.agent = agent
        self.app = FastAPI(title="Aethel Overseer")

        # 允许跨域
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 懒加载 VectorStore
        self.vector_store = None

        self._register_routes()
        self._ws_clients: List[WebSocket] = []
        self.bus.subscribe_action(self._broadcast_action_to_ws)

    def _get_vector_store(self):
        if not self.vector_store:
            from core.memory.vector_store import VectorStore
            self.vector_store = VectorStore(self.agent.database, self.agent.api_client)
        return self.vector_store

    def _register_routes(self):

        @self.app.get("/", response_class=HTMLResponse)
        async def read_root():
            return "<h1>Aethel v3 Backend is Running</h1><p>请使用本地 GUI 客户端连接。</p>"

        @self.app.get("/api/state")
        async def get_state():
            return {
                "scratchpad": self.agent.scratchpad,
                "history_len": len(self.agent.history),
                "last_active": self.agent.last_interaction_time,
                "uptime": "running"
            }

        @self.app.get("/api/history")
        async def get_history():
            """[新增] 获取完整对话历史"""
            return {"history": self.agent.history}

        @self.app.post("/api/inject")
        async def inject_event(payload: Dict[str, Any]):
            msg = payload.get("message")
            if msg:
                event = OneBotEvent(
                    type=EventType.MESSAGE,
                    detail_type=DetailType.PRIVATE,
                    sub_type="gui",
                    source=EventSource(platform="gui", user_id="admin"),
                    message=msg,
                    alt_message=msg
                )
                self.bus.publish_event(event)
                return {"status": "ok"}
            return {"status": "error"}

        # --- 记忆管理 API ---

        @self.app.get("/api/memories")
        async def get_memories(type: str = "semantic", limit: int = 50, offset: int = 0):
            store = self._get_vector_store()
            data = await store.list_memories(type, "admin_console", limit, offset)  # 假设默认用户
            return {"items": data}

        @self.app.delete("/api/memories")
        async def delete_memory(id: str, type: str):
            """[新增] 删除记忆"""
            store = self._get_vector_store()
            success = await store.delete_memory(type, id, "admin_console")
            if success:
                return {"status": "ok"}
            raise HTTPException(status_code=404, detail="删除失败")

        @self.app.put("/api/memories")
        async def update_memory(payload: Dict[str, Any]):
            """[新增] 更新记忆内容"""
            id = payload.get("id")
            type = payload.get("type")
            content = payload.get("content")

            store = self._get_vector_store()
            success = await store.update_memory_content(type, id, content, "admin_console")
            if success:
                return {"status": "ok"}
            raise HTTPException(status_code=500, detail="更新失败")

        # --- WebSocket ---

        @self.app.websocket("/ws/logs")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except Exception:
                pass
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)

    async def _broadcast_action_to_ws(self, action: Action):
        if not self._ws_clients: return
        payload = {
            "type": action.action,
            "params": action.params,
            "target": action.target_platform
        }
        to_remove = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(payload)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self._ws_clients.remove(ws)

    async def run(self):
        port = 8000
        config = uvicorn.Config(self.app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        logger.info(f"后端 API 服务已启动: http://localhost:{port}")
        await server.serve()
