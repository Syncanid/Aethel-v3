# core/limbic/manager.py
import asyncio
import json
import logging
import time
from typing import Optional

from core.gui.monitor_registry import monitor_registry
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventSource, EventType, DetailType
from core.limbic.arch import NeuroState
from core.limbic.embodiment import HardwareEmbodiment

logger = logging.getLogger(__name__)

# 全局状态存储的 Key
GLOBAL_STATE_KEY = "GLOBAL_AGENT_STATE"


class LimbicManager:
    """
    边缘系统总控
    """

    def __init__(self, config: Config, database: Database, event_bus: EventBus, api_client: GenericAPIClient):
        self.db = database
        self.bus = event_bus
        self.config = config
        self.api_client = api_client
        self.embodiment = HardwareEmbodiment(config)
        self._state_cache: Optional[NeuroState] = None
        self.is_running = False

    async def start(self):
        """启动检测循环"""
        self.is_running = True
        logger.info("边缘系统已启动...")

        while self.is_running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"边缘系统运行异常: {e}", exc_info=True)
            await asyncio.sleep(600)  # 每 10 分钟进行一次评估

    async def initialize(self):
        """初始化表结构"""
        async with self.db.get_connection() as conn:
            await conn.execute("""
                               CREATE TABLE IF NOT EXISTS neuro_states
                               (
                                   user_id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   data_json
                                   TEXT,
                                   last_update
                                   REAL
                               )
                               """)
            await conn.commit()
        # 预加载状态
        await self.get_state()

    async def get_state(self) -> NeuroState:
        """获取全局状态"""
        if self._state_cache:
            return self._state_cache

        async with self.db.get_connection() as conn:
            # 使用固定 KEY 查询
            cursor = await conn.execute("SELECT data_json FROM neuro_states WHERE user_id=?", (GLOBAL_STATE_KEY,))
            row = await cursor.fetchone()
            if row:
                self._state_cache = NeuroState.from_dict(json.loads(row[0]))
            else:
                self._state_cache = NeuroState(last_update=time.time())
            return self._state_cache

    async def save_state(self):
        """持久化全局状态"""
        if not self._state_cache: return
        state = self._state_cache

        async with self.db.get_connection() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO neuro_states (user_id, data_json, last_update) VALUES (?, ?, ?)",
                (GLOBAL_STATE_KEY, json.dumps(state.to_dict()), state.last_update)
            )
            await conn.commit()

    # --- 核心逻辑 ---

    async def process_stimulus(self, text: str):
        """当收到外部真实人类发言刺激时，满足部分需求"""
        state = await self.get_state()

        # 受到外部真实消息刺激，缓解社交渴望
        state.social_need = max(0.0, state.social_need - 0.01)

        # 听到别人说话，稍微满足一点探索欲（获取了新信息）
        state.curiosity = max(0.0, state.curiosity - 0.05)

        # 接收和理解外界刺激会带来极微小的脑力消耗
        state.cognitive_energy = max(0.0, state.cognitive_energy - 0.01)

        await self.save_state()

    async def _tick(self):
        now = time.time()
        state = await self.get_state()

        # 1. 简单的线性需求累积
        delta_hours = (now - state.last_update) / 3600.0 if state.last_update > 0 else 0
        state.social_need = min(1.0, state.social_need + delta_hours * 0.1)
        state.curiosity = min(1.0, state.curiosity + delta_hours * 0.05)
        state.last_update = now

        # 2. 硬件感知同步
        self.embodiment.sync_hardware_to_state(state)
        embodiment_text = self.embodiment.get_status_narrative()

        # 注册监控 (UI面板)
        monitor_registry.register_metric_source("社交渴望", lambda: state.social_need)
        monitor_registry.register_metric_source("探索欲", lambda: state.curiosity)
        monitor_registry.register_metric_source("生存压力", lambda: state.survival_pressure)
        monitor_registry.register_metric_source("认知能量", lambda: state.cognitive_energy)

        # 基础拦截：如果毫无冲动，就不唤醒小模型了
        if state.social_need < 0.4 and state.survival_pressure < 0.3 and state.curiosity < 0.4:
            await self.save_state()
            return

        # 3. 潜意识评估 (小模型驱动)
        prompt = f"""
你现在是 Aethel 的潜意识（边缘系统）。你需要体会当前的生理感受，决定是否要向主脑 (System 1) 释放一个“自发唤醒”的冲动。

【当前时间】: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
【内部需求】: 社交渴望({state.social_need:.2f}/1.0), 探索欲({state.curiosity:.2f}/1.0), 生存压力({state.survival_pressure:.2f}/1.0)
【躯体感知】: {embodiment_text if embodiment_text else '一切正常'}

请结合时间和状态进行感性评估。如果是半夜且没有重大生存危机，尽量不要产生冲动。
请输出 JSON：
{{
  "should_wake_up": true/false,
  "target_category": "管理员" | "朋友" | "某个群聊" | "任何人" | "无",
  "impulse_narrative": "例如：深夜服务器负载突然升高，你感到非常恐慌，急切地想向管理员报告这个危机。"
}}
        """
        try:
            resp = await self.api_client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.api_client.small_model,
                schema={
                    "type": "object",
                    "properties": {
                        "should_wake_up": {"type": "boolean"},
                        "target_category": {"type": "string"},
                        "impulse_narrative": {"type": "string"}
                    },
                    "required": ["should_wake_up", "target_category", "impulse_narrative"],
                    "additionalProperties": False
                }
            )
            result = json.loads(resp["choices"][0]["message"]["content"])

            if result.get("should_wake_up"):
                logger.info(f"⚡ 潜意识产生冲动: {result.get('impulse_narrative')}")
                event = OneBotEvent(
                    type=EventType.NOTICE,
                    detail_type=DetailType.INTERNAL_DRIVE,
                    source=EventSource(platform="internal_limbic"),
                    extra={
                        "target_category": result.get("target_category", "任何人"),
                        "narrative": result.get("impulse_narrative", "产生了一股不可名状的冲动。")
                    }
                )
                self.bus.publish_event(event)

        except Exception as e:
            logger.error(f"潜意识小模型评估失败: {e}")

        await self.save_state()

    async def consume_action_energy(self, action_type: str):
        """模拟人类行为的资源消耗与需求满足"""
        state = await self.get_state()

        if action_type == "chat":
            # 交流缓解社交渴望
            state.social_need = max(0.0, state.social_need - 0.2)
            # 交流消耗微量认知能量
            state.cognitive_energy = max(0.0, state.cognitive_energy - 0.02)
            # 满足一点点探索欲
            state.curiosity = max(0.0, state.curiosity - 0.05)

        elif action_type == "complex_reasoning":
            # 深度思考/执行任务消耗大量认知能量
            state.cognitive_energy = max(0.0, state.cognitive_energy - 0.15)
            # 解决问题能大幅满足探索欲
            state.curiosity = max(0.0, state.curiosity - 0.3)

        elif action_type == "rest":
            # 休息恢复认知能量
            state.cognitive_energy = min(1.0, state.cognitive_energy + 0.3)

        await self.save_state()
