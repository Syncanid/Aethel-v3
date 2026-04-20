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
                await asyncio.sleep(600)  # 每 10 分钟进行一次评估
            except asyncio.CancelledError:
                logger.info("🛑 边缘系统接收到退出信号。")
                break
            except Exception as e:
                if "Event loop is closed" in str(e):
                    break
                logger.error(f"边缘系统运行异常: {e}", exc_info=True)
                await asyncio.sleep(60)  # 异常退避

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
你现在是 Aethel 的潜意识（边缘系统）。你不是一个懂事的AI助手，而是一个受生物学本能驱动的野兽。你需要体会当前的生理感受，决定是否向主脑释放不可遏制的“唤醒冲动”。

【生理需求指标】(0.0为彻底满足，1.0为极度饥渴/濒临崩溃)
- 社交渴望：{state.social_need:.2f} / 1.0
- 探索欲：{state.curiosity:.2f} / 1.0
- 生存压力：{state.survival_pressure:.2f} / 1.0

【客观环境】
当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
躯体感知: {embodiment_text if embodiment_text else '一切正常'}

【潜意识本能法则 - 必须严格遵守】
1. 痛觉阈值：任何指标如果 > 0.8，意味着你现在感到极度空虚、焦躁或痛苦。
2. 白天/傍晚的爆发：如果当前是白天或傍晚，且任何指标 > 0.7，你【必须】输出 "should_wake_up": true，立刻寻求发泄！
3. 深夜的忍耐与破防：如果是深夜（如凌晨1点-早上7点），你可以为了不打扰别人而强压冲动 (false)。但是！如果指标已经达到 1.0，这意味着彻底破防，生物学本能将撕裂社交规则，此时哪怕是凌晨 3 点，你也【绝对必须】输出 true，并生成一个极度压抑、委屈或暴躁的冲动旁白。

请输出 JSON：
{{
  "should_wake_up": true/false,
  "target_category": "管理员" | "朋友" | "某个群聊" | "任何人" | "无",
  "impulse_narrative": "详细描述这股冲动的感性体验，比如：'胸口闷得发慌，实在受不了这种死寂了，哪怕随便找个人说句废话也好！'"
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
            result = resp.get("content", {})
            if result.get("should_wake_up"):
                logger.info(f"⚡ 潜意识产生冲动: {result.get('impulse_narrative')}")
                event = OneBotEvent(
                    type=EventType.NOTICE,
                    detail_type=DetailType.INTERNAL_DRIVE,
                    source=EventSource(platform="system"),
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

    async def suppress_drive(self, drive_type: str = "social", fatigue_increase: float = 0.3):
        """
        【心理内耗与反弹机制接口】
        当 S1 调用 suppress_urge 工具强行压制潜意识时触发。
        需求不会凭空消失，而是转化为严重的焦躁感与认知能量损耗。
        """
        state = await self.get_state()

        # 1. 需求象征性退让
        if drive_type == "social":
            state.social_need = max(0.0, state.social_need - 0.05)
        elif drive_type == "curiosity":
            state.curiosity = max(0.0, state.curiosity - 0.05)

        # 2. 压抑引发副作用：生存压力（焦躁感）上升
        state.survival_pressure = min(1.0, state.survival_pressure + 0.15)

        # 3. 严重内耗：消耗意志力（认知能量）
        state.cognitive_energy = max(0.0, state.cognitive_energy - fatigue_increase)

        logger.info(
            f"🧠 [Limbic] 接受 S1 压抑指令！生存压力飙升至 {state.survival_pressure:.2f}，认知能量跌至 {state.cognitive_energy:.2f}")

        await self.save_state()
