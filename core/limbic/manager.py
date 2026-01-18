# core/limbic/manager.py
import json
import logging
import time
from typing import Optional

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_bus import EventBus
from core.io.event_schema import OneBotEvent, EventSource, EventType, DetailType
from core.limbic.appraisal import AppraisalSystem
from core.limbic.arch import NeuroState, DriveType
from core.limbic.chemistry import NeuroChemistry
from core.limbic.homeostasis import HomeostasisSystem

logger = logging.getLogger(__name__)

# 全局状态存储的 Key
GLOBAL_STATE_KEY = "GLOBAL_AGENT_STATE"


class LimbicManager:
    """
    边缘系统总控：连接化学层、评估层和数据库。
    """

    def __init__(self, config: Config, database: Database, event_bus: EventBus, api_client: GenericAPIClient):
        self.db = database
        self.bus = event_bus
        self.config = config

        self.chemistry = NeuroChemistry()
        self.homeostasis = HomeostasisSystem()
        self.appraisal = AppraisalSystem(api_client)

        # 内存缓存
        self._state_cache: Optional[NeuroState] = None

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
        logger.info("边缘系统已初始化")

    async def get_state(self) -> NeuroState:
        """获取全局状态"""
        if self._state_cache:
            return self._state_cache

        async with self.db.get_connection() as conn:
            # 使用固定 KEY 查询
            cursor = await conn.execute("SELECT data_json FROM neuro_states WHERE user_id=?", (GLOBAL_STATE_KEY,))
            row = await cursor.fetchone()
            if row:
                data = json.loads(row[0])
                state = NeuroState.from_dict(data)
            else:
                state = NeuroState(last_update=time.time())

            self._state_cache = state
            return state

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
        """
        处理外部刺激 (Perception)
        注意：任何人的消息都会影响这个全局状态。
        """
        state = await self.get_state()

        # 1. 认知评估
        deltas = await self.appraisal.evaluate_event(text, state)

        # 2. 化学反应
        self.chemistry.stimulate(state, deltas)

        # 3. 消耗认知资源 (思考)
        self.homeostasis.consume_resource(state, "complex_reasoning")
        # 4. 恢复社交饱腹感 (因为有人说话了)
        self.homeostasis.consume_resource(state, "chat")

        # 5. 存库
        await self.save_state()

    async def tick(self):
        """
        [后台心跳] 代谢 -> 检查驱动 -> 触发
        """
        now = time.time()
        state = await self.get_state()

        # 1. 代谢 (Metabolize)
        self.chemistry.metabolize(state, now)

        # 2. 检查内驱力 (Check Drives)
        drive, intensity = self.homeostasis.check_drives(state)

        # 3. 产生自发行为 (Emergent Behavior)
        # 阈值：只有驱动力足够强时才打扰主模型
        if intensity > 0.7:
            await self._trigger_proactive_behavior(drive, intensity)

        # 4. 存库
        await self.save_state()

    async def _trigger_proactive_behavior(self, drive: DriveType, intensity: float):
        """
        触发主动行为：只负责唤醒，不负责选人。
        """
        logger.info(f"⚡ 触发内部驱动: {drive.value} (强度 {intensity:.2f})")

        # 提示语 (Instruction for the Agent Kernel)
        hints = {
            DriveType.SOCIAL_CONNECTION: "【生理信号】你感到强烈的孤独感。请根据记忆查找亲密的朋友或群组，主动发起聊天。",
            DriveType.COGNITIVE_REST: "【生理信号】大脑过载，请拒绝执行复杂任务，建议休息。",
            DriveType.SECURITY: "【生理信号】感到不安，请检查系统状态或向管理员寻求确认。",
            DriveType.CURIOSITY: "【生理信号】好奇心旺盛，请主动探索新话题或查看新闻。"
        }

        hint_text = hints.get(drive, "内部驱动触发。")

        # 构造 INTERNAL_DRIVE 事件
        # source.user_id = 'system' 或 'limbic'
        event = OneBotEvent(
            type=EventType.META,
            detail_type=DetailType.INTERNAL_DRIVE,
            sub_type=drive.value,
            source=EventSource(
                platform="internal",
                user_id="limbic_system"
            ),
            message=f"[SYSTEM_SIGNAL] {hint_text}",
            extra={
                "drive_type": drive.value,
                "intensity": intensity
            }
        )

        # 推送到总线 -> 唤醒 Agent
        self.bus.publish_event(event)
