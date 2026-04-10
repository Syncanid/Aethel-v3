# core/kernel/attention.py
import json
import logging
import time
from enum import Enum
from typing import Optional, List, Tuple, Dict

from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.io.event_schema import OneBotEvent, EventType, DetailType
from core.kernel.prompt import PromptManager
from core.limbic.arch import NeuroState

logger = logging.getLogger(__name__)

# 全局状态 Key (需与 LimbicManager 保持一致)
GLOBAL_STATE_KEY = "GLOBAL_AGENT_STATE"
# 兴趣存储 Key
INTEREST_STORE_KEY = "CURRENT_INTEREST_VECTOR"


class ReactionType(str, Enum):
    REPLY = "reply"  # 必须回复 (直接交互/强相关/系统任务)
    INTERJECT = "interject"  # 主动插话 (弱交互/熟人特权/极度感兴趣)
    OBSERVE = "observe"  # 静默观察 (无价值/插话阈值过高)
    IGNORE = "ignore"  # 完全忽略 (黑名单/绝对不想理睬)


class AttentionFilter:
    def __init__(self, config: Config, prompt: PromptManager, api_client: GenericAPIClient, database: Database):
        self.config = config
        self.api_client = api_client
        self.db = database
        self.prompt = prompt

        # 加载身份配置
        self.bot_self_id = str(config.get("system.bot_self_id", ""))
        self.nickname = prompt.role_data.get("identity", {}).get("name", "Aethel")

        # --- 内部状态 ---
        self._cached_interest_vector: Optional[List[float]] = None
        self._cached_interest_text: str = ""
        self._last_interest_update = 0.0

        # 记录不同上下文 (私聊/群聊) 的最后一次回复时间，用于计算“对话惯性”
        self.active_conversations: Dict[str, float] = {}

        # 基础阈值 (阈值越低越容易触发插话)
        self.BASE_THRESHOLD = 0.65

    def _get_context_id(self, event: OneBotEvent) -> str:
        """获取当前事件的上下文 ID"""
        if event.source.group_id:
            return f"group_{event.source.group_id}"
        return f"private_{event.source.user_id}"

    def _update_inertia(self, event: OneBotEvent):
        """更新对话惯性时间戳"""
        ctx_id = self._get_context_id(event)
        self.active_conversations[ctx_id] = time.time()

    async def evaluate(self, event: OneBotEvent, recent_history: Optional[List[Dict]] = None) -> ReactionType:
        """
        [主入口] 三级漏斗式注意力过滤网
        """
        # ==========================================
        # 第一级漏斗：绝对本能反射 (Hard Rules)
        # ==========================================
        hard_reaction = self._check_hard_rules(event)
        if hard_reaction:
            if hard_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
                self._update_inertia(event)
            return hard_reaction

        # 过了硬规则后，只有 MESSAGE 事件才值得继续分析
        if event.type != EventType.MESSAGE:
            return ReactionType.IGNORE

        # ==========================================
        # 第二级漏斗：社交亲密度门控 (Social Gate)
        # ==========================================
        social_reaction = await self._check_social_gate(event)
        if social_reaction:
            if social_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
                self._update_inertia(event)
            return social_reaction

        # ==========================================
        # 第三级漏斗：边缘系统驱动的动态评估 (LLM Soft Rules)
        # ==========================================
        dynamic_threshold, state_desc = await self._calculate_dynamic_threshold(event)
        soft_reaction = await self._check_soft_rules(event, dynamic_threshold, state_desc, recent_history)

        if soft_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
            self._update_inertia(event)

        return soft_reaction

    def _check_hard_rules(self, event: OneBotEvent) -> Optional[ReactionType]:
        """第一级漏斗：无条件触发的系统规则"""

        # 0. 系统后台强中断 (绝对优先，解决拦截任务更新的痛点)
        if getattr(event, "type", "") in [EventType.TASK, "task"] or getattr(event, "detail_type", "") in [
            DetailType.TASK_PROGRESS, DetailType.TASK_COMPLETE, "ask_system1_for_help", "task_update", "task_dispatch"
        ]:
            logger.info("⚡ [Attention] 触发本能反射：接收到后台任务调度/更新。")
            return ReactionType.REPLY

        if getattr(event, "detail_type", "") == DetailType.INTERNAL_DRIVE:
            logger.info("⚡ [Attention] 触发本能反射：生理驱动力告警。")
            return ReactionType.REPLY

        # 仅限文本消息的检查
        if event.type != EventType.MESSAGE:
            return None

        msg_text = getattr(event, "alt_message", "") or ""

        # 1. 明确提及 (@我)
        if self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in msg_text:
            return ReactionType.REPLY

        # 2. 呼叫名字开头
        if self.nickname and msg_text.strip().startswith(self.nickname):
            return ReactionType.REPLY

        # 3. 私聊强制接管 (私聊中没有插话概念，所有消息必须被看到)
        if getattr(event, "detail_type", "") in [DetailType.PRIVATE, "private"]:
            return ReactionType.REPLY

        return None

    async def _check_social_gate(self, event: OneBotEvent) -> Optional[ReactionType]:
        """第二级漏斗：熟人特权与黑名单过滤"""
        puid = f"{event.source.platform}:{event.source.user_id}"

        async with self.db.get_connection() as conn:
            cursor = await conn.execute("SELECT data_json FROM social_users WHERE puid=?", (puid,))
            row = await cursor.fetchone()

        if not row:
            return None  # 陌生人，放入下一级 LLM 评估

        user_data = json.loads(row[0])
        trust = user_data.get("trust", 0.0)
        favorability = user_data.get("favorability", 0.0)
        intimacy = user_data.get("intimacy", 0.0)

        # 1. 厌恶/黑名单过滤 (好感度或信任度极低)
        if favorability < -30 or trust < -50:
            logger.info(f"🛑 [Attention] 社交门控：静默过滤厌恶用户 ({puid})")
            return ReactionType.IGNORE

        # 2. 挚友/管理员插话特权 (如果在群聊中，且关系极好，给予极高概率免 LLM 插话)
        if event.source.group_id:
            score = (trust + favorability + intimacy) / 300.0  # 粗略归一化到 0~1
            if score > 0.8:
                logger.info(f"✨ [Attention] 社交门控：高亲密度特权放行 ({puid})")
                return ReactionType.INTERJECT

        return None

    async def _calculate_dynamic_threshold(self, event: OneBotEvent) -> Tuple[float, str]:
        """
        计算潜意识动态阈值，影响 LLM 的插话敏感度。
        公式: Thr = Base - (Curiosity * 0.2) + (SurvivalPressure * 0.3) - (SocialNeed * 0.3) - (Inertia)
        """
        state = await self._get_neuro_state()
        threshold = self.BASE_THRESHOLD
        factors = []

        # A. 探索欲 (好奇心) -> 降低阈值 (更愿意参与新话题，话痨)
        if state.curiosity > 0.5:
            mod = (state.curiosity - 0.5) * 0.4
            threshold -= mod
            factors.append(f"Curious-{mod:.2f}")

        # B. 生存压力 (高负载/报错) -> 大幅提高阈值 (变得自闭/高冷，不想理会普通闲聊)
        if state.survival_pressure > 0.5:
            mod = (state.survival_pressure - 0.5) * 0.6
            threshold += mod
            factors.append(f"Stressed+{mod:.2f}")

        # C. 社交渴望 (孤独) -> 降低阈值 (极度渴望聊天)
        if state.social_need > 0.5:
            mod = (state.social_need - 0.5) * 0.5
            threshold -= mod
            factors.append(f"Lonely-{mod:.2f}")

        # D. 对话惯性 (Recently Active) -> 降低阈值 (保持聊天连贯)
        ctx_id = self._get_context_id(event)
        last_time = self.active_conversations.get(ctx_id, 0)
        time_diff = time.time() - last_time
        if time_diff < 60:
            mod = 0.25 * (1 - (time_diff / 60))
            threshold -= mod
            factors.append(f"Inertia-{mod:.2f}")

        threshold = max(0.1, min(0.95, threshold))
        return threshold, ",".join(factors)

    async def _check_soft_rules(self, event: OneBotEvent, threshold: float, state_desc: str,
                                recent_history: Optional[List[Dict]] = None) -> ReactionType:
        """
        第三级漏斗：LLM 动态评估。
        此时已经排除了明确呼叫和系统事件，主要用于判断是否要在群聊中“主动插话”。
        """
        content = getattr(event, "alt_message", "")
        sender = event.source.user_id

        context_str = "无"
        if recent_history:
            lines = []
            for msg in recent_history[-4:]:  # 只取最近 4 条，防止干扰
                role = msg.get("role", "unknown")
                text = str(msg.get("content", "")).replace("\n", " ")[:100]
                lines.append(f"[{role}]: {text}")
            context_str = "\n".join(lines)

        if threshold > 0.7:
            mode_desc = "高冷/自闭模式：你现在压力很大或很专心，除非话题极其重要或有强烈的情绪共鸣，否则保持沉默。"
        elif threshold < 0.4:
            mode_desc = "话痨/渴望模式：你现在精力旺盛或感到孤独，即使是普通的群聊也积极寻找话题切入点。"
        else:
            mode_desc = "标准模式：按正常逻辑判断是否接话。"

        prompt = f"""
你是一个拟人化 AI 助手 ({self.nickname}) 的潜意识注意力门控。
当前群聊中有人发了一条消息，你没有被提及。你需要决定是否要“主动插话”。

【当前生理状态】
{mode_desc} (插话阻力值: {threshold:.2f}，越低越容易插话。生理影响因子: [{state_desc}])
当前关注点: "{await self.get_current_interest_text()}"

【近期上下文】
{context_str}

【当前消息】
{sender} 说: "{content}"

【决策要求】
分析该消息是否触及了你的“关注点”，或者是否有强烈的情绪需要你安抚。
严格输出 JSON，不要任何多余内容：
{{
    "decision": "INTERJECT" | "OBSERVE",
    "reason": "为什么插话或为什么无视（少于15字）",
    "confidence": 0.0 到 1.0 之间的浮点数 (插话意愿有多强)
}}
"""
        try:
            response = await self.api_client.create_chat_completion_once(
                messages=prompt,
                system_prompt="你是一个冷酷高效的决策引擎。",
                model=self.api_client.small_model,
                schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["INTERJECT", "OBSERVE"]},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["decision", "reason", "confidence"],
                    "additionalProperties": False
                }
            )
            result = response.get("content", {})
            if isinstance(result, str):
                return ReactionType.OBSERVE
            decision = result.get("decision", "OBSERVE")
            confidence = result.get("confidence", 0.0)

            if decision == "INTERJECT":
                # 与动态生理阈值对抗
                if confidence >= threshold:
                    logger.info(
                        f"🗣️ [Attention] 决定插话: 置信度 {confidence:.2f} >= 阻力 {threshold:.2f} ({result.get('reason')})")
                    return ReactionType.INTERJECT
                else:
                    logger.info(f"🛑 [Attention] 放弃插话: 意愿 {confidence:.2f} 不足以克服当前阻力 {threshold:.2f}")
                    return ReactionType.OBSERVE

            return ReactionType.OBSERVE

        except Exception as e:
            logger.error(f"Attention LLM check failed: {e}")
            return ReactionType.OBSERVE

    async def get_current_interest_text(self) -> str:
        """
        公开接口：获取当前兴趣文本
        如果尚未加载，会触发一次 DB 读取
        """
        if not self._cached_interest_text:
            # 触发加载逻辑
            await self._get_current_interest_vector()
        return self._cached_interest_text or "General Assistant"

    async def update_interest(self, new_interest_text: str):
        """
        [公开接口] 更新 Agent 的关注点
        通常由 Tool 调用 (如 self_reflect)
        """
        logger.info(f"🔄 更新注意力关注点: {new_interest_text}")
        vec = await self.api_client.create_embedding(new_interest_text)
        if vec:
            async with self.db.get_connection() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO preference_store (key, value, updated_at) VALUES (?, ?, ?)",
                    (INTEREST_STORE_KEY, json.dumps({"text": new_interest_text, "vector": vec}), time.time())
                )
                await conn.commit()
            # 刷新缓存
            self._cached_interest_vector = vec
            self._cached_interest_text = new_interest_text
            self._last_interest_update = time.time()

    async def _get_current_interest_vector(self) -> List[float]:
        """获取兴趣向量 (带缓存)"""
        # 缓存有效性检查 (例如每 5 分钟强制刷新一次，或永久缓存直到 update)
        if self._cached_interest_vector:
            return self._cached_interest_vector

        # 查库
        async with self.db.get_connection() as conn:
            cursor = await conn.execute("SELECT value FROM preference_store WHERE key=?", (INTEREST_STORE_KEY,))
            row = await cursor.fetchone()

            if row:
                data = json.loads(row[0])
                self._cached_interest_vector = data["vector"]
                self._cached_interest_text = data.get("text", "")
            else:
                # 初始化默认兴趣
                default_interest = "technology, ai, python programming, video games, casual chat"
                await self.update_interest(default_interest)

        return self._cached_interest_vector

    async def _get_neuro_state(self) -> NeuroState:
        """从数据库读取最新的神经状态"""
        async with self.db.get_connection() as conn:
            try:
                cursor = await conn.execute("SELECT data_json FROM neuro_states WHERE user_id=?", (GLOBAL_STATE_KEY,))
                row = await cursor.fetchone()
                if row:
                    return NeuroState.from_dict(json.loads(row[0]))
            except Exception:
                pass
        return NeuroState()
