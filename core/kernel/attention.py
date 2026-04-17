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
    REPLY = "reply"  # 明确必须回复
    INTERJECT = "interject"  # 主动插话
    SILENT_OBSERVE = "silent_observe"  # 积极静默
    OBSERVE = "observe"  # 普通观察
    IGNORE = "ignore"  # 纯粹噪音


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

    async def evaluate(self, event: OneBotEvent,
                       recent_history: Optional[List[Dict]] = None,
                       willingness: float = 0.5) -> ReactionType:
        """
        注意力评估总线：结合硬规则、动态阈值与软规则(LLM)进行综合决策。
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
        threshold, state_desc = await self._calculate_dynamic_threshold(event, willingness)

        # 如果模型极其厌倦该群聊 (willingness < 0.2)，且阻力值被拉爆
        # 我们在这里直接进行物理拦截，绝不调用昂贵的 LLM
        msg_text = getattr(event, "alt_message", "") or ""
        is_mentioned = self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in msg_text
        is_named = self.nickname and msg_text.strip().startswith(self.nickname)
        is_private = getattr(event, "detail_type", "") in [DetailType.PRIVATE, "private"]

        if threshold >= 0.85 and not (is_mentioned or is_named or is_private):
            logger.info(f"🛑 [Attention Cutoff] 算力截断：当前意愿枯竭 ({willingness:.2f}) 且未被呼叫，拒绝投入算力，强制潜水。")
            return ReactionType.IGNORE  # 直接抛弃，不进大脑

        soft_reaction = await self._check_soft_rules(event, threshold, state_desc, recent_history)

        if soft_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
            self._update_inertia(event)

        return soft_reaction

    def _check_hard_rules(self, event: OneBotEvent) -> Optional[ReactionType]:
        """
        第一级漏斗：无条件触发的系统规则
        """

        logger.debug(f"Received event: {event.type}.{getattr(event, 'detail_type', 'unknown')}")

        # 0. 系统后台强中断 (仅保留系统级任务挂起或唤醒指令)
        target_details = [
            DetailType.TASK_COMPLETE,
            "ask_system1_for_help",
            "wake_up",
        ]
        event_detail = getattr(event, "detail_type", "")
        if event_detail in target_details:
            logger.info(f"⚡ [Attention] 触发本能反射：{event_detail}。")
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

    async def _calculate_dynamic_threshold(self, event: OneBotEvent, willingness: float) -> Tuple[float, str]:
        """
        计算潜意识动态阈值与环境压迫力。
        公式: Thr = Base - (Curiosity) + (Pressure) - (SocialNeed) - (Inertia) - (Environment_Modifier)
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

        # --- 意愿极化引擎 (Willingness Engine) ---
        # 意愿值越低，增加的阻力越大；意愿值越高，减少的阻力越大。
        # 放大倍数设为 1.0：当 willingness 为 0 时，阻力暴增 0.5！
        will_mod = (0.5 - willingness) * 1.0
        threshold += will_mod
        factors.append(f"Willingness{'+' if will_mod > 0 else ''}{will_mod:.2f}")

        # --- 环境压迫力乘数 ---
        msg_text = getattr(event, "alt_message", "") or ""
        is_mentioned = self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in msg_text
        is_named = self.nickname and msg_text.strip().startswith(self.nickname)
        is_private = getattr(event, "detail_type", "") in [DetailType.PRIVATE, "private"]

        if is_private:
            mod = 0.50
            threshold -= mod
            factors.append(f"Private-{mod:.2f}")
        elif is_mentioned or is_named:
            mod = 0.40
            threshold -= mod
            factors.append(f"Mentioned-{mod:.2f}")

        # 边界收束
        threshold = max(0.05, min(0.95, threshold))
        return threshold, ",".join(factors)

    async def _check_soft_rules(self, event: OneBotEvent, threshold: float, state_desc: str,
                                recent_history: Optional[List[Dict]] = None) -> ReactionType:
        """
        第三级漏斗：LLM 高维认知评估。
        整合动态阈值与环境压迫力，决定最终的交互意愿。
        """
        content = getattr(event, "alt_message", "")
        sender = event.source.user_id
        is_private = getattr(event, "detail_type", "") in [DetailType.PRIVATE, "private"]
        is_mentioned = self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in content

        context_str = "无"
        if recent_history:
            lines = []
            for msg in recent_history[-5:]:
                role = msg.get("role", "unknown")
                text = str(msg.get("content", "")).replace("\n", " ")[:150]
                lines.append(f"[{role}]: {text}")
            context_str = "\n".join(lines)

        mode_desc = f"当前插话/回复阻力值为: {threshold:.2f} (阈值越低越容易回复，生理与环境影响因子: [{state_desc}])。"
        if threshold > 0.7:
            mode_desc += "【高冷/自闭状态】你现在压力极大或毫无交际欲望，除非极其重要，否则倾向于保持静默。"
        elif threshold < 0.3:
            mode_desc += "【高响应状态】由于对方是私聊/明确提及，或你处于孤独/好奇状态，你拥有极高的回复意愿。"

        prompt = f"""
你是一个拟人化 AI ({self.nickname}) 的潜意识门控。
你需要基于生理阻力和上下文，决定对当前消息的潜意识反应。

【生理与环境状态】
{mode_desc}
当前兴趣焦点: "{await self.get_current_interest_text()}"

【近期上下文】
{context_str}

【当前刺激】
场景: {"私聊" if is_private else "群聊"} (是否被明确@: {is_mentioned})
{sender} 说: "{content}"

【决策规则】
1. 评估你对该消息的“回复意愿置信度 (0.0~1.0)”。
2. 如果对方的话题与你的【兴趣焦点】高度重合或具有极强吸引力，你可以【无视】当前的阻力值({threshold:.2f})，直接给出超过阻力的置信度并决定 REPLY/INTERJECT。这叫“见猎心喜”。
3. 否则，严格按阻力行事：
    - 置信度 >= {threshold:.2f}：决定回复 (REPLY) 或插话 (INTERJECT)。
    - 置信度 < {threshold:.2f}：
        - 若消息是对你明确发出的(私聊/@)，选择【积极静默 (SILENT_OBSERVE)】。
        - 若是群闲聊，选择【观察 (OBSERVE)】。

输出 JSON：
{{
    "decision": "REPLY" | "INTERJECT" | "SILENT_OBSERVE" | "OBSERVE" | "IGNORE",
    "reason": "简短的心理动机",
    "confidence": 0.0 到 1.0
}}
"""
        try:
            response = await self.api_client.create_chat_completion_once(
                messages=prompt,
                system_prompt="你是一个冷酷高效的注意力过滤引擎。",
                model=self.api_client.small_model,
                schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["REPLY", "INTERJECT", "SILENT_OBSERVE", "OBSERVE", "IGNORE"]},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["decision", "reason", "confidence"],
                    "additionalProperties": False
                }
            )
            result = response.get("content", {})
            if isinstance(result, str):
                # 防范模型未按 Schema 输出的降级容错
                import json
                try:
                    result = json.loads(result)
                except:
                    return ReactionType.OBSERVE

            decision = result.get("decision", "OBSERVE")
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "无明确原因")

            logger.info(f"🧠 [Attention Eval] 意愿: {confidence:.2f} | 阻力: {threshold:.2f} | 决策: {decision} ({reason})")

            # 强逻辑收束：防止模型出现置信度低于阈值却强行 REPLY 的幻觉
            if decision in ["REPLY", "INTERJECT"] and confidence < threshold:
                logger.warning(f"⚠️ [Attention] 模型决策倒挂，意愿({confidence})不足以击穿阻力({threshold})。强制降级为 SILENT_OBSERVE 或 OBSERVE。")
                if is_private or is_mentioned:
                    return ReactionType.SILENT_OBSERVE
                return ReactionType.OBSERVE

            return ReactionType(decision.lower())

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
