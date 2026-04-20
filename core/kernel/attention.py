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
        platform_name = getattr(event.source, "platform", "unknown")
        if event.source.group_id:
            return f"group_{platform_name}:{event.source.group_id}"
        return f"private_{platform_name}:{event.source.user_id}"

    def _update_inertia(self, event: OneBotEvent):
        """更新对话惯性时间戳"""
        ctx_id = self._get_context_id(event)
        self.active_conversations[ctx_id] = time.time()

    async def evaluate(self, event: OneBotEvent,
                       recent_history: Optional[List[Dict]] = None,
                       willingness: float = 0.5) -> Tuple[ReactionType, str, float]:
        """
        注意力评估总线：结合硬规则、动态阈值与软规则(LLM)进行综合决策。
        返回：(反应类型, 建议行动/态度, 意愿偏移量)
        """
        # ==========================================
        # 第一级漏斗：绝对本能反射 (Hard Rules)
        # ==========================================
        hard_reaction = self._check_hard_rules(event)
        if hard_reaction:
            if hard_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
                self._update_inertia(event)
            return hard_reaction, "系统级本能驱动的绝对反应", 0.0

        # 过了硬规则后，只有 MESSAGE 事件才值得继续分析
        if event.type != EventType.MESSAGE:
            return ReactionType.IGNORE, "非消息事件环境噪音，忽略", 0.0

        # ==========================================
        # 第二级漏斗：社交亲密度门控 (Social Gate)
        # ==========================================
        social_reaction = await self._check_social_gate(event)
        if social_reaction:
            if social_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
                self._update_inertia(event)
            return social_reaction, "触发社交门控机制", 0.0

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
            logger.info(f"🛑 [Attention Cutoff] 算力截断：当前意愿枯竭 ({willingness:.2f}) 且未被呼叫，强制潜水。")
            return ReactionType.IGNORE, "意愿枯竭，强制休眠", 0.0

        soft_reaction, should_do, will_shift = await self._check_soft_rules(event, threshold, state_desc,
                                                                            recent_history)

        if soft_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
            self._update_inertia(event)

        return soft_reaction, should_do, will_shift

    def _check_hard_rules(self, event: OneBotEvent) -> Optional[ReactionType]:
        """
        第一级漏斗：无条件触发的系统规则
        """

        logger.debug(f"Received event: {event.type}.{getattr(event, 'detail_type', 'unknown')}")

        # 0. 系统后台强中断
        target_details = [
            DetailType.INTERNAL_DRIVE,
            DetailType.TASK_COMPLETE,
            "ask_system1_for_help",
            "wake_up",
        ]
        event_detail = getattr(event, "detail_type", "")
        if event_detail in target_details:
            logger.info(f"⚡ [Attention] 触发本能反射：{event_detail}。")
            return ReactionType.INTERJECT

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
            msg_text = getattr(event, "alt_message", "") or ""
            is_mentioned = self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in msg_text
            is_named = self.nickname and msg_text.strip().startswith(self.nickname)
            is_private = getattr(event, "detail_type", "") in [DetailType.PRIVATE, "private"]

            # 评估是否为“强侵入”
            if is_private or is_mentioned or is_named:
                logger.warning(f"⚠️ [Attention] 社交门控：检测到敌对目标 ({puid}) 的强侵入，已注入预警并放行。")
                # 物理劫持：强行在输入的最前端打上系统烙印，迫使 LLM 改变对该事件的定性
                hijack_warning = f"【SYSTEM OVERRIDE: 这是一个你极其厌恶或毫不信任的实体。当前信任度:{trust}, 好感度:{favorability}。你拥有绝对防卫权，请展现冷漠、敌意或发起防卫反击】\n"
                event.alt_message = hijack_warning + msg_text
                return None  # 放行至第三级漏斗 (LLM) 进行反击决策
            else:
                logger.info(f"🛑 [Attention] 社交门控：静默过滤厌恶用户 ({puid}) 的环境噪音。")
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
                                recent_history: Optional[List[Dict]] = None) -> Tuple[ReactionType, str, float]:
        """
        第三级漏斗：LLM 高维认知评估。
        整合动态阈值与环境压迫力，决定最终的交互意愿。
        """
        content = getattr(event, "alt_message", "")
        sender = event.source.user_id
        is_private = getattr(event, "detail_type", "") in [DetailType.PRIVATE, "private"]
        is_mentioned = self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in content

        is_reply_to_others = False
        if f"[CQ:at,qq=" in content and not is_mentioned:
            # 如果回复的目标不是自己，判定为旁观者
            is_reply_to_others = True
            threshold = min(0.95, threshold + 0.25)
            state_desc += ", BystanderPenalty+0.25"

        context_str = "无"
        if recent_history:
            lines = []
            for msg in recent_history[-5:]:
                role = msg.get("role", "unknown")
                text = str(msg.get("content", "")).replace("\n", " ")[:150]
                lines.append(f"[{role}]: {text}")
            context_str = "\n".join(lines)

        prompt = f"""
你是 AI ({self.nickname}) 的“注意力与社交决策门控”。
你的任务是：基于当前状态与对话语境，做出一个符合人类直觉的“是否介入”判断。
请用简短的内部推理（think）来帮助你稳定决策。
———
【你的状态】
- 行动阻力: {threshold:.2f} （越高越不想参与）
- 当前兴趣: "{await self.get_current_interest_text()}"
- 状态影响: [{state_desc}]
———
【对话环境】
- 场景: {"私聊" if is_private else "群聊"}
- 是否被提及: {is_mentioned}
- 社交语境: {"他人对话" if is_reply_to_others else "与你相关或开放"}
- 发送者: {sender}
———
【消息内容】
"{content}"
【上下文】
{context_str}
———
【行为语义】
- REPLY: 自然回应，对方在等你
- INTERJECT: 主动切入（任务 / 强兴趣 / 必须处理）
- SILENT_OBSERVE: 与你相关，但不想回
- OBSERVE: 与你无关，仅围观
- IGNORE: 噪音 / 无意义
———
【决策直觉】
你只需回答4个隐含问题：
1. 这是否与我有关？
2. 我现在有没有动力参与？（受阻力影响）
3. 是否存在必须处理或强吸引点？
4. 该事件对我当前的继续交流意愿产生了怎样的微弱影响？
———
【重要倾向】
- 阻力高 → 更倾向 OBSERVE / SILENT_OBSERVE / IGNORE
- 被明确提及 → 提高 REPLY 概率
- 存在明确任务/指令 → 倾向 INTERJECT
- 他人对话 → 除非强相关，否则不要介入"""
        try:
            response = await self.api_client.create_chat_completion_once(
                messages=prompt,
                system_prompt="你是一个冷酷高效的注意力过滤引擎。",
                model=self.api_client.small_model,
                schema={
                    "type": "object",
                    "properties": {
                        "think": {
                            "type": "string",
                            "description": "简短内部推理：包含意图判断、是否与自己相关、是否值得介入，以及一句话总结行为动机"
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["REPLY", "INTERJECT", "SILENT_OBSERVE", "OBSERVE", "IGNORE"],
                            "description": "最终选择的社交介入姿态"
                        },
                        "should_do": {
                            "type": "string",
                            "description": "根据当前判断，建议下一步采取的具体行动或态度"
                        },
                        "willingness_shift": {
                            "type": "number",
                            "description": "对话意愿的微调值（极其克制，范围 -0.1 到 0.1）。无感为0.0，厌烦为负，吸引为正。"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "对该决策的确信度"
                        }
                    },
                    "required": ["think", "decision", "should_do", "willingness_shift", "confidence"],
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
                    return ReactionType.OBSERVE, "注意力系统降级容错", 0.0

            decision = result.get("decision", "OBSERVE")
            confidence = float(result.get("confidence", 0.0))
            should_do = result.get("should_do", "暂无特别建议")
            will_shift = float(result.get("willingness_shift", 0.0))
            think = result.get("think", "")

            logger.info(
                f"🧠 [Attention Eval] 置信度: {confidence:.2f} | 阻力: {threshold:.2f} | 决策: {decision} ({should_do}) | 意愿偏离: {will_shift:+.2f}\n{think}")

            # 强逻辑收束：防止模型出现置信度低于阈值却强行 REPLY 的幻觉
            reaction = ReactionType(decision.lower())
            if decision in ["REPLY", "INTERJECT"] and confidence < threshold:
                logger.warning(f"⚠️ [Attention] 置信度不足以击穿阻力。强制降级。")
                if is_private or is_mentioned:
                    reaction = ReactionType.SILENT_OBSERVE
                else:
                    reaction = ReactionType.OBSERVE

            return reaction, should_do, will_shift

        except Exception as e:
            logger.error(f"Attention LLM check failed: {e}")
            return ReactionType.OBSERVE, "注意力评估抛出异常", 0.0

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
        # 缓存有效性检查
        current_time = time.time()
        if self._cached_interest_vector and (current_time - self._last_interest_update < 300):  # 300秒TTL
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
