# core/kernel/attention.py
import json
import logging
import math
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
    REPLY = "reply"  # 必须回复 (直接交互/强相关)
    INTERJECT = "interject"  # 主动插话 (弱交互/感兴趣)
    OBSERVE = "observe"  # 静默观察 (无价值/不相关/插话阈值过高)
    IGNORE = "ignore"  # 完全忽略 (如黑名单/无关系统通知)


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
        self.last_reply_time = 0.0
        self._cached_interest_vector: Optional[List[float]] = None
        self._cached_interest_text: str = ""
        self._last_interest_update = 0.0

        # --- 默认配置 ---
        # 基础阈值：越低越容易触发
        self.BASE_THRESHOLD = 0.60
        # 对话惯性：刚说过话后，阈值降低多少 (更容易接话)
        self.CONVERSATION_INERTIA = 0.25
        # 惯性持续时间 (秒)
        self.INERTIA_WINDOW = 120

    async def evaluate(self, event: OneBotEvent, recent_history: Optional[List[Dict]] = None) -> ReactionType:
        """
        [主入口] 评估事件重要性
        流程: 硬规则 -> 语义门控(Pre-Filter) -> LLM软规则(Soft-Rule)
        """
        # 1. 基础过滤
        if event.type != EventType.MESSAGE:
            if event.detail_type == DetailType.INTERNAL_DRIVE:
                return ReactionType.REPLY
            return ReactionType.IGNORE

        # 2. 硬规则 - 绝对优先
        # 包含: @我, 私聊, 呼叫名字
        if hard_reaction := self._check_hard_rules(event):
            self.last_reply_time = time.time()  # 更新活跃时间
            return hard_reaction

        # --- 获取动态生理阈值 ---
        # 这个阈值将贯穿 语义门控 和 LLM决策 两个阶段
        dynamic_threshold, state_desc = await self._calculate_dynamic_threshold()

        # 3. 语义门控
        # 这是一个低成本的向量检查，决定是否值得消耗 Token 进 LLM
        gate_passed, gate_reason = await self._semantic_gate_check(event, dynamic_threshold, state_desc)

        if not gate_passed:
            # 门控未通过 -> 直接忽略
            return ReactionType.IGNORE

        # 4. 软规则
        # 只有过了门控的精英消息，才有资格让大脑(LLM)思考
        soft_reaction = await self._check_soft_rules(event, dynamic_threshold, state_desc, recent_history)

        # 如果 LLM 决定回复，更新活跃时间
        if soft_reaction in [ReactionType.REPLY, ReactionType.INTERJECT]:
            self.last_reply_time = time.time()

        return soft_reaction

    async def _calculate_dynamic_threshold(self) -> Tuple[float, str]:
        """
        根据边缘系统状态计算动态阈值
        公式: Thr = Base + (Dopamine * 0.15) - (Boredom * 0.3) - (Inertia)
        """
        state = await self._get_neuro_state()
        threshold = self.BASE_THRESHOLD
        factors = []

        # A. 多巴胺 (快乐/忙碌) -> 提高阈值 (变高冷)
        # 范围 0.0~1.0 -> 修正 +0.0 ~ +0.15
        dopa_mod = state.dopamine * 0.15
        threshold += dopa_mod
        factors.append(f"Dopa+{dopa_mod:.2f}")

        # B. 社交饱腹感 (孤独/无聊) -> 降低阈值 (变渴望)
        # Satiety 1.0 (饱) -> Mod 0.0
        # Satiety 0.0 (饿) -> Mod -0.25
        boredom_mod = (1.0 - state.social_satiety) * 0.25
        threshold -= boredom_mod
        factors.append(f"Lonely-{boredom_mod:.2f}")

        # C. 对话惯性 (Recently Active) -> 大幅降低阈值
        # 刚刚还在说话，应该很容易接话
        time_diff = time.time() - self.last_reply_time
        if time_diff < self.INERTIA_WINDOW:
            inertia_mod = self.CONVERSATION_INERTIA * (1 - (time_diff / self.INERTIA_WINDOW))
            threshold -= inertia_mod
            factors.append(f"Inertia-{inertia_mod:.2f}")

        # 钳位
        threshold = max(0.2, min(0.95, threshold))

        return threshold, ",".join(factors)

    # --- 核心组件 1：语义门控 ---
    async def _semantic_gate_check(self, event: OneBotEvent, threshold: float, state_desc: str) -> Tuple[bool, str]:
        """
        计算 (消息向量 vs 兴趣向量) 的相似度，并与 (生理驱动动态阈值) 比较。
        """
        message_text = event.alt_message
        if not message_text or len(message_text) < 2:
            return False, "消息太短"

        # 1. 获取当前兴趣向量
        interest_vec = await self._get_current_interest_vector()
        if not interest_vec:
            # 如果没有兴趣向量（初始化失败），默认放行，依靠 LLM
            return True, "无兴趣向量，默认放行"

        # 2. 计算消息向量 (调用 Embedding API)
        msg_vec = await self.api_client.create_embedding(message_text)
        if not msg_vec:
            return False, "Embedding 生成失败"

        # 3. 计算相似度
        similarity = self._cosine_similarity(interest_vec, msg_vec)

        # 4. 判定
        passed = similarity >= threshold

        log_msg = (f"Sim={similarity:.2f} | Thr={threshold:.2f} "
                   f"({state_desc}) | Interest='{self._cached_interest_text}'")

        return passed, log_msg

    # --- 核心组件 2：LLM 软规则 ---
    async def _check_soft_rules(self, event: OneBotEvent, threshold: float, state_desc: str, recent_history: Optional[List[Dict]] = None) -> ReactionType:
        """
        LLM 决策层：只有通过了语义门控的消息才会到达这里。
        """
        content = event.alt_message
        sender = event.source.user_id

        context_str = "无"
        if recent_history:
            lines = []
            for msg in recent_history:
                role = msg.get("role", "unknown")
                text = str(msg.get("content", ""))
                # 简单清洗与截断，防止 Token 爆炸
                text = text.replace("\n", " ")
                if len(text) > 60:
                    text = text[:60] + "..."
                lines.append(f"- {role}: {text}")
            context_str = "\n".join(lines)

        # 将数学阈值转换为自然语言指导
        if threshold > 0.7:
            mode_desc = "高冷模式: 你很忙或心情好，只回复极具价值、有趣或紧急的消息。忽略无聊的闲聊。"
        elif threshold < 0.4:
            mode_desc = "渴望模式: 你感到无聊或孤独，即使是普通的闲聊也应该积极回复以建立连接。"
        else:
            mode_desc = "标准模式: 正常评估消息的价值。"

        # 构造 Prompt
        prompt = f"""
你是一个群聊中的 AI 助手 ({self.nickname})。请作为“注意力过滤器”，评估以下用户消息，决定是否需要介入。

当前状态: {mode_desc} (生理阈值: {threshold:.2f}, 状态: {state_desc})。
当前兴趣: "{self._cached_interest_text}"。

【近期上下文】
{context_str}

【当前消息】
发送者: {sender}
内容: "{content}"

【决策标准】
1. REPLY (回复): 
   - 用户在向你提问 (即使没 @ 你)。
   - 话题与你高度相关。
   - 检测到用户情绪激动，需要安抚。
   - 上下文显示这是对你上一句回复的追问。

2. INTERJECT (插话): 
   - 用户在聊其他话题，但你觉得非常有趣、有梗。
   - 你的专业知识能提供巨大帮助。
   - 注意：不要做一个烦人的插话者，只有高质量的插话才被允许。

3. OBSERVE (观察): 
   - 闲聊、无关话题。
   - 争吵、辱骂等负面内容。
   - 你插不上话，或者不需要你参与的内容。

请输出 JSON:
{{
    "decision": "REPLY" | "INTERJECT" | "OBSERVE",
    "reason": "简短理由",
    "confidence": 0.0~1.0
}}
"""
        try:
            # 使用 create_chat_completion 的 schema 模式强制结构化输出
            response = await self.api_client.create_chat_completion(
                messages=[{"role": "system", "content": prompt}],
                model=self.api_client.small_model,
                schema={
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["REPLY", "INTERJECT", "OBSERVE"]},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["decision", "reason", "confidence"],
                    "additionalProperties": False
                }
            )

            content_str = response["choices"][0]["message"]["content"]
            result = json.loads(content_str)

            decision = result.get("decision", "OBSERVE")
            confidence = result.get("confidence", 0.0)

            # 决策逻辑
            if decision == "REPLY":
                return ReactionType.REPLY

            if decision == "INTERJECT":
                # 插话需要高置信度
                if confidence >= threshold:
                    return ReactionType.INTERJECT
                else:
                    logger.info(f"🛑 [Attention] 抑制插话意图 (置信度 {confidence:.2f} < {threshold})")
                    return ReactionType.OBSERVE

            return ReactionType.OBSERVE

        except Exception as e:
            logger.error(f"Attention LLM check failed: {e}")
            # 发生错误时保持安静，避免刷屏
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

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """手撸余弦相似度，避免 numpy 依赖"""
        if not v1 or not v2: return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _check_hard_rules(self, event: OneBotEvent) -> Optional[ReactionType]:
        # 1. @我
        if self.bot_self_id and f"[CQ:at,qq={self.bot_self_id}]" in event.alt_message:
            return ReactionType.REPLY

        # 2. 呼叫名字
        if self.nickname and event.alt_message.strip().startswith(self.nickname):
            return ReactionType.REPLY

        # # 3. 私聊 (可选，视配置而定)
        # if event.detail_type == DetailType.PRIVATE:
        #     return ReactionType.REPLY

        return None
