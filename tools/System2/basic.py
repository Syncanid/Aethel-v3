import datetime
import json
import logging
from typing import Literal, Optional, Dict, Any, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dateutil import parser

from core.infrastructure.api_client import GenericAPIClient
from core.io.event_bus import EventBus
from core.io.event_schema import Action, OneBotEvent, EventType, EventSource
from core.kernel.agent import AutonomousAgent
from core.tool_manager.registry import register

logger = logging.getLogger(__name__)


# --- 内部辅助函数：发送唤醒事件 ---
async def _dispatch_wake_up(event_bus: EventBus, reason: str):
    """调度器回调：发送唤醒事件"""
    logger.info(f"⏰ 等待结束，触发唤醒: {reason}")
    event = OneBotEvent(
        type=EventType.NOTICE,  # 使用 Notice 类型
        detail_type="wake_up",
        sub_type="timer",
        source=EventSource(platform="system"),
        message=f"【系统唤醒】: {reason}",
        alt_message=f"【系统唤醒】: {reason}",
        extra={"status": "wake_up"}
    )
    event_bus.publish_event(event)


@register()
async def wait(
        scheduler: AsyncIOScheduler,
        event_bus: EventBus,
        agent: AutonomousAgent,
        duration: Optional[float] = None,
        until: Optional[str] = None,
        reason: Optional[str] = None,
) -> Optional[str]:
    """
    让系统进入等待/挂起状态。
    系统会挂起当前任务，直到指定时间到达后通过事件被唤醒。

    Args:
        duration: 等待的分钟 (相对时间)。
        until: 等待直到具体的日期时间 (绝对时间)，ISO 格式。
        reason: 启动等待的原因。
    """

    # 1. 计算触发时间
    if until:
        try:
            # 使用 dateutil 解析自然语言或 ISO 时间
            run_date = parser.parse(until)
            # 如果解析出的时间没有时区，且当前是 Awareness 的，需处理 (这里简化，假设本地时间)
            if run_date < datetime.datetime.now():
                return f"错误: 目标时间 {until} 已经是过去式了。"
        except Exception:
            return f"错误: 无法解析时间字符串 '{until}'。请使用 ISO 格式 (YYYY-MM-DD HH:MM:SS)。"

    elif duration is not None:
        if duration <= 0:
            return "错误: 等待时长必须大于 0。"
        run_date = datetime.datetime.now() + datetime.timedelta(minutes=duration)

    else:
        return "错误: 必须提供 'duration' (分) 或 'until' (日期字符串) 其中之一。"

    # 2. 添加调度任务
    if scheduler and run_date:
        # 获取 job 对象
        job = scheduler.add_job(
            _dispatch_wake_up,
            'date',
            run_date=run_date,
            args=[event_bus, "等待超时"]
        )

        # 将 Job ID 绑定到 Agent 实例，用于后续取消
        if agent:
            agent.wakeup_job_id = job.id
            agent.is_sleeping = True

        return None
    else:
        return "系统错误: 调度器未初始化。"


@register()
async def wait_forever(
        agent: Any = None,
        event_bus: EventBus = None,
        reason: Optional[str] = None,
) -> Optional[str]:
    """
    [Control] 进入无限期休眠状态，直到收到外部事件唤醒。
    优先使用wait工具，尽量不使用wait_forever工具。

    Args:
        reason: 启动休眠的原因。
    """
    if agent:
        # 如果存在之前的定时唤醒任务（例如之前的 wait 设置的），则取消它
        # 避免定时器在休眠期间意外触发唤醒
        if hasattr(agent, "wakeup_job_id") and agent.wakeup_job_id:
            try:
                # 假设 agent.scheduler 是 AsyncIOScheduler 实例
                agent.scheduler.remove_job(agent.wakeup_job_id)
                logger.info(f"[WaitForever] 已移除原有的定时唤醒任务: {agent.wakeup_job_id}")
            except Exception as e:
                # 忽略任务不存在的错误
                logger.debug(f"[WaitForever] 移除定时任务失败 (可能已不存在): {e}")
            agent.wakeup_job_id = None

        # 强制设置休眠标志
        agent.is_sleeping = True

    # 广播系统状态变更
    if event_bus:
        event_bus.publish_action(Action(
            action="broadcast_log",
            params={"content": f"💤 系统进入无限期待命状态: {reason}"}
        ))

    return None


@register()
async def think(event_bus: EventBus, thought: str) -> None:
    """
    [Cognition] 执行思考
    当你面对复杂问题、需要制定计划、或者反思错误时，必须使用此工具。

    Args:
        thought: 你的思考过程
    """
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"💭 {thought}"}
    ))
    return None


@register()
async def advanced_think(
        api_client: GenericAPIClient,
        tool_manager: Any,
        goal: str,
        mode: Literal["plan", "reflect", "decompose", "validate", "brainstorm"],
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    [Cognition] 执行结构化思考过程，为 AI Agent 提供可解释的推理步骤。
    当think无法解决问题时使用此工具。

    Args:
        goal: 当前需要解决的核心目标或问题。
        mode: 思考模式 ("plan"=规划, "reflect"=反思, "decompose"=拆解, "validate"=验证, "brainstorm"=头脑风暴)。
        context: 相关的上下文信息或背景数据。
        constraints: 思考过程需遵守的约束条件列表。
    """

    # --- 0. 获取当前环境真实可用的工具列表 ---
    tools_context_str = "暂无工具信息"
    if tool_manager:
        try:
            # 获取所有工具的 Schema
            schemas = tool_manager.get_tool_schemas()
            tool_lines = []
            for s in schemas:
                func = s.get("function", {})
                name = func.get("name", "unknown")
                if name == "advanced_think": continue
                desc = func.get("description", "无描述").strip()
                tool_lines.append(f"- {name}: {desc}")
            tools_context_str = "\n".join(tool_lines)
        except Exception as e:
            logger.warning(f"获取工具列表失败: {e}")

    # --- 1. 内部辅助函数：生成 Prompt ---
    def _generate_system_prompt(mode: str, constraints: Optional[List[str]]) -> str:
        mode_descriptions = {
            "plan": "你是一个注重实效的敏捷执行者。你的目标是快速找到解决当前问题的切入点，而不是制定完美的长期蓝图。",
            "reflect": "你是一个问题分析专家。分析先前行动失败的原因，识别根本问题，并提出改进建议。",
            "decompose": "你是一个问题分解专家。将复杂问题拆解为独立、可管理的子问题，确保覆盖所有关键方面。",
            "validate": "你是一个验证专家。严格检查数据/假设的有效性，识别潜在错误或不一致，并提供验证方法。",
            "brainstorm": "你是一个创意生成专家。提出多种创新解决方案，考虑不同角度，并评估每个方案的可行性。"
        }

        base_prompt = (
            f"{mode_descriptions.get(mode, mode_descriptions['plan'])}\n\n"
            f"【环境约束】\n"
            f"1. 你运行在一个受限的 Python 异步环境中。你必须通过调用工具来与外界交互。\n"
            f"2. 以下是当前可用的工具列表，严禁使用列表之外的工具：\n"
            f"{tools_context_str}\n\n"
            "重要规则:\n"
            "1. 语言: 使用与用户输入相同的语言\n"
            "2. 步骤数量: 通常2-5步，根据复杂度调整\n"
            "3. 具体性: 步骤必须具体、可操作，并明确指出应调用的工具名称\n"
            "4. 拒绝过度规划: 不要想得太远。仅规划接下来最关键、最直接的步骤。\n"
            "5. 拥抱变化: 承认未来的不确定性，规划应聚焦于“获取信息”或“尝试执行”。"
        )

        if constraints:
            base_prompt += "\n\n额外约束:\n"
            for i, constraint in enumerate(constraints, 1):
                base_prompt += f"{i}. {constraint}\n"

        return base_prompt.strip()

    def _build_user_message(goal: str, context: Any, mode: str) -> str:
        message = f"当前目标: {goal}\n"

        if context:
            context_str = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
            message += f"\n上下文信息:\n{context_str}\n"

        mode_prompts = {
            "plan": "请规划完成此目标的具体步骤",
            "reflect": "请分析问题原因并提出改进建议",
            "decompose": "请将此问题拆解为关键子问题",
            "validate": "请验证相关数据/假设的有效性",
            "brainstorm": "请生成多种可行解决方案"
        }

        message += f"\n{mode_prompts.get(mode, '请进行结构化思考')}"
        return message.strip()

    # --- 2. 输入校验 ---
    if api_client is None:
        raise ValueError("Dependency 'api_client' is missing or None")

    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal 必须是非空字符串")

    valid_modes = ["plan", "reflect", "decompose", "validate", "brainstorm"]
    if mode not in valid_modes:
        raise ValueError(f"无效的思考模式: {mode}, 必须是 {valid_modes} 之一")

    if constraints and not all(isinstance(c, str) for c in constraints):
        raise ValueError("constraints 必须是字符串列表")

    if context and not isinstance(context, (str, dict)):
        raise ValueError("context 必须是字符串或字典类型")

    # --- 3. 准备调用 ---
    system_prompt = _generate_system_prompt(mode, constraints)
    user_message = _build_user_message(goal, context, mode)

    # 定义 Schema
    output_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "编号的思考步骤列表"
            },
            "rationale": {
                "type": "string",
                "description": "选择此思考路径的核心逻辑"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "置信度分数 (0.0~1.0)"
            },
            "suggested_next_action": {
                "type": ["string", "null"],
                "description": "建议的后续行动"
            }
        },
        "required": ["steps", "rationale", "confidence", "suggested_next_action"],
        "additionalProperties": False
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # --- 4. 调用 LLM ---
    try:
        # 使用 v3 的 GenericAPIClient
        logger.info(f"🤔 高级思考激活: [{mode}] {goal}")
        response_data = await api_client.create_chat_completion(
            messages=messages,
            schema=output_schema  # 直接传入 Schema 启用 Structured Outputs
        )

        # 提取内容 (GenericAPIClient 返回完整的 OpenAI 格式 dict)
        raw_content = response_data["choices"][0]["message"]["content"]

        # 解析 JSON
        result = json.loads(raw_content)

        # 注入元数据
        result["mode"] = mode

        # 裁剪置信度
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))

        return result

    except Exception as e:
        logger.error(f"高级思考失败: {e}", exc_info=True)
        return {
            "error": f"Cognitive process failed: {str(e)}",
            "mode": mode,
            "steps": [],
            "confidence": 0.0
        }


@register()
async def groundbreaking_analysis(
        problem_statement: str,
        context: str,
        api_client: GenericAPIClient,
        event_bus: EventBus
) -> Dict[str, Any]:
    """
    [Cognition] 开创性分析工具。
    当你发现自己陷入重复思考、无法推进任务、或者不知道下一步该做什么时，必须调用此工具。
    它会帮助你跳出当前的思维定势，进行深度的根本原因分析，并提供创新性的解决方案。

    Args:
        problem_statement: 当前遇到的核心障碍或死锁原因。
        context: 相关的背景信息或之前的尝试。
    """

    # 广播开始分析的信号
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"💡 正在进行开创性分析: {problem_statement}"}
    ))

    # 1. 定义输出 Schema
    analysis_schema = {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "死锁诊断：分析为什么会卡在当前的循环中（例如：过度谨慎、缺乏工具、指令模糊）。"
            },
            "lateral_thinking": {
                "type": "string",
                "description": "侧向思维：提供一个完全不同于之前尝试的视角或解决思路。"
            },
            "actionable_plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": "行动指令：3个具体的、可立即执行的下一步行动建议（必须包含具体的工具调用建议）。"
            }
        },
        "required": ["diagnosis", "lateral_thinking", "actionable_plan"],
        "additionalProperties": False
    }

    # 2. 简化 Prompt，专注于任务逻辑而非格式
    system_prompt = """
你是一个高级 AI 策略顾问。你的客户（一个 AI Agent）陷入了思维死锁或循环。
请帮助它破局。你的任务不是直接给出代码，而是提供策略性的指导。
运用“第一性原理”和“侧向思维”来重新审视问题。
"""

    user_prompt = f"问题描述: {problem_statement}\n上下文: {context}"

    try:
        # 3. 调用 API 并传入 schema
        result = await api_client.create_chat_completion_once(
            messages=user_prompt,
            system_prompt=system_prompt,
            schema=analysis_schema
        )

        content = result["content"]

        # 4. 解析结果
        try:
            analysis_data = json.loads(content)

            # 记录分析结果
            logger.info(f"分析完成: {analysis_data.get('diagnosis')}")
            return analysis_data

        except json.JSONDecodeError:
            logger.error(f"尽管使用了 Schema，返回内容仍非有效 JSON: {content}")
            return {"error": "解析分析结果失败", "raw": content}

    except Exception as e:
        logger.error(f"开创性分析失败: {e}", exc_info=True)
        return {"error": str(e)}
