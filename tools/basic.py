import json
import logging
from typing import Literal, Optional, Dict, Any, List

from core.infrastructure.api_client import GenericAPIClient
from core.io.event_bus import EventBus
from core.io.event_schema import Action
from core.tool_manager.registry import register
from core.utilities import process_llm_response

logger = logging.getLogger(__name__)


@register()
async def think(
        content: str,
        event_bus: EventBus
) -> str:
    """
    [Think] 进行深度思考、推理或规划。
    在执行关键操作之前，或者遇到复杂问题时，请先使用此工具整理思路。
    思考内容会被记录在系统日志中，作为你的思维链 (Chain of Thought)，有助于保持逻辑清晰。

    Args:
        content: 思考的具体内容、分析过程或下一步计划。
    """
    # 将思考过程广播到控制台，使用不同于普通日志的图标
    event_bus.publish_action(Action(
        action="broadcast_log",
        params={"content": f"🧠 [思维链]: {content}"}
    ))

    # 返回值会被存入历史，强化记忆
    return "思考过程已记录。"


@register()
async def advanced_think(
        api_client: GenericAPIClient,
        goal: str,
        mode: Literal["plan", "reflect", "decompose", "validate", "brainstorm"] = "plan",
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    [Cognition] 执行结构化思考过程，为 AI Agent 提供可解释的推理步骤。
    当你面对复杂问题、需要制定计划、或者反思错误时，必须使用此工具。

    Args:
        goal: 当前需要解决的核心目标或问题。
        mode: 思考模式 ("plan"=规划, "reflect"=反思, "decompose"=拆解, "validate"=验证, "brainstorm"=头脑风暴)。
        context: 相关的上下文信息或背景数据。
        constraints: 思考过程需遵守的约束条件列表。

    Returns:
        包含步骤 (steps)、原理 (rationale) 和置信度 (confidence) 的结构化字典。
    """

    # --- 1. 内部辅助函数：生成 Prompt ---
    def _generate_system_prompt(mode: str, constraints: Optional[List[str]]) -> str:
        mode_descriptions = {
            "plan": "你是一个任务规划专家。将目标分解为清晰、有序、可执行的步骤。每个步骤应具体且无歧义。",
            "reflect": "你是一个问题分析专家。分析先前行动失败的原因，识别根本问题，并提出改进建议。",
            "decompose": "你是一个问题分解专家。将复杂问题拆解为独立、可管理的子问题，确保覆盖所有关键方面。",
            "validate": "你是一个验证专家。严格检查数据/假设的有效性，识别潜在错误或不一致，并提供验证方法。",
            "brainstorm": "你是一个创意生成专家。提出多种创新解决方案，考虑不同角度，并评估每个方案的可行性。"
        }

        base_prompt = (
            f"{mode_descriptions.get(mode, mode_descriptions['plan'])}\n\n"
            "输出必须严格遵循JSON Schema格式。\n"
            "重要规则:\n"
            "1. 语言: 使用与用户输入相同的语言\n"
            "2. 步骤数量: 通常3-7步，根据复杂度调整\n"
            "3. 具体性: 步骤必须具体、可操作，避免模糊表述"
        )

        if constraints:
            base_prompt += "\n\n必须遵守的约束条件:\n"
            for i, constraint in enumerate(constraints, 1):
                base_prompt += f"{i}. {constraint}\n"

        return base_prompt.strip()

    def _build_user_message(goal: str, context: Any, mode: str) -> str:
        message = f"当前目标: {goal}\n"

        if context:
            context_str = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
            message += f"\n相关上下文:\n{context_str}\n"

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
                "description": "建议的后续行动，如 tool 名称"
            }
        },
        "required": ["steps", "rationale", "confidence"],
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
            schema=output_schema # 直接传入 Schema 启用 Structured Outputs
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