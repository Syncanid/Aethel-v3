# core/evolution/skill_fabricator.py
import asyncio
import json
import logging
import os

from core.infrastructure.api_client import GenericAPIClient
from core.io.event_bus import EventBus
from core.io.event_schema import Action
from tools.System2.skill_installer import create_custom_skill

logger = logging.getLogger(__name__)


class SkillFabricator:
    def __init__(self, event_bus: EventBus, api_client: GenericAPIClient):
        self.event_bus = event_bus
        self.api_client = api_client
        # 订阅 Action
        self.event_bus.subscribe_action(self._on_action)

    async def _on_action(self, action: Action):
        # 拦截我们自定义的内部方法
        if action.action == "extract_skill" and action.target_platform == "internal":
            task_id = action.params.get("task_id")
            if task_id:
                # 放入 EventBus 的后台任务集合（EventBus 自身支持 safe_execute）
                asyncio.create_task(self.extract_and_install_skill(task_id))

    async def extract_and_install_skill(self, task_id: str):
        logger.info(f"🧬 [演化系统] 启动总结器，开始为任务 {task_id} 提纯技能...")

        # 此时任务记录 100% 已经安全写入磁盘
        record_path = f"data/task_records/{task_id}.json"
        if not os.path.exists(record_path):
            logger.error("提纯失败：未找到任务黑匣子记录。")
            return

        with open(record_path, "r", encoding="utf-8") as f:
            record_data = json.load(f)

        history = record_data.get("execution_history", [])
        goal = record_data.get("goal", "")

        # 2. 构造给总结器 LLM 的 Prompt (严格约束 JSON 输出)
        system_prompt = """
你是一个 AI 核心架构师。你需要从以下一个 AI 代理执行任务的完整历史记录中，提取出成功的经验，并封装成一个标准 Skill。
注意：代理在历史记录中可能犯错、走弯路。你需要**抛弃这些弯路**，只提取出成功达成目标的**最短正确逻辑和代码**。

你需要输出一个严格格式化的 JSON 对象，包含以下字段：
- `skill_name`: 技能名称 (英文，小写，使用横线连接，如 'github-analyzer')。
- `description`: 一段给其他 AI 看的精准描述，说明什么时候该触发这个技能。
- `required_params`: 字符串数组。触发这个技能必须提供的入参名称。
- `sop_markdown`: 这个技能的标准操作流程 (SOP)。详细写明 AI 在使用这个技能时应该按什么顺序调用哪些工具。
- `tools_python_code`: (可选) 如果任务中涉及了自定义的 Python 逻辑处理，请将其编写为一个符合 Aethel 规范的工具。如果没有，留空字符串。

注意：不要自己生造基础能力（如读写文件），仅保留具有领域专业性的工具代码。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"目标: {goal}\n执行历史(含潜在的弯路):\n{json.dumps(history, ensure_ascii=False)}"}
        ]

        # 3. 强制 LLM 输出所需的 JSON 格式
        schema = {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string"},
                "description": {"type": "string"},
                "required_params": {"type": "array", "items": {"type": "string"}},
                "sop_markdown": {"type": "string"},
                "tools_python_code": {"type": "string"}
            },
            "required": ["skill_name", "description", "required_params", "sop_markdown"]
        }

        try:
            response = await self.api_client.create_chat_completion(
                messages=messages,
                schema=schema
            )
            result_json = json.loads(response["choices"][0]["message"]["content"])

            logger.info(f"🧬 [演化系统] 提纯完成，开始安装技能: {result_json['skill_name']}")

            # 4. 直接复用现成的 create_custom_skill 工具进行落盘和热重载
            install_result = await create_custom_skill(
                skill_name=result_json["skill_name"],
                description=result_json["description"],
                required_params=result_json["required_params"],
                sop_markdown=result_json["sop_markdown"],
                tools_python_code=result_json.get("tools_python_code", "")
            )

            logger.info(f"🧬 [演化系统] 技能安装结果: {install_result}")

        except Exception as e:
            logger.error(f"技能提纯/安装过程中发生异常: {e}", exc_info=True)
