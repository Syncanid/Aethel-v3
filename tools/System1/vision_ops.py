# tools/System1/vision_ops.py
from core.infrastructure.api_client import GenericAPIClient
from core.infrastructure.database import Database
from core.tool_manager.registry import register


@register()
async def reparse_image(
        image_id: str,
        target_goal: str,
        database: Database = None,
        api_client: GenericAPIClient = None
) -> str:
    """
    当需要对用户发送过的历史图片进行特定目标的深度提取或重新观察时调用。
    注意：系统上下文中的图片描述仅为浅层摘要。涉及解题、文字提取、翻译或审阅图片细节时，必须调用此工具。
    :param image_id: 图片的唯一ID (位于上下文中的 [图片附件 ID: xxx] 标签，如 img_a1b2c3d4)
    :param target_goal: 你希望底层视觉模型重点观察的目标指令 (例如 "详细翻译上面的英文", "提取表格数据输出为JSON", "图里代码的第10行写了什么")
    """
    if not database:
        return "【系统错误】数据库组件未就绪，无法检索视觉库。"

    # 从 SQLite 中物理穿透提取图像数据
    async with database.get_connection() as conn:
        cursor = await conn.execute("SELECT b64_data_uri FROM multimodal_cache WHERE image_id=?", (image_id,))
        row = await cursor.fetchone()

    if not row:
        return f"【提取失败】视觉库中未找到图片 {image_id}。该图片可能已被 LRU 机制淘汰或 ID 拼写错误。"

    b64_data_uri = row[0]

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"请你作为顶级视觉分析专家，仔细观察该图片并专门针对以下目标给出详尽解析：\n【分析目标】: {target_goal}\n请直接给出精确且硬核的结论，拒绝说正确的废话。"},
                {"type": "image_url", "image_url": {"url": b64_data_uri}}
            ]
        }]

        response = await api_client.create_chat_completion(
            messages=messages
        )

        return f"【图片 {image_id} 深度解析反馈】:\n{response.get('content')}"

    except Exception as e:
        return f"【系统告警】底层视觉穿透模型调用失败: {str(e)}"
