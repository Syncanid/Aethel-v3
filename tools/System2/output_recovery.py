# tools/System2/output_recovery.py
from core.tool_manager.output_cache import ToolOutputCache
from core.tool_manager.registry import register


@register()
async def read_full_output(receipt_id: str) -> str:
    """
    [读取] 获取重型工具（如深度搜索、文件读取）的【完整原始输出】。
    当提炼引擎给出的摘要不够用，你必须查看原始的完整细节时调用。
    注意：这可能会消耗大量上下文，请谨慎使用。

    Args:
        receipt_id: 系统提示给你的凭证号 (如 'out_a1b2c3d4')
    """
    content = ToolOutputCache.get(receipt_id)
    if content:
        # 给模型一个硬性警告，防止其把几万字全吃进去
        if len(content) > 5000:
            return f"【警告】原始输出长达 {len(content)} 字符，直接读取会导致你遗忘上下文。\n强烈建议改用 `refine_tool_output` 工具更换搜索目的。"
        return content
    return f"【错误】找不到凭证号 {receipt_id} 的缓存数据，可能已过期。"


@register()
async def refine_tool_output(receipt_id: str, new_purpose: str) -> str:
    """
    [提炼] 改变视角，重新提炼历史工具输出。
    当你发现上一次调用工具时给的“目的”找错了方向，导致提炼出的摘要不对时，
    无需重新调用原工具，直接使用此工具并赋予【新的目的】，系统会对着暂存的原始数据重新为你总结。

    Args:
        receipt_id: 系统提示给你的凭证号 (如 'out_a1b2c3d4')
        new_purpose: 你现在想从这份数据里找什么？(例如："忽略代码实现，只帮我总结出它的报错码大全")
    """
    content = ToolOutputCache.get(receipt_id)
    if content:
        refined = await ToolOutputCache.refine_content(content, new_purpose)
        return f"重新提炼结果 ({new_purpose}):\n\n{refined}"
    return f"【错误】找不到凭证号 {receipt_id} 的缓存数据。"