# System1Tools/s1_interact.py
import json
import os
import time

from core.io.event_schema import OneBotEvent, EventType, DetailType, TaskPayload, EventSource
from core.tool_manager.registry import register


@register()
async def dispatch_background_task(
        task_description: str,
        parameters: dict,
        event_bus=None,
) -> str:
    """
    当用户要求执行复杂任务（如：搜索网络、分析文件、写代码、查系统状态、多步推理等）时，调用此工具将任务派发给后台 System 2 处理。
    :param task_description: 任务的详细描述，例如 '搜索今天的新闻并总结' 或 '扫描本地端口'
    :param parameters: 任务所需的参数字典，如 {"url": "xxx"}，如果没有可传 {}
    """
    task_id = f"task_{int(time.time())}"

    # 构造 TaskPayload (复用我们阶段 1 写的结构)
    payload = TaskPayload(
        task_id=task_id,
        description=task_description,
        parameters=parameters
    )

    source = EventSource(
        platform="internal",
    )

    # 发送内部事件到 EventBus
    event = OneBotEvent(
        type=EventType.TASK,
        detail_type=DetailType.TASK_DISPATCH,
        source=source,
        extra={"task_payload": payload.model_dump()}
    )

    event_bus.publish_event(event)

    # 告诉 System 1 任务已成功扔出去了
    return f"任务已成功派发至后台 (任务ID: {task_id})。"


@register()
async def send_info_to_running_task(
        task_id: str,
        info: str,
        event_bus=None,
) -> str:
    """
    向正在运行的后台任务发送补充信息。
    :param task_id: 任务的唯一ID (如 task_1710000000)
    :param info: 要传递给后台的具体信息文本
    """
    payload = TaskPayload(
        task_id=task_id,
        description=info
    )

    source = EventSource(
        platform="internal"
    )

    event = OneBotEvent(
        type=EventType.TASK,
        detail_type=DetailType.TASK_UPDATE,
        source=source,
        extra={"task_payload": payload.model_dump()}
    )

    event_bus.publish_event(event)
    return f"已成功将补充信息发送给任务 {task_id}。"


@register()
async def query_task_history(
        task_id: str,
        user_question: str,
        api_client=None
) -> str:
    """
    查询已完成的历史任务。因为记录可能很长，会自动通过 LLM 提取关键信息。
    :param task_id: 任务的唯一ID
    :param user_question: 用户想了解的具体细节，例如 "它用了哪些工具？" 或 "最后一步的报错详情是什么？"
    """
    filepath = f"data/task_records/{task_id}.json"
    if not os.path.exists(filepath):
        return f"查询失败：未找到任务 {task_id} 的存档文件。"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            record_data = json.load(f)

        # 如果仅仅是问最后的结果，直接返回
        if "结果" in user_question and "过程" not in user_question and "详细" not in user_question:
            return f"任务结果：\n{record_data.get('final_result')}"

        # 如果问细节，为了防止 S1 被长达几万 token 的 history 撑爆，
        # 我们在这里借用 api_client (小模型) 做一次 Map-Reduce 压缩阅读。
        history_text = json.dumps(record_data.get("execution_history", []), ensure_ascii=False)

        # 为了防止超长，截断至大约 20000 字符
        if len(history_text) > 20000:
            history_text = history_text[-20000:]

        prompt = (
            f"你是一个分析员。以下是后台执行引擎的底层日志片段。\n"
            f"请根据用户的疑问，从日志中提取答案，用简明扼要的自然语言回复，不要输出多余的废话。\n\n"
            f"【用户的疑问】: {user_question}\n"
            f"【底层日志】:\n{history_text}"
        )

        # 使用 api_client 的 create_chat_completion_once (无记忆单次调用)
        # 依赖于传入的 api_client (由依赖注入提供)
        response = await api_client.create_chat_completion_once(
            messages=prompt,
            system_prompt="你是一个精准的日志提取工具。",
            model=api_client.small_model  # 使用较快的小模型
        )

        return f"历史记录查询结果：\n{response.get('content')}"

    except Exception as e:
        return f"读取或分析记录时发生错误：{str(e)}"


@register()
async def cancel_background_task(
        task_id: str,
        event_bus=None,
) -> str:
    """
    当用户明确表示要取消、终止或放弃正在后台执行的任务时，调用此工具。
    :param task_id: 任务的唯一ID
    """
    payload = TaskPayload(task_id=task_id)
    source = EventSource(
        platform="internal",
    )

    event = OneBotEvent(
        type=EventType.TASK,
        detail_type=DetailType.TASK_CANCEL,
        source=source,
        extra={"task_payload": payload.model_dump()}
    )
    event_bus.publish_event(event)
    return f"已向后台发送取消指令，任务 {task_id} 即将终止。"
