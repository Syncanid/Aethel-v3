def process_llm_response(message: dict):
    tools = []
    if message.get("tool_calls", None):
        for tool in message["tool_calls"]:
            tools.append({
                "id": tool["id"],
                "name": tool["function"]["name"],
                "args": tool["function"]["arguments"]
            })
    msg = message["content"]
    return msg, tools
