import asyncio
import json
import logging

import websockets

from core.infrastructure.api_client import GenericAPIClient
# 导入底层基建
from core.infrastructure.config_loader import Config
from core.infrastructure.database import Database
from core.infrastructure.logger import setup_logger
from core.io.event_schema import OneBotEvent, EventType, DetailType, EventSource
# 导入需要被隔离测试的目标模块
from core.kernel.attention import AttentionFilter
from core.kernel.prompt import PromptManager

logger = logging.getLogger("AttentionTest")

# ==========================================
# 🛑 核心测试配置：修改为你要观测的指定群聊 ID
# ==========================================
TARGET_GROUP_ID = "882134528"


async def main():
    print("初始化系统基建 (沙盒模式)...")

    # 1. 挂载只读/默认基建，绝对物理隔离 Agent 与 EventBus
    config = Config()
    config.load("data/config-for-testing.yaml")

    # 初始化日志系统
    setup_logger(config)
    logger.info("日志系统加载完毕，正在拉起沙盒环境...")

    # 2. 连接数据库
    db = Database(config)
    await db.init()

    # 3. 初始化 API 客户端与 Prompt 管理器
    api_client = GenericAPIClient(config)
    prompt_manager = PromptManager(config)

    # 4. 实例化注意力过滤器 (只读不写，无实体 Agent)
    attention = AttentionFilter(
        config=config,
        prompt=prompt_manager,
        api_client=api_client,
        database=db
    )

    # 预热并缓存当前的兴趣焦点
    interest = await attention.get_current_interest_text()

    logger.info(f"Attention 系统加载完毕。当前昵称: {attention.nickname}")
    logger.info(f"当前认知焦点 (Interest): {interest}")
    logger.info("⚠️ 沙盒机制已确认：未加载 Agent 中枢与事件总线，绝对禁止物理发包。")

    # 5. 读取正向 WS 配置
    ws_url = config.get("system.onebot_url", "ws://127.0.0.1:3001")
    token = config.get("system.onebot_token", None)

    logger.info(f"准备建立正向 WebSocket 连接，目标地址: {ws_url}")

    # 6. 建立正向 OneBot 接收端 (复刻 onebot_v11.py 的连接循环)
    while True:
        try:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            async with websockets.connect(ws_url, additional_headers=headers) as ws:
                logger.info("✅ 正向 OneBot WebSocket 已成功连接！开始监听数据流...")

                async for message in ws:
                    try:
                        data = json.loads(message)

                        # A. 过滤非消息事件
                        if data.get("post_type") != "message":
                            continue

                        # B. 过滤非群聊消息
                        if data.get("message_type") != "group":
                            continue

                        group_id = str(data.get("group_id"))

                        # C. 【核心过滤】仅接受指定群聊
                        if group_id != TARGET_GROUP_ID:
                            continue

                        user_id = str(data.get("user_id"))
                        # 防火墙：忽略自己发出的消息
                        if user_id == str(config.get("system.bot_self_id")):
                            continue

                        text = data.get("raw_message", "")

                        # 组装 OneBotEvent 标准协议
                        event = OneBotEvent(
                            type=EventType.MESSAGE,
                            detail_type=DetailType.GROUP,
                            source=EventSource(platform="onebot", group_id=group_id, user_id=user_id),
                            message=text,
                            alt_message=text,
                            raw_data=data
                        )

                        logger.info(f"📥 [外部刺激] {user_id} @ 群 {group_id}: {text}")

                        # 触发纯粹的注意力求值
                        try:
                            reaction = await attention.evaluate(
                                event=event,
                                recent_history=[],
                                willingness=0.5
                            )
                            logger.info(f"🧠 [门控决策] -> 判定结果: {reaction.value.upper()}\n")
                        except Exception as e:
                            logger.error(f"Attention 评估流崩溃: {e}", exc_info=True)

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.error(f"处理 OneBot 消息帧失败: {e}", exc_info=True)

        except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
            logger.warning(f"OneBot 连接断开或遭拒 ({e})，5秒后尝试重新连接...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"沙盒网络层发生未捕获异常: {e}", exc_info=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    import sys

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n沙盒测试已终止。")
