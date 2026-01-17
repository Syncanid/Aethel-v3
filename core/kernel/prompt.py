import datetime
import os
import platform
import sys
import time

from core.infrastructure.config_loader import Config


class PromptManager:
    def __init__(self, config: Config):
        self.config = config
        self.prompt_path = "data/prompts/system_prompt.md"
        self._ensure_prompt_file()
        self.start_time = time.time()

    def _ensure_prompt_file(self):
        if not os.path.exists(self.prompt_path):
            os.makedirs(os.path.dirname(self.prompt_path), exist_ok=True)
            with open(self.prompt_path, "w", encoding="utf-8") as f:
                f.write("你是一个名为 Aethel 的自主 AI 助手。")

    def _get_system_context(self) -> str:
        """获取系统环境上下文 (新增方法)"""
        try:
            sys_platform = f"{platform.system()} {platform.release()} ({platform.machine()})"
            py_version = sys.version.split()[0]
            cwd = os.getcwd()

            return f"""
## 运行时环境
- 操作系统: {sys_platform}
- Python 版本: {py_version}
- 当前工作路径 (CWD): {cwd}
"""
        except Exception as e:
            return f"\n[System Context Error: {e}]\n"

    def _get_time_context(self, last_interaction_time: float = None) -> str:
        """生成时间上下文块"""
        now = datetime.datetime.now()
        weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weekday_map[now.weekday()]

        # 判断时段
        hour = now.hour
        if 5 <= hour < 12:
            period = "上午"
        elif 12 <= hour < 18:
            period = "下午"
        elif 18 <= hour < 22:
            period = "晚上"
        else:
            period = "深夜"

        uptime = int(time.time() - self.start_time)
        uptime_str = f"{uptime // 3600}小时{(uptime % 3600) // 60}分钟"

        context = f"""
## 时间与状态
- 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} ({weekday} {period})
- 系统启动时间: {uptime_str}
"""
        if last_interaction_time:
            diff = int(time.time() - last_interaction_time)
            if diff < 60:
                since = "刚刚"
            else:
                since = f"{diff // 60}分钟前"
            context += f"- 上次交互: {since}\n"

        return context

    def get_system_prompt(self, last_interaction_time: float = None) -> str:
        """获取带有动态时间上下文的 System Prompt"""
        try:
            with open(self.prompt_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                content = f.read()

        # 统一换行符，避免 token 噪声
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # 防止首尾隐性字符影响 system 权重
        base_prompt = content.strip()

        system_block = self._get_system_context()
        time_block = self._get_time_context(last_interaction_time)
        return f"{base_prompt}\n{system_block}\n{time_block}"
