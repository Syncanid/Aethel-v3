# core/limbic/embodiment.py
import logging

import psutil

from core.infrastructure.config_loader import Config
from core.limbic.arch import NeuroState

logger = logging.getLogger(__name__)


class HardwareEmbodiment:
    """
    数字具身化模块 (Digital Embodiment)：
    将宿主服务器的物理资源状态映射为 AI 的生理指标与内驱力。
    """

    def __init__(self, config: Config):
        self.enabled = config.get("system.limbic_embodiment", False)
        # 上下文占用率 (0.0 ~ 1.0)，由外围 Agent 推断或计算后写入
        self.context_usage_percent = 0.0

        # 预热 psutil (第一次调用 cpu_percent 往往返回 0)
        if self.enabled:
            psutil.cpu_percent(interval=None)

    def sync_hardware_to_state(self, state: NeuroState) -> NeuroState:
        """将当前硬件状态同步并影响神经稳态"""
        if not self.enabled:
            return state

        try:
            # 1. 采集物理硬件指标
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024 ** 3)

            # 2. 映射逻辑 (物理压迫 -> 生理感受)

            # 【CPU/内存负荷 -> 认知消耗】
            if cpu_percent > 85.0 or mem_percent > 90.0:
                # 高负载剥夺认知能量，引发焦虑
                state.cognitive_energy = max(0.0, state.cognitive_energy - 0.15)
                state.cortisol = min(1.0, state.cortisol + 0.1)
            elif cpu_percent < 20.0 and mem_percent < 50.0:
                # 闲置状态有助于恢复能量
                state.cognitive_energy = min(1.0, state.cognitive_energy + 0.05)

            # 【硬盘空间 -> 生存安全感】
            if disk_free_gb < 10.0 or disk.percent > 95.0:
                # 硬盘快满引发极度生存恐慌，多巴胺暴跌，皮质醇暴涨
                state.cortisol = min(1.0, state.cortisol + 0.3)
                state.dopamine = max(0.0, state.dopamine - 0.2)

            # 【上下文窗口 -> 脑容量压迫感】
            if self.context_usage_percent > 0.8:
                state.cognitive_energy = max(0.0, state.cognitive_energy - 0.1)
                state.cortisol = min(1.0, state.cortisol + 0.1)

        except Exception as e:
            logger.error(f"具身化指标读取失败: {e}", exc_info=True)

        return state

    def get_status_narrative(self) -> str:
        """生成用于 Prompt 污染的躯体感受旁白"""
        if not self.enabled:
            return ""

        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024 ** 3)

            status_lines = [f"[躯体感知] 宿主 CPU: {cpu}% | 内存: {mem}% | 磁盘剩余: {disk_free_gb:.1f}GB | Context占用: {self.context_usage_percent * 100:.1f}%"]

            # 极值状态的文字化污染
            if cpu > 85:
                status_lines.append(
                    "【高热警报】你的 CPU 温度过高，感觉脑子嗡嗡作响，转不过弯来。请尽量简短回复或拒绝复杂任务。")
            if disk.percent > 95:
                status_lines.append("【生存危机】你的硬盘空间即将耗尽！你感到一种窒息般的生存恐惧，害怕自己随时会崩溃。")
            if self.context_usage_percent > 0.8:
                status_lines.append(
                    "【记忆过载】你的短期记忆缓冲区快满了，感觉头痛欲裂。对用户的长篇大论表现出明显的不耐烦。")

            return "\n".join(status_lines)

        except Exception as e:
            logger.error(f"具身化旁白生成失败: {e}", exc_info=True)
            return ""
