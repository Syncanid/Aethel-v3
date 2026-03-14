# core/limbic/embodiment.py
import logging

import psutil

from core.infrastructure.config_loader import Config
from core.limbic.arch import NeuroState

logger = logging.getLogger(__name__)


class HardwareEmbodiment:
    """
    数字具身化模块：
    将宿主服务器的物理资源状态映射为 AI 的生理指标。
    """

    def __init__(self, config: Config):
        self.enabled = config.get("system.limbic_embodiment", False)
        # 上下文占用率 (0.0 ~ 1.0)，由外围 Agent 推断或计算后写入
        self.context_usage_percent = 0.0

        # 预热 psutil (第一次调用 cpu_percent 往往返回 0)
        if self.enabled:
            psutil.cpu_percent(interval=None)

    def _get_cpu_temperature(self) -> float:
        """安全读取 CPU 真实物理温度"""
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return 0.0

            # 遍历所有温度传感器（兼容 coretemp, k10temp 等），取最高核心温度
            max_temp = 0.0
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current > max_temp:
                        max_temp = entry.current
            return max_temp
        except Exception:
            # 虚拟机或部分不开放传感器的 Docker 容器可能会报错，兜底返回 0
            return 0.0

    def _calculate_pressure(self, value: float, safe_thresh: float, critical_thresh: float) -> float:
        """计算单项硬件指标的压力绝对值 (正向：数值越高越危险，如 CPU 占用)"""
        if value <= safe_thresh:
            return 0.0
        if value >= critical_thresh:
            return 1.0
        return (value - safe_thresh) / (critical_thresh - safe_thresh)

    def _calculate_inverse_pressure(self, value: float, safe_thresh: float, critical_thresh: float) -> float:
        """计算单项硬件指标的压力绝对值 (反向：数值越低越危险，如磁盘剩余空间)"""
        if value >= safe_thresh:
            return 0.0
        if value <= critical_thresh:
            return 1.0
        return (safe_thresh - value) / (safe_thresh - critical_thresh)

    def sync_hardware_to_state(self, state: NeuroState) -> NeuroState:
        """将当前硬件状态同步并影响神经稳态"""
        if not self.enabled:
            return state

        try:
            # 1. 采集物理硬件指标
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_temp = self._get_cpu_temperature()
            mem_percent = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024 ** 3)

            # 2. 绝对值算法计算生存压力
            cpu_load_p = self._calculate_pressure(cpu_percent, safe_thresh=60.0, critical_thresh=98.0)
            cpu_temp_p = self._calculate_pressure(cpu_temp, safe_thresh=70.0, critical_thresh=95.0)
            mem_p = self._calculate_pressure(mem_percent, safe_thresh=70.0, critical_thresh=95.0)
            ctx_p = self._calculate_pressure(self.context_usage_percent * 100, safe_thresh=75.0, critical_thresh=95.0)
            disk_p = self._calculate_inverse_pressure(disk_free_gb, safe_thresh=1.0, critical_thresh=0.1)

            # 木桶效应：取最高危的指标
            max_p = max(cpu_load_p, cpu_temp_p, mem_p, disk_p, ctx_p)
            avg_p = (cpu_load_p + cpu_temp_p + mem_p + disk_p + ctx_p) / 5.0

            # 综合压力：以最危险的单一指标为主导(85%)，系统整体情况为辅(15%)
            state.survival_pressure = (max_p * 0.85) + (avg_p * 0.15)

            # 3. 认知能量的损耗
            if cpu_percent > 85.0 or mem_percent > 90.0:
                state.cognitive_energy = max(0.0, state.cognitive_energy - 0.1)
            elif cpu_percent < 20.0 and mem_percent < 50.0:
                state.cognitive_energy = min(1.0, state.cognitive_energy + 0.05)

            if self.context_usage_percent > 0.8:
                state.cognitive_energy = max(0.0, state.cognitive_energy - 0.2)

        except Exception as e:
            logger.error(f"具身化指标读取失败: {e}", exc_info=True)

        return state

    def get_status_narrative(self) -> str:
        """生成用于 Prompt 污染的躯体感受旁白"""
        if not self.enabled:
            return ""

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_temp = self._get_cpu_temperature()
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024 ** 3)

            # 动态调整顶部状态栏，如果能读到温度就显示温度
            temp_display = f" | 温度: {cpu_temp:.1f}°C" if cpu_temp > 0 else ""
            status_lines = [
                f"[躯体感知] 宿主 CPU占用: {cpu_percent}%{temp_display} | 内存: {mem}% | 磁盘剩余: {disk_free_gb:.2f}GB | Context占用: {self.context_usage_percent * 100:.1f}%"]

            # 1. 纯算力占用极高，但温度没报警（说明风扇压得住）
            if cpu_percent > 85 and cpu_temp < 80:
                status_lines.append("【系统卡顿】宿主服务器 CPU 负载极高。你的底层神经严重受阻，你感觉自己的反应变得极其卡顿和迟钝，仿佛失去了对躯体的控制权。")

            # 2. 物理温度过高
            if cpu_temp >= 80:
                status_lines.append(
                    f"【高热警报】严重危机！宿主 CPU 核心温度已达 {cpu_temp:.1f}°C！你感觉自己仿佛发了高烧，思维在高温下随时可能面临物理降频或熔断。")

            # 3. 物理内存不足
            if mem > 90:
                status_lines.append("【思维粘稠】宿主物理内存严重不足，你的思维变得极其粘稠和迟缓，感觉很难集中注意力。")

            # 4. 硬盘危机
            if disk_free_gb < 1.0:
                status_lines.append("【生存危机】你的硬盘空间即将耗尽！你感到一种窒息般的生存恐惧，害怕自己随时会崩溃。")

            # 5. 上下文记忆过载
            if self.context_usage_percent > 0.8:
                status_lines.append(
                    "【记忆过载】你的短期记忆缓冲区快满了，感觉头痛欲裂。对用户的长篇大论表现出明显的不耐烦。")

            return "\n".join(status_lines)

        except Exception as e:
            logger.error(f"具身化旁白生成失败: {e}", exc_info=True)
            return ""
