# core/gui/dashboard.py
import json
import sys

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QTextEdit, QLabel, QProgressBar, QGroupBox, QSplitter)

from core.gui.monitor_registry import monitor_registry


class MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aethel Trinity - 实时神经监控矩阵")
        self.resize(1200, 800)

        # 定时器：每 500ms 刷新一次界面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(500)

        # UI 组件缓存，避免重复创建
        self.text_widgets = {}
        self.metric_widgets = {}

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # === 左侧：数值监控 (Limbic System & Metrics) ===
        left_panel = QGroupBox("生理指标")
        left_layout = QVBoxLayout()
        self.metrics_container = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_container)
        self.metrics_layout.addStretch()  # 顶部对齐
        left_layout.addWidget(self.metrics_container)
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(300)

        # === 右侧：文本监控 (Tabs for Scratchpad, Prompt, etc.) ===
        self.tabs = QTabWidget()

        # 布局组合
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

    def update_ui(self):
        """周期性从 Registry 拉取数据并更新 UI"""

        # 1. 更新数值指标 (进度条)
        metrics = monitor_registry.get_metric_data()
        for label, data in metrics.items():
            if label not in self.metric_widgets:
                # 动态创建进度条
                container = QWidget()
                l = QVBoxLayout(container)
                l.setContentsMargins(0, 5, 0, 5)
                lbl = QLabel(f"{label}: 0.00")
                pbar = QProgressBar()
                pbar.setTextVisible(False)
                pbar.setFixedHeight(15)
                l.addWidget(lbl)
                l.addWidget(pbar)
                self.metrics_layout.insertWidget(self.metrics_layout.count() - 1, container)

                self.metric_widgets[label] = {"label": lbl, "bar": pbar}

            # 更新值
            val = data["value"]
            w = self.metric_widgets[label]
            w["label"].setText(f"{label}: {val:.4f}")

            # 归一化到 0-100
            range_val = data["max"] - data["min"]
            if range_val == 0: range_val = 1
            percent = int((val - data["min"]) / range_val * 100)
            w["bar"].setValue(max(0, min(100, percent)))
            w["bar"].setStyleSheet("QProgressBar::chunk { background-color: #69F0AE; }")  # 绿色

        # 2. 更新文本数据 (Tabs)
        text_groups = monitor_registry.get_text_data()

        # 检查是否需要创建新的 Tab
        current_tabs = set(self.text_widgets.keys())
        incoming_tabs = set(text_groups.keys())

        for category in incoming_tabs:
            if category not in current_tabs:
                # 创建新 Tab
                tab_widget = QWidget()
                layout = QVBoxLayout(tab_widget)
                self.tabs.addTab(tab_widget, category)
                self.text_widgets[category] = {}
                # 暂存 layout以便后续添加具体的文本框
                self.text_widgets[category]["_layout"] = layout

            # 遍历该分类下的数据项
            items = text_groups[category]  # list of (label, callback)
            layout = self.text_widgets[category]["_layout"]

            for label, callback in items:
                widget_key = f"{category}_{label}"

                if widget_key not in self.text_widgets:
                    # 创建文本框
                    gb = QGroupBox(label)
                    gbl = QVBoxLayout(gb)
                    txt = QTextEdit()
                    txt.setReadOnly(True)
                    txt.setStyleSheet("background-color: #263238; color: #ECEFF1; font-family: Consolas;")
                    gbl.addWidget(txt)
                    layout.addWidget(gb)
                    self.text_widgets[widget_key] = txt

                # 获取数据并更新
                try:
                    raw_data = callback()
                    if isinstance(raw_data, (dict, list)):
                        content = json.dumps(raw_data, indent=2, ensure_ascii=False)
                    else:
                        content = str(raw_data)

                    # 只有内容变化时才 setHtml/setText 以避免滚动条跳动
                    # 这里简单处理，直接 setText
                    current_text = self.text_widgets[widget_key].toPlainText()
                    if content != current_text:
                        self.text_widgets[widget_key].setText(content)

                except Exception as e:
                    self.text_widgets[widget_key].setText(f"Error fetching data: {e}")


def run_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MonitorWindow()
    window.show()
    return app.exec()
