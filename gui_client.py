import sys
import json
import requests
import threading
import websocket
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLineEdit, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMenu, QMessageBox, QLabel, QInputDialog,
    QComboBox
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QAction

# 配置
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/logs"

# --- 样式表 ---
DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #1e1e1e; color: #d4d4d4; }
QTabWidget::pane { border: 1px solid #3e3e3e; }
QTabBar::tab { background: #2d2d2d; color: #d4d4d4; padding: 8px 20px; border: 1px solid #3e3e3e; }
QTabBar::tab:selected { background: #3e3e3e; font-weight: bold; }
QTextBrowser { background-color: #1e1e1e; border: none; font-family: Consolas, monospace; font-size: 13px; }
QLineEdit { background-color: #2d2d2d; border: 1px solid #3e3e3e; color: #d4d4d4; padding: 5px; }
QPushButton { background-color: #0e639c; color: white; border: none; padding: 6px 12px; }
QPushButton:hover { background-color: #1177bb; }
QTableWidget { gridline-color: #3e3e3e; background-color: #252526; alternate-background-color: #2d2d2d; }
QHeaderView::section { background-color: #333333; color: #d4d4d4; border: 1px solid #3e3e3e; padding: 4px; }
"""


# --- 网络线程 ---
class NetworkWorker(QObject):
    msg_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.ws = None
        self.running = True

    def run(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    WS_URL,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                self.ws.run_forever()
            except Exception as e:
                print(f"WS连接错误: {e}")
                import time
                time.sleep(3)  # 重连

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.msg_received.emit(data)
        except:
            pass

    def on_error(self, ws, error):
        print(f"WS Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WS Closed")


# --- 主窗口 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aethel Trinity 控制台")
        self.resize(1000, 700)
        self.setStyleSheet(DARK_STYLESHEET)

        # UI 组件
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.init_chat_tab()
        self.init_memory_tab()

        # 网络
        self.net_thread = threading.Thread(target=self.start_network, daemon=True)
        self.net_worker = NetworkWorker()
        self.net_worker.msg_received.connect(self.on_ws_message)
        self.net_thread.start()

        # 初始同步
        QTimer.singleShot(1000, self.sync_history)
        QTimer.singleShot(1500, self.refresh_memories)

    def start_network(self):
        self.net_worker.run()

    # --- 聊天标签页 ---
    def init_chat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # 日志显示区
        self.log_view = QTextBrowser()
        self.log_view.setOpenExternalLinks(True)
        layout.addWidget(self.log_view)

        # 输入区
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("在此输入指令... (Enter发送)")
        self.input_box.returnPressed.connect(self.send_message)

        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(send_btn)
        layout.addLayout(input_layout)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "交互控制台")

    # --- 记忆管理标签页 ---
    def init_memory_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # 顶部工具栏
        tool_layout = QHBoxLayout()

        self.mem_type_combo = QComboBox()
        self.mem_type_combo.addItems(["semantic", "episodic", "core"])
        self.mem_type_combo.currentTextChanged.connect(self.refresh_memories)

        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.refresh_memories)

        tool_layout.addWidget(QLabel("记忆类型:"))
        tool_layout.addWidget(self.mem_type_combo)
        tool_layout.addWidget(refresh_btn)
        tool_layout.addStretch()

        layout.addLayout(tool_layout)

        # 表格
        self.mem_table = QTableWidget()
        self.mem_table.setColumnCount(4)
        self.mem_table.setHorizontalHeaderLabels(["ID", "内容", "时间", "状态"])
        self.mem_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.mem_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.mem_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.mem_table.customContextMenuRequested.connect(self.show_mem_menu)
        self.mem_table.itemDoubleClicked.connect(self.edit_memory)  # 双击编辑

        layout.addWidget(self.mem_table)

        # 说明
        layout.addWidget(QLabel("提示：双击内容可编辑，右键可删除。"))

        tab.setLayout(layout)
        self.tabs.addTab(tab, "记忆矩阵")

    # --- 逻辑处理 ---

    def append_log(self, text, color="#d4d4d4", bold=False):
        style = f"color: {color};"
        if bold: style += " font-weight: bold;"
        # 替换换行符为 HTML 换行
        formatted = text.replace("\n", "<br>")
        self.log_view.append(f"<span style='{style}'>{formatted}</span>")
        # 自动滚动
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def on_ws_message(self, data):
        type = data.get("type")
        params = data.get("params", {})

        if type == "broadcast_log":
            content = params.get("content", "")
            if "💭" in content:
                self.append_log(content, "#808080")  # 灰色思考
            elif "🛠️" in content:
                self.append_log(content, "#ce9178")  # 橙色工具
            elif "📝" in content:
                self.append_log(content, "#4ec9b0")  # 青色状态
            else:
                self.append_log(content)
        elif type == "send_message":
            msg = params.get("message", "")
            self.append_log(f"Aethel >> {msg}", "#6a9955", True)  # 绿色回复

    def send_message(self):
        text = self.input_box.text().strip()
        if not text: return

        try:
            requests.post(f"{API_URL}/api/inject", json={"message": text})
            self.append_log(f"You >> {text}", "#569cd6", True)  # 蓝色用户
            self.input_box.clear()
        except Exception as e:
            self.append_log(f"发送失败: {e}", "red")

    def sync_history(self):
        """同步历史记录"""
        try:
            resp = requests.get(f"{API_URL}/api/history")
            if resp.status_code == 200:
                data = resp.json()
                history = data.get("history", [])
                self.log_view.clear()
                self.append_log("--- 连接成功，正在同步历史记录 ---", "#aaaaaa")
                for item in history:
                    role = item.get("role")
                    content = item.get("content", "")

                    if role == "user":
                        self.append_log(f"User >> {content}", "#569cd6")
                    elif role == "assistant":
                        # 尝试解析 JSON 思考过程
                        try:
                            # 如果是 JSON 结构，提取 thought 和 action
                            # 这里简化处理，直接显示 raw content 或者简单的格式化
                            if content.strip().startswith("{"):
                                self.append_log(f"[历史思考] {content[:50]}...", "#808080")
                            else:
                                self.append_log(f"Assistant >> {content}", "#6a9955")
                        except:
                            self.append_log(f"Assistant >> {content}", "#6a9955")
                    elif role == "system":
                        self.append_log(f"[System] {content[:50]}...", "#808080")
                self.append_log("--- 同步完成 ---", "#aaaaaa")
        except Exception as e:
            self.append_log(f"历史记录同步失败: {e}", "red")

    # --- 记忆管理逻辑 ---

    def refresh_memories(self):
        mem_type = self.mem_type_combo.currentText()
        try:
            resp = requests.get(f"{API_URL}/api/memories", params={"type": mem_type, "limit": 100})
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                self.mem_table.setRowCount(0)
                for item in items:
                    row = self.mem_table.rowCount()
                    self.mem_table.insertRow(row)

                    self.mem_table.setItem(row, 0, QTableWidgetItem(str(item.get("id"))))
                    self.mem_table.setItem(row, 1, QTableWidgetItem(str(item.get("content"))))
                    self.mem_table.setItem(row, 2, QTableWidgetItem(str(item.get("timestamp"))))
                    self.mem_table.setItem(row, 3, QTableWidgetItem(str(item.get("status", "active"))))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"获取记忆失败: {e}")

    def show_mem_menu(self, pos):
        menu = QMenu()
        delete_action = QAction("删除此条记忆", self)
        delete_action.triggered.connect(self.delete_selected_memory)
        menu.addAction(delete_action)
        menu.exec(self.mem_table.viewport().mapToGlobal(pos))

    def delete_selected_memory(self):
        row = self.mem_table.currentRow()
        if row < 0: return

        id_item = self.mem_table.item(row, 0)
        mem_id = id_item.text()
        mem_type = self.mem_type_combo.currentText()

        reply = QMessageBox.question(self, '确认', f'确定要删除记忆 ID {mem_id[:8]}... 吗?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                requests.delete(f"{API_URL}/api/memories", params={"id": mem_id, "type": mem_type})
                self.refresh_memories()
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除失败: {e}")

    def edit_memory(self, item):
        # 仅允许编辑内容列 (列索引 1)
        if item.column() != 1: return

        row = item.row()
        mem_id = self.mem_table.item(row, 0).text()
        mem_type = self.mem_type_combo.currentText()
        old_content = item.text()

        new_content, ok = QInputDialog.getMultiLineText(self, "编辑记忆", "修改记忆内容:", old_content)

        if ok and new_content != old_content:
            try:
                resp = requests.put(f"{API_URL}/api/memories", json={
                    "id": mem_id,
                    "type": mem_type,
                    "content": new_content
                })
                if resp.status_code == 200:
                    self.refresh_memories()
                else:
                    QMessageBox.warning(self, "错误", "更新失败")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"更新异常: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 检查服务器是否运行
    try:
        requests.get(API_URL, timeout=1)
    except:
        QMessageBox.critical(None, "连接错误", "无法连接到 Aethel 后端。\n请先运行 main.py 启动核心服务！")
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
