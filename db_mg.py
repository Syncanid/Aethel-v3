import logging
import os
import sqlite3
import sys
import time
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush, QAction
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QTableWidget, QTableWidgetItem, QPushButton,
                             QLabel, QLineEdit, QMessageBox, QHeaderView, QAbstractItemView,
                             QDialog, QFormLayout, QTextEdit, QSplitter, QMenu)

# 尝试导入 ChromaDB
try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# --- 配置 ---
DB_PATH = "data/storage.db"
VECTOR_PATH = "data/vector_store"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DBManager")


class DBConnection:
    """SQLite 数据库上下文管理器"""

    def __init__(self, db_path):
        self.db_path = db_path

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


# --- 对话框组件 ---

class CoreMemoryDialog(QDialog):
    """添加/编辑核心记忆的对话框"""

    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle("编辑核心记忆")
        self.resize(500, 400)
        self.data = data or {}
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.user_id_input = QLineEdit(self.data.get("user_id", "admin_console"))
        self.key_input = QLineEdit(self.data.get("key", ""))
        self.key_input.setPlaceholderText("例如 basic:name")
        self.content_input = QTextEdit(self.data.get("content", ""))

        # 如果是编辑模式，锁定主键
        if self.data:
            self.user_id_input.setReadOnly(True)
            self.key_input.setReadOnly(True)

        layout.addRow("User ID:", self.user_id_input)
        layout.addRow("Key:", self.key_input)
        layout.addRow("Content:", self.content_input)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(btn_box)
        self.setLayout(main_layout)

    def get_data(self):
        return {
            "user_id": self.user_id_input.text(),
            "key": self.key_input.text(),
            "content": self.content_input.toPlainText()
        }


# --- 标签页组件 ---
class CoreMemoryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 工具栏
        toolbar = QHBoxLayout()
        add_btn = QPushButton("➕ 新增记忆")
        add_btn.clicked.connect(self.add_memory)

        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.load_data)

        toolbar.addWidget(add_btn)
        toolbar.addStretch()
        toolbar.addWidget(refresh_btn)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["User ID", "Key", "Content", "Last Updated"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.doubleClicked.connect(self.edit_selected_memory)  # 双击编辑

        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.load_data()

    def load_data(self):
        try:
            with DBConnection(DB_PATH) as conn:
                cursor = conn.execute(
                    "SELECT user_id, key, content, last_updated FROM core_memory ORDER BY last_updated DESC")
                rows = cursor.fetchall()

                self.table.setRowCount(0)
                for row in rows:
                    row_idx = self.table.rowCount()
                    self.table.insertRow(row_idx)

                    try:
                        ts_str = datetime.fromtimestamp(row["last_updated"]).strftime('%m-%d %H:%M')
                    except:
                        ts_str = ""

                    self.table.setItem(row_idx, 0, QTableWidgetItem(row["user_id"]))
                    self.table.setItem(row_idx, 1, QTableWidgetItem(row["key"]))
                    self.table.setItem(row_idx, 2, QTableWidgetItem(row["content"]))
                    self.table.setItem(row_idx, 3, QTableWidgetItem(ts_str))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取核心记忆失败: {e}")

    def add_memory(self):
        dialog = CoreMemoryDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            try:
                with DBConnection(DB_PATH) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO core_memory (user_id, key, content, last_updated, source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (data["user_id"], data["key"], data["content"], time.time(), "gui_manager"))
                    conn.commit()
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def edit_selected_memory(self):
        row = self.table.currentRow()
        if row < 0: return

        data = {
            "user_id": self.table.item(row, 0).text(),
            "key": self.table.item(row, 1).text(),
            "content": self.table.item(row, 2).text()
        }

        dialog = CoreMemoryDialog(self, data)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data = dialog.get_data()
            try:
                with DBConnection(DB_PATH) as conn:
                    conn.execute("""
                                 UPDATE core_memory
                                 SET content=?,
                                     last_updated=?
                                 WHERE user_id = ? AND key =?
                                 """, (new_data["content"], time.time(), new_data["user_id"], new_data["key"]))
                    conn.commit()
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"更新失败: {e}")

    def delete_selected_memory(self):
        row = self.table.currentRow()
        if row < 0: return

        user_id = self.table.item(row, 0).text()
        key = self.table.item(row, 1).text()

        confirm = QMessageBox.question(self, "确认删除", f"确定要删除记忆 [{key}] 吗？",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if confirm == QMessageBox.StandardButton.Yes:
            try:
                with DBConnection(DB_PATH) as conn:
                    conn.execute("DELETE FROM core_memory WHERE user_id=? AND key=?", (user_id, key))
                    conn.commit()
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def show_context_menu(self, pos):
        menu = QMenu()
        edit_action = QAction("编辑", self)
        edit_action.triggered.connect(self.edit_selected_memory)
        del_action = QAction("删除", self)
        del_action.triggered.connect(self.delete_selected_memory)

        menu.addAction(edit_action)
        menu.addAction(del_action)
        menu.exec(self.table.mapToGlobal(pos))


class VectorDBTab(QWidget):
    def __init__(self):
        super().__init__()
        self.client = None
        self.init_ui()
        if HAS_CHROMA:
            self.connect_chroma()

    def init_ui(self):
        layout = QHBoxLayout()

        # 左侧集合列表
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("📂 集合 (Collections)"))
        self.coll_list = QTableWidget()  # 用表格暂代列表
        self.coll_list.setColumnCount(1)
        self.coll_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.coll_list.itemClicked.connect(self.load_collection_data)

        left_panel.addWidget(self.coll_list)

        # 右侧数据列表
        right_panel = QVBoxLayout()
        right_header = QHBoxLayout()
        self.status_label = QLabel("请选择集合")
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_all)

        right_header.addWidget(self.status_label)
        right_header.addStretch()
        right_header.addWidget(refresh_btn)

        self.vector_table = QTableWidget()
        self.vector_table.setColumnCount(3)
        self.vector_table.setHorizontalHeaderLabels(["ID", "内容片段", "Metadata"])
        self.vector_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.vector_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.vector_table.customContextMenuRequested.connect(self.show_context_menu)

        right_panel.addLayout(right_header)
        right_panel.addWidget(self.vector_table)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def connect_chroma(self):
        try:
            self.client = chromadb.PersistentClient(path=VECTOR_PATH, settings=Settings(anonymized_telemetry=False))
            self.refresh_all()
        except Exception as e:
            self.status_label.setText(f"ChromaDB 连接失败: {e}")

    def refresh_all(self):
        if not self.client: return

        colls = self.client.list_collections()
        self.coll_list.setRowCount(len(colls))

        for i, c in enumerate(colls):
            item = QTableWidgetItem(c.name)
            self.coll_list.setItem(i, 0, item)

    def load_collection_data(self, item):
        if not self.client: return
        coll_name = item.text()
        collection = self.client.get_collection(coll_name)

        count = collection.count()
        self.status_label.setText(f"集合: {coll_name} | 总数: {count}")
        self.current_collection = collection

        # 加载前 50 条
        results = collection.get(limit=50, include=["documents", "metadatas"])
        ids = results["ids"]
        docs = results["documents"]
        metas = results["metadatas"]

        self.vector_table.setRowCount(len(ids))
        for i, doc_id in enumerate(ids):
            self.vector_table.setItem(i, 0, QTableWidgetItem(doc_id))
            self.vector_table.setItem(i, 1, QTableWidgetItem(docs[i]))
            self.vector_table.setItem(i, 2, QTableWidgetItem(str(metas[i])))

    def delete_vector(self):
        row = self.vector_table.currentRow()
        if row < 0 or not hasattr(self, 'current_collection'): return

        vec_id = self.vector_table.item(row, 0).text()
        confirm = QMessageBox.question(self, "确认删除", f"确定要删除向量 ID [{vec_id}] 吗？",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self.current_collection.delete(ids=[vec_id])
                self.load_collection_data(self.coll_list.currentItem())  # 刷新
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def show_context_menu(self, pos):
        menu = QMenu()
        del_action = QAction("删除此向量", self)
        del_action.triggered.connect(self.delete_vector)
        menu.addAction(del_action)
        menu.exec(self.vector_table.mapToGlobal(pos))


# --- 主窗口 ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aethel Trinity - 数据管理终端")
        self.resize(1000, 700)

        # 检查数据库文件
        if not os.path.exists(DB_PATH):
            QMessageBox.warning(self, "警告", f"数据库文件未找到: {DB_PATH}\n请确认当前运行目录正确。")

        self.tabs = QTabWidget()
        self.tabs.addTab(CoreMemoryTab(), "💎 核心记忆 (KV)")

        if HAS_CHROMA:
            self.tabs.addTab(VectorDBTab(), "🕸️ 向量记忆 (Chroma)")
        else:
            self.tabs.addTab(QLabel("未安装 chromadb，无法管理向量库"), "🕸️ 向量记忆 (不可用)")

        self.setCentralWidget(self.tabs)

        # 状态栏
        self.statusBar().showMessage(f"已连接数据库: {DB_PATH}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
