import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QTableWidget, QTableWidgetItem, QPushButton,
                             QLabel, QLineEdit, QMessageBox, QHeaderView, QAbstractItemView,
                             QDialog, QFormLayout, QTextEdit, QSplitter, QMenu, QComboBox)

# --- 配置 ---
DB_PATH = "data/storage.db"
VECTOR_PATH = "data/vector_store"

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DBManagerV2")

# 尝试导入 ChromaDB
try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


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


class DynamicRowDialog(QDialog):
    """
    通用行编辑/新增对话框
    自动根据表结构生成输入框，并支持 JSON 格式化
    """

    def __init__(self, parent=None, table_name="", columns_info=None, row_data=None):
        super().__init__(parent)
        self.table_name = table_name
        self.columns_info = columns_info  # List of dict: {name, type, pk}
        self.row_data = row_data or {}  # dict: {col_name: value}
        self.is_edit = row_data is not None
        self.widgets = {}

        mode = "编辑" if self.is_edit else "新增"
        self.setWindowTitle(f"{mode}记录 - {table_name}")
        self.resize(600, 500)
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        for col in self.columns_info:
            col_name = col['name']
            col_type = col['type'].upper()
            is_pk = col['pk'] > 0

            val = self.row_data.get(col_name, "")
            if val is None: val = ""

            label = QLabel(f"{col_name} ({col_type})")

            # 智能判断控件类型
            # 1. 如果是 JSON 或长文本，使用 TextEdit
            is_json_col = "JSON" in col_name.upper() or "CONTENT" in col_name.upper() or (
                    isinstance(val, str) and (val.strip().startswith("{") or val.strip().startswith("[")))

            if is_json_col:
                widget = QTextEdit()
                # 尝试格式化 JSON 显示
                if isinstance(val, str) and val:
                    try:
                        formatted_json = json.dumps(json.loads(val), indent=2, ensure_ascii=False)
                        widget.setText(formatted_json)
                    except:
                        widget.setText(str(val))
                else:
                    widget.setText(str(val))
                widget.setMinimumHeight(100)
            else:
                widget = QLineEdit()
                widget.setText(str(val))
                if is_pk and self.is_edit:
                    widget.setReadOnly(True)  # 主键在编辑模式下通常只读
                    widget.setStyleSheet("background-color: #f0f0f0;")

            self.widgets[col_name] = {"widget": widget, "is_json": is_json_col}
            layout.addRow(label, widget)

        btn_box = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.validate_and_accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(btn_box)
        self.setLayout(main_layout)

    def validate_and_accept(self):
        # 简单验证 JSON 格式
        for col_name, info in self.widgets.items():
            if info["is_json"]:
                text = info["widget"].toPlainText().strip()
                if text:
                    try:
                        # 尝试压缩回单行存储，或者保持格式（视需求，这里压缩以节省空间）
                        json_obj = json.loads(text)
                        # info["widget"].setText(json.dumps(json_obj, ensure_ascii=False))
                    except json.JSONDecodeError as e:
                        QMessageBox.warning(self, "格式错误", f"列 '{col_name}' 的内容不是有效的 JSON:\n{e}")
                        return
        self.accept()

    def get_data(self):
        data = {}
        for col_name, info in self.widgets.items():
            widget = info["widget"]
            if isinstance(widget, QTextEdit):
                val = widget.toPlainText()
                # 如果是 JSON 字段，保存时压缩去空格（可选）
                if info["is_json"] and val:
                    try:
                        val = json.dumps(json.loads(val), ensure_ascii=False)
                    except:
                        pass
            else:
                val = widget.text()
            data[col_name] = val
        return data


class UniversalTableTab(QWidget):
    """
    通用表管理标签页
    """

    def __init__(self, table_name):
        super().__init__()
        self.table_name = table_name
        self.columns_info = []  # [{"cid":0, "name":"id", "type":"INTEGER", ...}]
        self.pk_names = []
        self.init_data_schema()
        self.init_ui()

    def init_data_schema(self):
        try:
            with DBConnection(DB_PATH) as conn:
                # 获取表结构
                cursor = conn.execute(f"PRAGMA table_info({self.table_name})")
                self.columns_info = [dict(row) for row in cursor.fetchall()]
                self.pk_names = [col['name'] for col in self.columns_info if col['pk'] > 0]
        except Exception as e:
            logger.error(f"Failed to load schema for {self.table_name}: {e}")

    def init_ui(self):
        layout = QVBoxLayout()

        # 工具栏
        toolbar = QHBoxLayout()
        add_btn = QPushButton("➕ 新增记录")
        add_btn.clicked.connect(self.add_record)

        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.load_data)

        # 简单的筛选框
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("所有列", "all")
        for col in self.columns_info:
            self.filter_combo.addItem(col['name'], col['name'])

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索...")
        self.search_input.returnPressed.connect(self.load_data)
        search_btn = QPushButton("🔍 搜索")
        search_btn.clicked.connect(self.load_data)

        toolbar.addWidget(add_btn)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("筛选:"))
        toolbar.addWidget(self.filter_combo)
        toolbar.addWidget(self.search_input)
        toolbar.addWidget(search_btn)
        toolbar.addWidget(refresh_btn)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.columns_info))
        self.table.setHorizontalHeaderLabels([c['name'] for c in self.columns_info])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # 禁止直接在格子编辑，使用弹窗
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.doubleClicked.connect(self.edit_selected_record)

        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.load_data()

    def load_data(self):
        search_term = self.search_input.text().strip()
        search_col = self.filter_combo.currentData()

        sql = f"SELECT * FROM {self.table_name}"
        params = []

        if search_term:
            if search_col == "all":
                conditions = [f"{col['name']} LIKE ?" for col in self.columns_info]
                sql += " WHERE " + " OR ".join(conditions)
                params = [f"%{search_term}%"] * len(self.columns_info)
            else:
                sql += f" WHERE {search_col} LIKE ?"
                params = [f"%{search_term}%"]

        # 默认按最后一个 PK 倒序，或者第一列倒序
        sort_col = self.pk_names[-1] if self.pk_names else self.columns_info[0]['name']
        sql += f" ORDER BY {sort_col} DESC LIMIT 100"

        try:
            with DBConnection(DB_PATH) as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                self.table.setRowCount(0)
                for row in rows:
                    row_idx = self.table.rowCount()
                    self.table.insertRow(row_idx)
                    for col_idx, col_info in enumerate(self.columns_info):
                        val = row[col_info['name']]
                        # 时间戳格式化
                        if "time" in col_info['name'].lower() and isinstance(val, (int, float)) and val > 1000000000:
                            try:
                                val_display = datetime.fromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                val_display = str(val)
                        else:
                            val_display = str(val) if val is not None else "NULL"

                        # 截断过长文本
                        if len(val_display) > 100:
                            val_display = val_display[:100] + "..."

                        item = QTableWidgetItem(val_display)
                        item.setToolTip(str(val))  # 鼠标悬停显示全文
                        self.table.setItem(row_idx, col_idx, item)
        except Exception as e:
            QMessageBox.critical(self, "查询错误", f"查询失败: {e}")

    def add_record(self):
        dialog = DynamicRowDialog(self, self.table_name, self.columns_info)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            cols = data.keys()
            placeholders = ",".join(["?" for _ in cols])
            col_str = ",".join(cols)
            values = list(data.values())

            try:
                with DBConnection(DB_PATH) as conn:
                    conn.execute(f"INSERT INTO {self.table_name} ({col_str}) VALUES ({placeholders})", values)
                    conn.commit()
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"插入失败: {e}")

    def edit_selected_record(self):
        row = self.table.currentRow()
        if row < 0: return

        # 获取当前行的完整数据（需要重新查询以获取未截断的数据）
        # 构建 WHERE 子句使用 PK
        if not self.pk_names:
            QMessageBox.warning(self, "警告", "该表没有主键，无法精确定位行进行编辑。")
            return

        pk_criteria = {}
        for pk in self.pk_names:
            # 找到 PK 在 table 中的列索引
            col_idx = next(i for i, c in enumerate(self.columns_info) if c['name'] == pk)
            val_display = self.table.item(row, col_idx).text()
            # 注意：这里取的是显示值，如果是时间戳被格式化了可能会有问题，但在 DynamicRowDialog 里主要靠 toolTip 或者重新 query
            # 更严谨的做法是 hidden data，这里简化处理，假设 PK 很少被格式化
            pk_criteria[pk] = val_display

            # 查询完整行数据
        where_clause = " AND ".join([f"{k}=?" for k in pk_criteria.keys()])
        pk_values = list(pk_criteria.values())

        full_row_data = {}
        try:
            with DBConnection(DB_PATH) as conn:
                cursor = conn.execute(f"SELECT * FROM {self.table_name} WHERE {where_clause}", pk_values)
                row_obj = cursor.fetchone()
                if row_obj:
                    full_row_data = dict(row_obj)
                else:
                    QMessageBox.warning(self, "错误", "未在数据库中找到该行（可能已被删除）。")
                    return
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取行数据失败: {e}")
            return

        dialog = DynamicRowDialog(self, self.table_name, self.columns_info, full_row_data)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_data = dialog.get_data()

            # 构建 UPDATE 语句
            set_clause = ", ".join([f"{k}=?" for k in new_data.keys()])
            values = list(new_data.values()) + pk_values

            try:
                with DBConnection(DB_PATH) as conn:
                    conn.execute(f"UPDATE {self.table_name} SET {set_clause} WHERE {where_clause}", values)
                    conn.commit()
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"更新失败: {e}")

    def delete_selected_record(self):
        row = self.table.currentRow()
        if row < 0: return

        if not self.pk_names:
            QMessageBox.warning(self, "错误", "无主键表不支持删除操作。")
            return

        pk_vals = []
        pk_display = []
        for pk in self.pk_names:
            col_idx = next(i for i, c in enumerate(self.columns_info) if c['name'] == pk)
            val = self.table.item(row, col_idx).text()
            pk_vals.append(val)
            pk_display.append(f"{pk}={val}")

        confirm = QMessageBox.question(self, "确认删除", f"确定要删除 [{', '.join(pk_display)}] 吗？",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if confirm == QMessageBox.StandardButton.Yes:
            where_clause = " AND ".join([f"{pk}=?" for pk in self.pk_names])
            try:
                with DBConnection(DB_PATH) as conn:
                    conn.execute(f"DELETE FROM {self.table_name} WHERE {where_clause}", pk_vals)
                    conn.commit()
                self.load_data()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def show_context_menu(self, pos):
        menu = QMenu()
        edit_action = QAction("编辑", self)
        edit_action.triggered.connect(self.edit_selected_record)
        del_action = QAction("删除", self)
        del_action.triggered.connect(self.delete_selected_record)

        menu.addAction(edit_action)
        menu.addAction(del_action)
        menu.exec(self.table.mapToGlobal(pos))


class VectorDBTab(QWidget):
    """
    向量数据库管理 (保持原有功能)
    """

    def __init__(self):
        super().__init__()
        self.client = None
        self.init_ui()
        if HAS_CHROMA:
            self.connect_chroma()

    def init_ui(self):
        layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("📂 集合 (Collections)"))
        self.coll_list = QTableWidget()
        self.coll_list.setColumnCount(1)
        self.coll_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.coll_list.itemClicked.connect(self.load_collection_data)
        left_panel.addWidget(self.coll_list)

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
        try:
            colls = self.client.list_collections()
            self.coll_list.setRowCount(len(colls))
            for i, c in enumerate(colls):
                item = QTableWidgetItem(c.name)
                self.coll_list.setItem(i, 0, item)
        except Exception as e:
            logger.error(f"Chroma refresh failed: {e}")

    def load_collection_data(self, item):
        if not self.client: return
        coll_name = item.text()
        collection = self.client.get_collection(coll_name)

        count = collection.count()
        self.status_label.setText(f"集合: {coll_name} | 总数: {count}")
        self.current_collection = collection

        results = collection.get(limit=50, include=["documents", "metadatas"])
        ids = results["ids"]
        docs = results["documents"]
        metas = results["metadatas"]

        self.vector_table.setRowCount(len(ids))
        for i, doc_id in enumerate(ids):
            self.vector_table.setItem(i, 0, QTableWidgetItem(doc_id))
            doc_preview = docs[i][:100] + "..." if docs[i] and len(docs[i]) > 100 else docs[i]
            self.vector_table.setItem(i, 1, QTableWidgetItem(doc_preview))
            self.vector_table.setItem(i, 2, QTableWidgetItem(str(metas[i])))

    def delete_vector(self):
        row = self.vector_table.currentRow()
        if row < 0 or not hasattr(self, 'current_collection'): return
        vec_id = self.vector_table.item(row, 0).text()
        if QMessageBox.question(self, "删除", f"删除向量 {vec_id}?") == QMessageBox.StandardButton.Yes:
            self.current_collection.delete(ids=[vec_id])
            self.load_collection_data(self.coll_list.currentItem())

    def show_context_menu(self, pos):
        menu = QMenu()
        del_action = QAction("删除此向量", self)
        del_action.triggered.connect(self.delete_vector)
        menu.addAction(del_action)
        menu.exec(self.vector_table.mapToGlobal(pos))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aethel Trinity - 通用数据管理终端 v2")
        self.resize(1100, 750)

        if not os.path.exists(DB_PATH):
            QMessageBox.warning(self, "警告", f"数据库文件未找到: {DB_PATH}\n请确认当前运行目录正确。")

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 1. 自动发现并加载 SQLite 表
        self.load_sqlite_tabs()

        # 2. 加载向量数据库 Tab
        if HAS_CHROMA:
            self.tabs.addTab(VectorDBTab(), "🕸️ 向量记忆 (Chroma)")
        else:
            self.tabs.addTab(QLabel("未安装 chromadb，无法管理向量库"), "🕸️ 向量记忆 (不可用)")

        self.statusBar().showMessage(f"已连接: {DB_PATH}")

    def load_sqlite_tabs(self):
        try:
            with DBConnection(DB_PATH) as conn:
                # 获取所有表名
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]

                # 定义图标/Emoji映射，让界面好看点
                icon_map = {
                    "core_memory": "💎",
                    "chat_logs": "💬",
                    "social_users": "👥",
                    "neuro_states": "🧠",
                    "memory_archival_status": "📦"
                }

                for table in tables:
                    icon = icon_map.get(table, "📄")
                    self.tabs.addTab(UniversalTableTab(table), f"{icon} {table}")

        except Exception as e:
            QMessageBox.critical(self, "初始化失败", f"无法读取数据库表结构: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置深色字体优化（可选）
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
