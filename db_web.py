import json
import logging
import os
import sqlite3
import sys
from typing import List, Dict, Any, Optional

import aiosqlite
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
HOST = "0.0.0.0"
PORT = 8088

# --- 日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebDB")

app = FastAPI(title="Aethel Web DB Manager")

# 允许跨域 (方便调试)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 辅助类 ---
class UpdateRowRequest(BaseModel):
    pk_data: Dict[str, Any]
    new_data: Dict[str, Any]


class InsertRowRequest(BaseModel):
    data: Dict[str, Any]


class DeleteRowRequest(BaseModel):
    pk_data: Dict[str, Any]


# --- 路由: 页面 ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aethel Trinity 数据终端</title>
    <link href="https://jsd.cdn.zzko.cn/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://jsd.cdn.zzko.cn/npm/vue@3.3.4/dist/vue.global.prod.js"></script>
    <style>
        body { background-color: #f8f9fa; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        .sidebar { min-height: 100vh; background-color: #343a40; color: #fff; }
        .nav-link { color: rgba(255,255,255,.75); cursor: pointer; }
        .nav-link:hover, .nav-link.active { color: #fff; background-color: rgba(255,255,255,.1); }
        .code-cell { font-family: Consolas, monospace; font-size: 0.85em; max-height: 100px; overflow-y: auto; white-space: pre-wrap; }
        .json-editor { font-family: Consolas, monospace; min-height: 200px; }
        .modal-lg { max-width: 800px; }
        .badge-pk { background-color: #ffc107; color: #000; font-size: 0.7em; margin-left: 5px; }
    </style>
</head>
<body>
<div id="app" class="d-flex">
    <div class="sidebar p-3" style="width: 280px; flex-shrink: 0;">
        <h4 class="mb-4 text-center">🧠 Aethel DB</h4>

        <h6 class="text-uppercase text-muted small">SQLite Tables</h6>
        <ul class="nav flex-column mb-4">
            <li class="nav-item" v-for="t in tables" :key="t">
                <a class="nav-link" :class="{active: currentTab==='sqlite' && currentTable===t}" @click="loadTable(t)">
                    📄 {{ t }}
                </a>
            </li>
        </ul>

        <h6 class="text-uppercase text-muted small">Vector Store</h6>
        <ul class="nav flex-column">
            <li class="nav-item" v-if="!hasChroma">
                <span class="nav-link text-muted">🚫 未安装 Chroma</span>
            </li>
            <li class="nav-item" v-else v-for="c in collections" :key="c">
                <a class="nav-link" :class="{active: currentTab==='vector' && currentCollection===c}" @click="loadCollection(c)">
                    🕸️ {{ c }}
                </a>
            </li>
        </ul>
    </div>

    <div class="flex-grow-1 p-4" style="height: 100vh; overflow-y: auto;">

        <div v-if="currentTab === 'sqlite' && currentTable">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3>表: {{ currentTable }}</h3>
                <div>
                    <input type="text" class="form-control d-inline-block w-auto me-2" v-model="searchQuery" @keyup.enter="refreshTable()" placeholder="搜索...">
                    <button class="btn btn-primary btn-sm me-2" @click="openEditModal(null)">➕ 新增</button>
                    <button class="btn btn-secondary btn-sm" @click="refreshTable()">🔄 刷新</button>
                </div>
            </div>

            <div class="card shadow-sm">
                <div class="table-responsive">
                    <table class="table table-hover table-striped align-middle mb-0">
                        <thead class="table-dark">
                            <tr>
                                <th v-for="col in columns" :key="col.name">
                                    {{ col.name }}
                                    <span v-if="col.pk" class="badge badge-pk">PK</span>
                                </th>
                                <th style="width: 100px;">操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(row, idx) in tableData" :key="idx">
                                <td v-for="col in columns" :key="col.name">
                                    <div class="code-cell">{{ formatCell(row[col.name]) }}</div>
                                </td>
                                <td>
                                    <button class="btn btn-outline-primary btn-sm me-1" @click="openEditModal(row)">✏️</button>
                                    <button class="btn btn-outline-danger btn-sm" @click="deleteRow(row)">🗑️</button>
                                </td>
                            </tr>
                            <tr v-if="tableData.length === 0">
                                <td :colspan="columns.length + 1" class="text-center p-3 text-muted">无数据</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div v-if="currentTab === 'vector' && currentCollection">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3>集合: {{ currentCollection }}</h3>
                <button class="btn btn-secondary btn-sm" @click="loadCollection(currentCollection)">🔄 刷新</button>
            </div>

            <div class="card shadow-sm">
                <div class="table-responsive">
                    <table class="table table-hover align-middle mb-0">
                        <thead class="table-dark">
                            <tr>
                                <th style="width: 200px;">ID</th>
                                <th>Document Snippet</th>
                                <th>Metadata</th>
                                <th style="width: 80px;">操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(item, idx) in vectorData" :key="idx">
                                <td class="small font-monospace">{{ item.id }}</td>
                                <td><div class="code-cell">{{ item.document }}</div></td>
                                <td><div class="code-cell">{{ item.metadata }}</div></td>
                                <td>
                                    <button class="btn btn-outline-danger btn-sm" @click="deleteVector(item.id)">🗑️</button>
                                </td>
                            </tr>
                             <tr v-if="vectorData.length === 0">
                                <td colspan="4" class="text-center p-3 text-muted">该集合为空</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div v-if="!currentTab" class="d-flex h-100 justify-content-center align-items-center text-muted">
            <h4>👈 请从左侧选择表或集合</h4>
        </div>

    </div>

    <div class="modal fade" id="editModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">{{ isEditing ? '编辑记录' : '新增记录' }}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form v-if="currentTable">
                        <div class="mb-3" v-for="col in columns" :key="col.name">
                            <label class="form-label fw-bold">
                                {{ col.name }} <span class="text-muted small">({{ col.type }})</span>
                            </label>

                            <input v-if="col.pk && isEditing" type="text" class="form-control" v-model="editForm[col.name]" disabled>

                            <textarea v-else-if="isJsonCol(col.name) || isJsonContent(editForm[col.name])" 
                                      class="form-control json-editor" 
                                      v-model="editForm[col.name]"
                                      @blur="formatJsonField(col.name)"
                                      placeholder="请输入 JSON"></textarea>

                            <input v-else type="text" class="form-control" v-model="editForm[col.name]">
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                    <button type="button" class="btn btn-primary" @click="saveRecord">保存</button>
                </div>
            </div>
        </div>
    </div>

</div>

<script src="https://jsd.cdn.zzko.cn/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    const { createApp, ref, onMounted } = Vue;

    createApp({
        setup() {
            const hasChroma = ref(false);
            const tables = ref([]);
            const collections = ref([]);

            const currentTab = ref(null); // 'sqlite' or 'vector'
            const currentTable = ref(null);
            const currentCollection = ref(null);

            const columns = ref([]);
            const tableData = ref([]);
            const vectorData = ref([]);

            const searchQuery = ref("");
            const editModal = ref(null);
            const isEditing = ref(false);
            const editForm = ref({});
            const originalPkData = ref({}); // 用于 UPDATE 定位

            const API_BASE = "";

            onMounted(async () => {
                await fetchInitData();
                const modalEl = document.getElementById('editModal');
                editModal.value = new bootstrap.Modal(modalEl);
            });

            const fetchInitData = async () => {
                const res = await fetch(API_BASE + '/api/init');
                const data = await res.json();
                hasChroma.value = data.has_chroma;
                tables.value = data.tables;
                collections.value = data.collections;
            };

            const loadTable = async (tableName) => {
                currentTab.value = 'sqlite';
                currentTable.value = tableName;
                searchQuery.value = "";
                await refreshTable();
            };

            const refreshTable = async () => {
                const url = `${API_BASE}/api/table/${currentTable.value}?q=${encodeURIComponent(searchQuery.value)}`;
                const res = await fetch(url);
                const data = await res.json();
                columns.value = data.columns;
                tableData.value = data.rows;
            };

            const loadCollection = async (collName) => {
                currentTab.value = 'vector';
                currentCollection.value = collName;
                const res = await fetch(`${API_BASE}/api/vector/${collName}`);
                vectorData.value = await res.json();
            };

            const formatCell = (val) => {
                if (typeof val === 'object' && val !== null) return JSON.stringify(val, null, 2);
                return val;
            };

            const isJsonCol = (name) => {
                const n = name.toUpperCase();
                return n.includes('JSON') || n.includes('CONTENT');
            };

            const isJsonContent = (val) => {
                if (typeof val !== 'string') return false;
                val = val.trim();
                return (val.startsWith('{') && val.endsWith('}')) || (val.startsWith('[') && val.endsWith(']'));
            };

            const formatJsonField = (field) => {
                try {
                    const val = editForm.value[field];
                    if(val && typeof val === 'string') {
                        const obj = JSON.parse(val);
                        editForm.value[field] = JSON.stringify(obj, null, 4);
                    }
                } catch(e) { }
            };

            // --- CRUD 操作 ---

            const openEditModal = (row) => {
                editForm.value = {};
                if (row) {
                    isEditing.value = true;
                    // 复制数据
                    editForm.value = JSON.parse(JSON.stringify(row));
                    // 尝试格式化 JSON 字段以便阅读
                    for (const key in editForm.value) {
                         if (isJsonContent(editForm.value[key])) {
                             try {
                                 const obj = JSON.parse(editForm.value[key]);
                                 editForm.value[key] = JSON.stringify(obj, null, 4);
                             } catch(e){}
                         }
                    }

                    // 记录 PK 数据
                    originalPkData.value = {};
                    columns.value.forEach(c => {
                        if (c.pk) originalPkData.value[c.name] = row[c.name];
                    });
                } else {
                    isEditing.value = false;
                    columns.value.forEach(c => editForm.value[c.name] = "");
                }
                editModal.value.show();
            };

            const saveRecord = async () => {
                try {
                    // 压缩 JSON
                    const payloadData = {...editForm.value};
                    for (const key in payloadData) {
                        const val = payloadData[key];
                         if (typeof val === 'string' && isJsonContent(val)) {
                             try {
                                 payloadData[key] = JSON.stringify(JSON.parse(val));
                             } catch(e){}
                         }
                    }

                    let url = `${API_BASE}/api/table/${currentTable.value}`;
                    let method = 'POST';
                    let body = {};

                    if (isEditing.value) {
                        method = 'PUT';
                        body = { pk_data: originalPkData.value, new_data: payloadData };
                    } else {
                        body = { data: payloadData };
                    }

                    const res = await fetch(url, {
                        method: method,
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(body)
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || '保存失败');
                    }

                    editModal.value.hide();
                    refreshTable();
                } catch (e) {
                    alert("错误: " + e.message);
                }
            };

            const deleteRow = async (row) => {
                if(!confirm("确定要删除此记录吗？")) return;

                const pkData = {};
                let hasPk = false;
                columns.value.forEach(c => {
                    if (c.pk) {
                        pkData[c.name] = row[c.name];
                        hasPk = true;
                    }
                });

                if (!hasPk) {
                    alert("该表没有主键，无法删除单行。");
                    return;
                }

                try {
                    const res = await fetch(`${API_BASE}/api/table/${currentTable.value}`, {
                        method: 'DELETE',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ pk_data: pkData })
                    });
                    if (!res.ok) throw new Error((await res.json()).detail);
                    refreshTable();
                } catch(e) {
                    alert("删除失败: " + e.message);
                }
            };

            const deleteVector = async (id) => {
                if(!confirm("确定删除向量 " + id + " ?")) return;
                try {
                    const res = await fetch(`${API_BASE}/api/vector/${currentCollection.value}/${id}`, {
                        method: 'DELETE'
                    });
                    if (!res.ok) throw new Error("删除失败");
                    loadCollection(currentCollection.value);
                } catch(e) {
                    alert(e.message);
                }
            };

            return {
                hasChroma, tables, collections,
                currentTab, currentTable, currentCollection,
                columns, tableData, vectorData,
                searchQuery, isEditing, editForm,
                loadTable, refreshTable, loadCollection,
                formatCell, openEditModal, saveRecord, deleteRow, deleteVector,
                isJsonCol, formatJsonField, isJsonContent
            };
        }
    }).mount('#app');
</script>
</body>
</html>
    """


# --- 路由: API ---

@app.get("/api/init")
async def api_init():
    """获取表列表和集合列表"""
    tables = []
    collections = []

    # 1. SQLite Tables
    if os.path.exists(DB_PATH):
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name")
            tables = [row[0] for row in await cursor.fetchall()]

    # 2. Chroma Collections
    if HAS_CHROMA and os.path.exists(VECTOR_PATH):
        try:
            client = chromadb.PersistentClient(path=VECTOR_PATH, settings=Settings(anonymized_telemetry=False))
            colls = client.list_collections()
            collections = [c.name for c in colls]
        except Exception as e:
            logger.error(f"Chroma init error: {e}")

    return {"has_chroma": HAS_CHROMA, "tables": tables, "collections": collections}


@app.get("/api/table/{table_name}")
async def get_table_data(table_name: str, q: str = ""):
    """获取表数据（带简单搜索）"""
    if not os.path.exists(DB_PATH):
        return {"columns": [], "rows": []}

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 获取列信息
        cursor = await db.execute(f"PRAGMA table_info({table_name})")
        cols_info = await cursor.fetchall()
        columns = [{"name": c["name"], "type": c["type"], "pk": c["pk"] > 0} for c in cols_info]

        # 构建查询
        sql = f"SELECT * FROM {table_name}"
        params = []

        if q:
            clauses = [f"{c['name']} LIKE ?" for c in columns]
            sql += " WHERE " + " OR ".join(clauses)
            params = [f"%{q}%"] * len(columns)

        # 排序 (优先按 PK 倒序)
        pk_cols = [c["name"] for c in columns if c["pk"]]
        if pk_cols:
            sql += f" ORDER BY {pk_cols[-1]} DESC"

        sql += " LIMIT 100"

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

        # 转换 Row 为 dict
        row_list = [dict(row) for row in rows]

        return {"columns": columns, "rows": row_list}


@app.put("/api/table/{table_name}")
async def update_row(table_name: str, req: UpdateRowRequest):
    """更新记录"""
    pk_clause = " AND ".join([f"{k}=?" for k in req.pk_data.keys()])
    set_clause = ", ".join([f"{k}=?" for k in req.new_data.keys()])

    values = list(req.new_data.values()) + list(req.pk_data.values())

    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(f"UPDATE {table_name} SET {set_clause} WHERE {pk_clause}", values)
            await db.commit()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@app.post("/api/table/{table_name}")
async def insert_row(table_name: str, req: InsertRowRequest):
    """插入记录"""
    cols = ", ".join(req.data.keys())
    placeholders = ", ".join(["?" for _ in req.data])
    values = list(req.data.values())

    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})", values)
            await db.commit()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@app.delete("/api/table/{table_name}")
async def delete_row(table_name: str, req: DeleteRowRequest):
    """删除记录"""
    pk_clause = " AND ".join([f"{k}=?" for k in req.pk_data.keys()])
    values = list(req.pk_data.values())

    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(f"DELETE FROM {table_name} WHERE {pk_clause}", values)
            await db.commit()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@app.get("/api/vector/{collection_name}")
async def get_vectors(collection_name: str):
    """获取向量数据预览"""
    if not HAS_CHROMA: return []
    try:
        client = chromadb.PersistentClient(path=VECTOR_PATH, settings=Settings(anonymized_telemetry=False))
        coll = client.get_collection(collection_name)
        res = coll.get(limit=50, include=["documents", "metadatas"])

        data = []
        ids = res['ids']
        for i, doc_id in enumerate(ids):
            data.append({
                "id": doc_id,
                "document": res['documents'][i],
                "metadata": res['metadatas'][i]
            })
        return data
    except Exception as e:
        logger.error(f"Vector fetch error: {e}")
        return []


@app.delete("/api/vector/{collection_name}/{doc_id}")
async def delete_vector_item(collection_name: str, doc_id: str):
    """删除向量"""
    if not HAS_CHROMA: raise HTTPException(status_code=500, detail="Chroma unavailable")
    try:
        client = chromadb.PersistentClient(path=VECTOR_PATH, settings=Settings(anonymized_telemetry=False))
        coll = client.get_collection(collection_name)
        coll.delete(ids=[doc_id])
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    print(f"Starting Web DB Manager at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
