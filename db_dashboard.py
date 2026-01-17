import sqlite3
import pandas as pd
import streamlit as st
import chromadb
from chromadb.config import Settings
import os
import json
import time

# --- 配置 ---
DB_PATH = "data/storage.db"
VECTOR_DB_PATH = "data/vector_store"
PAGE_TITLE = "Aethel Trinity - Memory Manager"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon="🧠")

st.title(f"🧠 {PAGE_TITLE}")

# --- 侧边栏：状态概览 ---
st.sidebar.header("系统状态")
if os.path.exists(DB_PATH):
    st.sidebar.success(f"✅ SQLite 连接正常: {DB_PATH}")
else:
    st.sidebar.error(f"❌ SQLite 未找到: {DB_PATH}")

if os.path.exists(VECTOR_DB_PATH):
    st.sidebar.success(f"✅ ChromaDB 路径存在: {VECTOR_DB_PATH}")
else:
    st.sidebar.error(f"❌ ChromaDB 未找到: {VECTOR_DB_PATH}")


# --- 功能模块 ---

def get_sqlite_conn():
    return sqlite3.connect(DB_PATH)


def view_chat_logs():
    st.header("📜 聊天记录 (Chat Logs)")

    conn = get_sqlite_conn()
    try:
        # 分页查询
        limit = st.slider("显示条数", 10, 500, 50)
        query = f"SELECT * FROM chat_logs ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)

        if not df.empty:
            # 格式化时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce').dt.strftime(
                '%Y-%m-%d %H:%M:%S')

            # 简单过滤
            search = st.text_input("搜索内容 (Content/Role)", "")
            if search:
                df = df[df['content'].str.contains(search, case=False, na=False) | df['role'].str.contains(search,
                                                                                                           case=False,
                                                                                                           na=False)]

            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无聊天记录。")
    except Exception as e:
        st.error(f"读取 Chat Logs 失败: {e}")
    finally:
        conn.close()


def manage_core_memory():
    st.header("💎 核心记忆 (Core Memory - KV)")

    conn = get_sqlite_conn()

    # --- 读取 ---
    try:
        df = pd.read_sql_query("SELECT * FROM core_memory", conn)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("现有记忆列表")
            if not df.empty:
                df['last_updated'] = pd.to_datetime(df['last_updated'], unit='s', errors='coerce').dt.strftime(
                    '%Y-%m-%d %H:%M:%S')
                st.dataframe(df, use_container_width=True)
            else:
                st.info("暂无核心记忆。")

        # --- 写入/修改 ---
        with col2:
            st.subheader("✏️ 新增/修改记忆")
            with st.form("core_mem_form"):
                user_id = st.text_input("User ID", "admin_console")
                key = st.text_input("Key (e.g., basic:name)")
                content = st.text_area("Content")
                submitted = st.form_submit_button("保存")

                if submitted and user_id and key and content:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO core_memory (user_id, key, content, last_updated, source)
                            VALUES (?, ?, ?, ?, ?)
                        """, (user_id, key, content, time.time(), "dashboard_gui"))
                        conn.commit()
                        st.success(f"已保存: {key}")
                        st.rerun()  # 刷新页面
                    except Exception as e:
                        st.error(f"写入失败: {e}")

            # --- 删除 ---
            st.subheader("🗑️ 删除记忆")
            with st.form("del_mem_form"):
                del_user_id = st.text_input("User ID (删除)", "admin_console")
                del_key = st.text_input("Key (删除)")
                del_submit = st.form_submit_button("删除")

                if del_submit and del_user_id and del_key:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM core_memory WHERE user_id=? AND key=?", (del_user_id, del_key))
                        conn.commit()
                        st.warning(f"已删除: {del_key}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")

    except Exception as e:
        st.error(f"数据库错误: {e}")
    finally:
        conn.close()


def view_vector_store():
    st.header("🕸️ 向量记忆 (ChromaDB)")

    try:
        # 连接 ChromaDB (持久化模式)
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH, settings=Settings(anonymized_telemetry=False))

        # 获取所有集合
        collections = client.list_collections()
        coll_names = [c.name for c in collections]

        if not coll_names:
            st.warning("未发现任何向量集合。")
            return

        selected_coll_name = st.selectbox("选择集合 (Collection)", coll_names, index=0)
        collection = client.get_collection(selected_coll_name)

        # 获取集合统计
        count = collection.count()
        st.metric("条目数量", count)

        # 查看数据
        limit = st.slider("加载条数 (由于向量较大，建议限制)", 10, 200, 20)

        # 获取数据 (不包含 embeddings 以加速显示)
        results = collection.get(limit=limit, include=["documents", "metadatas"])

        if results["ids"]:
            data = []
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                doc = results["documents"][i] if results["documents"] else ""

                # 处理时间戳
                created_at = meta.get("created_at")
                if created_at:
                    try:
                        created_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(created_at)))
                    except:
                        pass

                data.append({
                    "ID": doc_id,
                    "Content": doc,
                    "Type": meta.get("type", "unknown"),
                    "Status": meta.get("status", "unknown"),
                    "Created At": created_at,
                    "Metadata": json.dumps(meta, ensure_ascii=False)
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # 删除功能
            st.divider()
            with st.expander("危险操作：删除向量条目"):
                del_id = st.text_input("输入要删除的向量 ID")
                if st.button("确认删除"):
                    if del_id:
                        try:
                            collection.delete(ids=[del_id])
                            st.success(f"已删除 ID: {del_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {e}")
        else:
            st.info("集合为空。")

    except Exception as e:
        st.error(f"连接 ChromaDB 失败 (可能被主程序占用): {e}")


# --- 主导航 ---
tab1, tab2, tab3 = st.tabs(["💎 核心记忆 (SQLite)", "📜 聊天记录 (SQLite)", "🕸️ 向量记忆 (ChromaDB)"])

with tab1:
    manage_core_memory()
with tab2:
    view_chat_logs()
with tab3:
    view_vector_store()
