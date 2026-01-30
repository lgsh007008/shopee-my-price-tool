import streamlit as st
import pandas as pd
from sympy import symbols, Eq, solve, Number
import json
from datetime import datetime
from sqlalchemy import create_engine, text

st.set_page_config(page_title="跨境多站点 SKU 藏价系统 (DB版)", layout="wide")

# --- 1. 连接数据库 (Supabase) ---
# 使用 Streamlit 提供的 SQL 连接器，它会自动读取 secrets 中的 [connections.db]
conn = st.connection("db", type="sql")

# 初始化：如果表不存在，自动创建
def init_db():
    with conn.session as s:
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                order_id TEXT NOT NULL,
                site TEXT NOT NULL,
                items TEXT NOT NULL,
                total_price FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        s.commit()

# 页面加载时尝试初始化表结构
try:
    init_db()
except Exception as e:
    st.error(f"数据库连接失败，请检查 Secrets 配置。错误: {e}")

# --- 2. 数据读取与写入 ---
def get_data():
    try:
        # 读取所有数据
        df = conn.query("SELECT * FROM orders ORDER BY created_at DESC;", ttl=0)
        return df
    except Exception:
        return pd.DataFrame(columns=['order_id', 'site', 'items', 'total_price', 'created_at'])

def save_order(order_id, site, items_dict, total_price):
    # 构建 SQL 插入语句
    items_json = json.dumps(items_dict)
    with conn.session as s:
        s.execute(
            text("INSERT INTO orders (order_id, site, items, total_price, created_at) VALUES (:oid, :site, :items, :price, :time)"),
            params={
                "oid": order_id, 
                "site": site, 
                "items": items_json, 
                "price": total_price,
                "time": datetime.now()
            }
        )
        s.commit()

# --- 3. 核心解算逻辑 (保持不变) ---
def solve_prices(df):
    if df.empty:
        return {}, []

    all_skus = set()
    equations_data = []
    
    for _, row in df.iterrows():
        try:
            item_str = str(row['items']).replace("'", '"')
            items = json.loads(item_str)
            equations_data.append({'items': items, 'total_price': float(row['total_price'])})
            all_skus.update(items.keys())
        except Exception:
            continue

    if not all_skus:
        return {}, []

    var_map = {name: symbols(name) for name in all_skus}
    equations = []
    
    for order in equations_data:
        expr = sum(count * var_map[sku] for sku, count in order['items'].items())
        equations.append(Eq(expr, order['total_price']))

    solution = solve(equations, dict=True)
    solved_dict = {}
    pending_relations = []

    if solution:
        res = solution[0]
        for sku in all_skus:
            val = res.get(var_map[sku])
            if val is not None:
                if isinstance(val, (int, float, Number)):
                    solved_dict[sku] = float(val)
                else:
                    pending_relations.append(f"{sku} = {val}")
    
    return solved_dict, pending_relations

# --- 4. 界面布局 ---
st.title("🌏 跨境电商 SKU 藏价系统 (Supabase版)")

# --- 侧边栏 ---
with st.sidebar:
    st.header("📝 新增订单")
    
    site_options = ["泰国", "菲律宾", "墨西哥"]
    site_input = st.selectbox("选择站点", site_options)
    order_id_input = st.text_input("订单编号", placeholder="例如 TH240101")
    
    st.markdown("👇 **录入产品明细：**")
    default_df = pd.DataFrame([{"产品编码": "", "数量": 1}])
    
    edited_df = st.data_editor(
        default_df,
        column_config={
            "产品编码": st.column_config.TextColumn("SKU", required=True),
            "数量": st.column_config.NumberColumn("数量", min_value=1, required=True)
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="editor"
    )

    total_price = st.number_input("订单总藏价", min_value=0.0, step=0.01, format="%.2f")
    submit_btn = st.button("💾 保存数据", type="primary")

    if submit_btn:
        if not order_id_input:
            st.error("❌ 缺少订单编号")
        elif edited_df.empty:
             st.error("❌ 请至少录入一个产品")
        else:
            try:
                items_dict = {}
                valid = False
                for _, row in edited_df.iterrows():
                    sku = str(row["产品编码"]).strip()
                    qty = int(row["数量"])
                    if sku:
                        items_dict[sku] = items_dict.get(sku, 0) + qty
                        valid = True
                
                if not valid:
                    st.error("❌ SKU不能为空")
                    st.stop()

                save_order(order_id_input.strip(), site_input, items_dict, total_price)
                st.success(f"✅ 保存成功！")
                st.rerun() # 刷新页面获取最新数据
            except Exception as e:
                st.error(f"保存失败: {e}")

# --- 主界面 ---
data = get_data()

if not data.empty:
    existing_sites = data['site'].unique().tolist()
    all_site_options = sorted(list(set(site_options + existing_sites)))
    
    st.divider()
    selected_view_site = st.selectbox("📊 选择站点查看数据：", all_site_options)
    
    site_data = data[data['site'] == selected_view_site]
    
    if not site_data.empty:
        solved, pending = solve_prices(site_data)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ 已计算藏价")
            if solved:
                df_res = pd.DataFrame(list(solved.items()), columns=['SKU', '单价'])
                st.dataframe(df_res.style.format({"单价": "{:.2f}"}), use_container_width=True)
            else:
                st.warning("⚠️ 数据不足或有冲突，无法计算")

        with col2:
            st.subheader("🔗 待定关系")
            if pending:
                for p in pending:
                    st.info(f"📐 {p}")
            else:
                st.write("无")

        st.subheader("📂 历史订单")
        st.dataframe(site_data[['order_id', 'items', 'total_price', 'created_at']], use_container_width=True)
    else:
        st.info(f"{selected_view_site} 暂无数据")
else:
    st.info("👋 数据库为空，请开始录入。")
