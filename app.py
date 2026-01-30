import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine, text

st.set_page_config(page_title="跨境 SKU 藏价系统 (智能均价版)", layout="wide")

# --- 1. 连接 Supabase 数据库 ---
# 确保 .streamlit/secrets.toml 中 [connections.db] 配置正确
conn = st.connection("db", type="sql")

# 初始化数据库表
def init_db():
    try:
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
    except Exception as e:
        st.error(f"数据库初始化失败: {e}")

# 页面加载时尝试初始化
init_db()

# --- 2. 数据库读写操作 ---
def get_data():
    try:
        # 读取数据
        return conn.query("SELECT * FROM orders ORDER BY created_at DESC;", ttl=0)
    except Exception:
        return pd.DataFrame(columns=['order_id', 'site', 'items', 'total_price', 'created_at'])

def save_order(order_id, site, items_dict, total_price):
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

# --- 3. 核心算法：最小二乘法 (计算加权平均 + 安全边际) ---
def solve_prices_smart(df, safety_margin=0.0):
    """
    safety_margin: 安全系数，0.1 代表上浮 10%
    """
    if df.empty:
        return {}

    # 1. 提取所有 SKU 和构建矩阵数据
    all_skus = set()
    rows = []     # 矩阵 A (系数)
    results = []  # 向量 b (结果)
    
    # 第一遍遍历：收集所有出现的 SKU
    parsed_orders = []
    for _, row in df.iterrows():
        try:
            item_str = str(row['items']).replace("'", '"')
            items = json.loads(item_str)
            parsed_orders.append({'items': items, 'price': float(row['total_price'])})
            all_skus.update(items.keys())
        except:
            continue
    
    if not all_skus:
        return {}

    sorted_skus = sorted(list(all_skus)) # 保证顺序固定
    
    # 第二遍遍历：构建矩阵
    for order in parsed_orders:
        # 构建这一行的系数，例如 [1, 2, 0] 代表 1个A, 2个B, 0个C
        sku_counts = [order['items'].get(sku, 0) for sku in sorted_skus]
        rows.append(sku_counts)
        results.append(order['price'])

    # 2. 转换为 NumPy 数组
    A = np.array(rows)
    b = np.array(results)

    # 3. 使用最小二乘法求解 (Least Squares)
    # rcond=None 让它处理“秩亏”情况（即方程不够解出所有变量时，给出最小范数解）
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except Exception as e:
        return {}

    # 4. 组装结果并应用安全系数
    solved_dict = {}
    for i, sku in enumerate(sorted_skus):
        # 原始计算价格
        raw_price = x[i]
        
        # 逻辑修正：价格不应该为负数 (数学上有可能算出负数，这里强制归零或取绝对值)
        if raw_price < 0:
            raw_price = 0.0
            
        # 应用安全边际 (比如 raw_price * 1.05)
        final_price = raw_price * (1 + safety_margin)
        solved_dict[sku] = final_price
        
    return solved_dict

# --- 4. 界面布局 ---
st.title("🛡️ 跨境 SKU 藏价系统 (Supabase 均价版)")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 设置与录入")
    
    # 新增：安全系数滑块
    st.info("💡 **计算逻辑说明**：\n系统会自动计算历史订单的**加权平均值**。如果同一个 SKU 在不同订单推导出的价格不同，系统会取中间值。")
    buffer_percent = st.slider("💰 藏价安全上浮比例 (Buffer)", 0, 30, 5, format="%d%%")
    safety_margin = buffer_percent / 100.0
    
    st.divider()
    
    st.subheader("📝 新增订单")
    site_options = ["泰国", "菲律宾", "墨西哥", "美国", "英国"]
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
                st.rerun()
            except Exception as e:
                st.error(f"保存失败: {e}")

# --- 主界面 ---
data = get_data()

if not data.empty:
    existing_sites = data['site'].unique().tolist()
    all_site_options = sorted(list(set(site_options + existing_sites)))
    
    st.divider()
    
    # 顶部选择栏
    c1, c2 = st.columns([1, 3])
    with c1:
        selected_view_site = st.selectbox("📊 查看哪个站点的藏价？", all_site_options)
    
    site_data = data[data['site'] == selected_view_site]
    
    if not site_data.empty:
        # 调用新的最小二乘法函数
        solved_prices = solve_prices_smart(site_data, safety_margin)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"✅ 建议藏价 (含 {buffer_percent}% 浮动)")
            if solved_prices:
                # 转换成 DataFrame 展示
                df_res = pd.DataFrame(list(solved_prices.items()), columns=['SKU', '建议设定藏价'])
                # 排序
                df_res = df_res.sort_values(by='SKU')
                st.dataframe(
                    df_res.style.format({"建议设定藏价": "{:.2f}"}).background_gradient(cmap="Blues", subset=["建议设定藏价"]), 
                    use_container_width=True,
                    height=500
                )
            else:
                st.warning("⚠️ 数据不足，无法计算")

        with col2:
            st.subheader("📂 历史订单数据源")
            st.dataframe(
                site_data[['created_at', 'order_id', 'items', 'total_price']], 
                use_container_width=True,
                height=500
            )
    else:
        st.info(f"{selected_view_site} 暂无数据")
else:
    st.info("👋 数据库为空，请在左侧开始录入第一笔订单。")
