import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sympy import symbols, Eq, solve, Number
import json
from datetime import datetime

st.set_page_config(page_title="多站点 SKU 藏价系统 (Pro版)", layout="wide")

# --- 1. 连接 Google Sheets ---
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    try:
        # 读取数据，如果表头对不上或为空，返回空结构
        df = conn.read(ttl="0s")
        expected_cols = ['order_id', 'site', 'items', 'total_price', 'created_at']
        # 简单的容错：如果缺少列，就补充空列
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype='object')
        return df
    except Exception:
        return pd.DataFrame(columns=['order_id', 'site', 'items', 'total_price', 'created_at'])

# --- 2. 核心解算逻辑 (逻辑不变) ---
def solve_prices(df):
    if df.empty:
        return {}, []

    all_skus = set()
    equations_data = []
    
    for _, row in df.iterrows():
        try:
            # 清洗数据，确保是有效的 JSON
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

# --- 3. 界面布局 ---
st.title("🌍 多站点 SKU 藏价推导系统 (Pro)")

# 获取数据
data = get_data()

# --- 侧边栏：录入数据 (全新升级) ---
with st.sidebar:
    st.header("📝 录入新订单")
    
    # 1. 基础信息
    site_input = st.text_input("站点名称", placeholder="例如 US, UK (必填)")
    order_id_input = st.text_input("订单编号", placeholder="例如 20240101-01 (必填)")
    
    # 2. 动态产品录入 (Data Editor)
    st.markdown("👇 **在下方表格录入产品详情：**")
    
    # 初始化一个空的 DataFrame 模板供用户填写
    default_df = pd.DataFrame(
        [{"产品编码": "", "数量": 1}], # 默认给一行
    )
    
    # 显示可编辑表格 (num_rows="dynamic" 允许增删行)
    edited_df = st.data_editor(
        default_df,
        column_config={
            "产品编码": st.column_config.TextColumn("产品编码 (SKU)", required=True),
            "数量": st.column_config.NumberColumn("数量", min_value=1, step=1, required=True)
        },
        num_rows="dynamic", # 关键：允许用户新增、删除行
        hide_index=True,
        use_container_width=True,
        key="editor"
    )

    # 3. 总价输入
    total_price = st.number_input("该订单总藏价", min_value=0.0, step=0.1)
    
    # 4. 提交按钮
    submit_btn = st.button("💾 保存订单", type="primary")

    if submit_btn:
        # --- 校验逻辑 ---
        if not site_input or not order_id_input:
            st.error("❌ 请填写【站点名称】和【订单编号】")
        elif edited_df.empty:
             st.error("❌ 请至少输入一个产品")
        else:
            try:
                # --- 数据转换 ---
                # 将表格数据转为 JSON 格式: {"A": 1, "B": 2}
                items_dict = {}
                valid_items = False
                
                for index, row in edited_df.iterrows():
                    sku = str(row["产品编码"]).strip()
                    qty = int(row["数量"])
                    if sku: # 只有 SKU 不为空才记录
                        items_dict[sku] = items_dict.get(sku, 0) + qty
                        valid_items = True
                
                if not valid_items:
                    st.error("❌ 产品编码不能为空")
                    st.stop()

                # --- 写入数据库 ---
                new_row = pd.DataFrame([{
                    "order_id": order_id_input.strip(),
                    "site": site_input.upper().strip(),
                    "items": json.dumps(items_dict),
                    "total_price": total_price,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])
                
                updated_df = pd.concat([data, new_row], ignore_index=True)
                conn.update(data=updated_df)
                
                st.success(f"✅ 订单 {order_id_input} 保存成功！")
                st.rerun() # 刷新页面
                
            except Exception as e:
                st.error(f"保存失败: {e}")

# --- 主界面：查看结果 ---

if 'site' in data.columns and not data.empty and len(data) > 0:
    # 获取所有站点
    unique_sites = data['site'].dropna().unique()
    if len(unique_sites) > 0:
        selected_site = st.selectbox("📊 请选择要分析的站点：", unique_sites)
        
        # 过滤数据
        site_data = data[data['site'] == selected_site]
        
        if not site_data.empty:
            solved, pending = solve_prices(site_data)

            st.markdown(f"### 📍 站点：{selected_site}")
            
            # 展示计算结果
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ 已推导出的 SKU 藏价")
                if solved:
                    # 格式化显示
                    res_df = pd.DataFrame(list(solved.items()), columns=['SKU', '单个藏价'])
                    st.dataframe(res_df.style.format({"单个藏价": "{:.2f}"}), use_container_width=True)
                else:
                    st.info("数据量不足，暂无确切解。")

            with col2:
                st.subheader("🔗 待定关系 / 需要更多数据")
                if pending:
                    for p in pending:
                        st.warning(f"📐 {p}")
                else:
                    st.success("无待定关系，所有涉及的 SKU 均已解出（或未录入）。")

            st.divider()
            
            # 展示历史记录 (只看需要的列)
            st.subheader(f"📂 {selected_site} 站点的历史订单")
            display_cols = ['order_id', 'items', 'total_price', 'created_at']
            # 确保列存在防止报错
            existing_display_cols = [c for c in display_cols if c in site_data.columns]
            st.dataframe(site_data[existing_display_cols].sort_values(by='created_at', ascending=False), use_container_width=True)
        else:
            st.info(f"站点 {selected_site} 暂无数据。")
    else:
         st.info("暂无站点数据。")
else:
    st.info("👋 欢迎！请在左侧录入第一笔订单。")
