import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from sympy import symbols, Eq, solve, Number
import json

st.set_page_config(page_title="多站点 SKU 藏价系统", layout="wide")

# --- 1. 连接 Google Sheets ---
# 请确保 .streamlit/secrets.toml 已配置好
conn = st.connection("gsheets", type=GSheetsConnection)

def get_data():
    # ttl=0 确保不缓存，每次读取最新
    return conn.read(ttl="0s")

# --- 2. 核心解算逻辑 (通用版) ---
def solve_prices(df):
    if df.empty:
        return {}, []

    all_skus = set()
    equations_data = []
    
    # 遍历筛选后的数据
    for _, row in df.iterrows():
        try:
            # 兼容处理：如果数据有些是单引号有些是双引号
            item_str = row['items'].replace("'", '"')
            items = json.loads(item_str)
            
            equations_data.append({'items': items, 'total_price': row['total_price']})
            all_skus.update(items.keys())
        except Exception:
            continue

    if not all_skus:
        return {}, []

    # 建立数学符号
    var_map = {name: symbols(name) for name in all_skus}
    equations = []
    
    # 构建方程组
    for order in equations_data:
        # 方程：数量 * 单价 + ... = 总价
        expr = sum(count * var_map[sku] for sku, count in order['items'].items())
        equations.append(Eq(expr, order['total_price']))

    # 调用 SymPy 求解
    solution = solve(equations, dict=True)
    
    solved_dict = {}
    pending_relations = []

    if solution:
        # 通常取第一个解集
        res = solution[0]
        for sku in all_skus:
            val = res.get(var_map[sku])
            if val is not None:
                # 区分是纯数字结果，还是依赖其他变量的公式
                if isinstance(val, (int, float, Number)):
                    solved_dict[sku] = float(val)
                else:
                    pending_relations.append(f"{sku} = {val}")
    
    return solved_dict, pending_relations

# --- 3. 界面布局 ---
st.title("🌍 多站点 SKU 藏价推导系统")

# 获取全部数据
data = get_data()

# --- 侧边栏：录入数据 ---
with st.sidebar:
    st.header("📝 录入新订单")
    with st.form("add_form", clear_on_submit=True):
        # 自动获取已有的站点列表供选择，也可以输入新的
        existing_sites = data['site'].unique().tolist() if 'site' in data.columns else []
        site_input = st.text_input("站点名称 (如 US, UK)", placeholder="可以直接输入新站点")
        
        items_input = st.text_input("产品及数量 (格式: A:1, B:2)", placeholder="A:1, B:1")
        total_price = st.number_input("该订单总藏价", min_value=0.0)
        
        submitted = st.form_submit_button("保存到云端")

        if submitted and items_input and site_input:
            try:
                # 解析输入 A:1, B:2 -> {'A': 1, 'B': 2}
                new_items = {k.strip(): int(v.strip()) for k, v in [item.split(':') for item in items_input.split(',')]}
                
                # 准备新行数据
                new_row = pd.DataFrame([{
                    "id": len(data) + 1,
                    "site": site_input.upper().strip(), # 统一大写
                    "items": json.dumps(new_items),
                    "total_price": total_price
                }])
                
                # 更新 Google Sheets
                updated_df = pd.concat([data, new_row], ignore_index=True)
                conn.update(data=updated_df)
                st.success(f"[{site_input}] 订单已保存！")
                st.rerun()
            except Exception as e:
                st.error(f"格式错误: {e}")

# --- 主界面：查看结果 ---

# 1. 站点选择器
if 'site' in data.columns and not data.empty:
    unique_sites = data['site'].unique()
    selected_site = st.selectbox("请选择要查看的站点：", unique_sites)
    
    # 2. 过滤数据并计算
    site_data = data[data['site'] == selected_site]
    solved, pending = solve_prices(site_data)

    st.markdown(f"### 📍 当前站点：{selected_site}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ 已确定单价")
        if solved:
            st.table(pd.DataFrame(list(solved.items()), columns=['SKU', '单个藏价']))
        else:
            st.info("数据不足，尚无确切解")

    with col2:
        st.subheader("⏳ 待定关系/公式")
        if pending:
            for p in pending:
                st.warning(p)
        else:
            st.write("无待定关系")

    st.divider()
    with st.expander(f"查看 {selected_site} 站点的原始订单记录"):
        st.dataframe(site_data, use_container_width=True)

else:
    st.info("暂无数据，请在左侧录入第一笔订单。")