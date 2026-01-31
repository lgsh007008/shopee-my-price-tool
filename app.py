import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from sympy import symbols, Eq, solve, Matrix
from scipy.linalg import lstsq

st.set_page_config(page_title="SKU藏价求解器-智能版", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 3rem !important;}
    .constraint-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 8px; margin: 5px 0; border-radius: 4px;}
    .exact-box {background-color: #d4edda; border-left: 4px solid #28a745; padding: 8px; margin: 5px 0; border-radius: 4px;}
    .avg-box {background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 8px; margin: 5px 0; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class SmartSolver:
    def __init__(self):
        self.supabase = init_supabase()
        self.safety_factor = 1.05  # 矛盾时放大系数
    
    def add_order(self, site: str, order_id: str, total_hidden_price: float, items: list):
        try:
            existing = self.supabase.table('orders').select("*").eq('site', site).eq('order_id', order_id).execute()
            if existing.data:
                return False, "订单号已存在"
            
            self.supabase.table('orders').insert({
                "site": site, "order_id": order_id,
                "total_hidden_price": total_hidden_price,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            for item in items:
                if item['sku']:
                    self.supabase.table('order_items').insert({
                        "site": site, "order_id": order_id,
                        "sku": item['sku'].upper().strip(),
                        "quantity": int(item['qty'])
                    }).execute()
            
            return True, "保存成功"
        except Exception as e:
            return False, str(e)
    
    def delete_order(self, site: str, order_id: str):
        try:
            self.supabase.table('order_items').delete().eq('site', site).eq('order_id', order_id).execute()
            self.supabase.table('orders').delete().eq('site', site).eq('order_id', order_id).execute()
            return True
        except Exception as e:
            st.error(f"删除失败: {e}")
            return False
    
    def get_site_data(self, site: str):
        orders = self.supabase.table('orders').select("*").eq('site', site).execute().data or []
        items = self.supabase.table('order_items').select("*").eq('site', site).execute().data or []
        return orders, items
    
    def solve_smart(self, site: str):
        """
        智能求解策略：
        1. 先尝试符号精确求解（获得确定值和约束关系）
        2. 如果符号求解失败（矛盾方程组），退回到最小二乘（平均）+ 放大
        3. 同时保留约束关系显示
        """
        orders, items = self.get_site_data(site)
        if not orders:
            return {}, [], [], []  # 确定值, 约束, 统计, 订单
        
        all_skus = sorted(list(set([it['sku'] for it in items])))
        if not all_skus:
            return {}, [], [], orders
        
        # 构建矩阵（用于矛盾检测和最小二乘）
        sku_idx = {s: i for i, s in enumerate(all_skus)}
        n_skus, n_orders = len(all_skus), len(orders)
        
        A = np.zeros((n_orders, n_skus))
        b = np.zeros(n_orders)
        
        for i, order in enumerate(orders):
            b[i] = order['total_hidden_price']
            for it in items:
                if it['order_id'] == order['order_id'] and it['sku'] in sku_idx:
                    A[i, sku_idx[it['sku']]] = it['quantity']
        
        # 尝试符号求解
        symbols_map = {sku: symbols(sku) for sku in all_skus}
        equations = []
        for i, order in enumerate(orders):
            order_items = [it for it in items if it['order_id'] == order['order_id']]
            expr = sum(it['quantity'] * symbols_map[it['sku']] for it in order_items)
            equations.append(Eq(expr, order['total_hidden_price']))
        
        try:
            sym_solution = solve(equations, list(symbols_map.values()), dict=True)
        except:
            sym_solution = []
        
        determined = {}      # SKU -> (值, 计算方式)
        constraints = []     # 约束关系式
        
        # 分析符号解
        has_exact_solution = False
        if sym_solution and len(sym_solution) > 0:
            sol = sym_solution[0]
            all_numeric = True
            
            for sku in all_skus:
                var = symbols_map[sku]
                if var in sol:
                    val = sol[var]
                    if val.is_number:
                        determined[sku] = (float(val), "exact")
                        has_exact_solution = True
                    else:
                        # 是表达式（含其他变量）
                        constraints.append(f"{sku} = {val}")
                        all_numeric = False
                else:
                    # 自由变量，从约束中提取
                    all_numeric = False
            
            # 如果符号求解给出完整数值解，直接返回（无矛盾）
            if has_exact_solution and all_numeric:
                return determined, constraints, [], orders
        
        # 如果符号求解失败或部分欠定，使用最小二乘（处理矛盾）
        # 这对应"同一个SKU在不同订单推出不同价格"的情况 -> 取平均
        if n_orders >= n_skus or not has_exact_solution:
            x, residuals, rank, s = lstsq(A, b)
            x = np.maximum(x, 0)  # 非负
            
            # 检测矛盾：如果残差很大，说明数据矛盾，需要放大
            has_conflict = False
            if isinstance(residuals, (list, np.ndarray)) and len(residuals) > 0:
                has_conflict = residuals[0] > 1e-6
            elif isinstance(residuals, (int, float)):
                has_conflict = residuals > 1e-6
            
            method = "avg_conflict" if has_conflict else "fitted"
            
            # 如果之前有符号解的部分确定值，优先用符号解（更精确）
            # 剩下的用最小二乘填充
            for i, sku in enumerate(all_skus):
                if sku not in determined:  # 未被符号求解确定
                    val = float(x[i])
                    if has_conflict:
                        val = val * self.safety_factor  # 放大
                    determined[sku] = (val, method)
        
        # 提取约束关系（用于显示欠定情况）
        if constraints or not has_exact_solution:
            constraints = self._extract_constraints_rref(A, b, all_skus, determined)
        
        return determined, constraints, [], orders
    
    def _extract_constraints_rref(self, A, b, all_skus, determined):
        """从行最简形提取约束关系"""
        M = Matrix(np.hstack([A, b.reshape(-1, 1)]))
        rref_matrix, pivot_cols = M.rref()
        
        constraints = []
        determined_skus = set(determined.keys())
        
        for row in rref_matrix.tolist():
            coeffs = row[:-1]
            const = row[-1]
            
            if abs(float(const)) < 1e-10 and all(abs(float(c)) < 1e-10 for c in coeffs):
                continue
            
            terms = []
            unknown_part = []
            known_sum = 0
            
            for i, c in enumerate(coeffs):
                c_float = float(c)
                if abs(c_float) > 1e-10:
                    sku = all_skus[i]
                    if sku in determined_skus:
                        known_sum += c_float * determined[sku][0]
                    else:
                        c_str = f"{int(c_float) if c_float == int(c_float) else f'{c_float:.1f}'}"
                        unknown_part.append(f"{c_str}{sku}")
            
            remaining = float(const) - known_sum
            
            if unknown_part and abs(remaining) > 1e-10:
                expr = " + ".join(unknown_part).replace("+ -", "- ")
                constraints.append(f"{expr} = {remaining:.2f}")
        
        return constraints

# ============ 界面 ============
try:
    solver = SmartSolver()
except Exception as e:
    st.error(f"连接失败: {e}")
    st.stop()

if 'sku_rows' not in st.session_state:
    st.session_state.sku_rows = [{"sku": "", "qty": 1}]
if 'delete_confirm' not in st.session_state:
    st.session_state.delete_confirm = {}
if 'current_site' not in st.session_state:
    st.session_state.current_site = ""

def add_row():
    st.session_state.sku_rows.append({"sku": "", "qty": 1})

def remove_row(index):
    if len(st.session_state.sku_rows) > 1:
        st.session_state.sku_rows.pop(index)

st.title("📦 SKU 藏价求解器")

# 站点选择
cols = st.columns([1, 3])
with cols[0]:
    st.markdown("**选择站点**")
with cols[1]:
    existing_sites = list(set([o['site'] for o in solver.supabase.table('orders').select("site").execute().data or []]))
    site_options = existing_sites + ["+ 新建站点"]
    
    index = 0
    if st.session_state.current_site in site_options:
        index = site_options.index(st.session_state.current_site)
    
    selected = st.selectbox("", site_options, index=index)

if selected == "+ 新建站点":
    new_site = st.text_input("输入新站点代码")
    if new_site:
        st.session_state.current_site = new_site.strip().upper()
else:
    st.session_state.current_site = selected

if not st.session_state.current_site or selected == "+ 新建站点":
    st.info("请选择或创建站点")
    st.stop()

site = st.session_state.current_site

# 主体布局
left, right = st.columns([4, 6])

with left:
    with st.container(border=True):
        st.subheader("录入订单")
        
        order_id = st.text_input("订单编号", value=f"{site}{datetime.now().strftime('%m%d%H%M')}")
        
        items = []
        for i, row in enumerate(st.session_state.sku_rows):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                sku = st.text_input(f"产品编码", value=row["sku"], key=f"sku_{i}", placeholder="如：A")
            with c2:
                qty = st.number_input(f"数量", min_value=1, value=row["qty"], key=f"qty_{i}")
            with c3:
                if len(st.session_state.sku_rows) > 1 and st.button("✕", key=f"del_{i}"):
                    remove_row(i)
                    st.rerun()
            
            if sku.strip():
                items.append({"sku": sku.strip().upper(), "qty": qty})
        
        if st.button("➕ 添加商品行", use_container_width=True):
            add_row()
            st.rerun()
        
        total = st.number_input("订单总藏价", min_value=0.0, value=0.0, step=10.0, format="%.2f")
        
        if st.button("🚀 提交并求解", type="primary", use_container_width=True):
            if not order_id: 
                st.error("请输入订单编号")
            elif not items: 
                st.error("请输入产品编码")
            elif total <= 0: 
                st.error("总藏价必须大于0")
            else:
                success, msg = solver.add_order(site, order_id, total, items)
                if success:
                    st.success("已保存")
                    st.session_state.sku_rows = [{"sku": "", "qty": 1}]
                    st.rerun()
                else:
                    st.error(msg)

with right:
    determined, constraints, _, orders = solver.solve_smart(site)
    
    # 统计
    exact_count = sum(1 for v, m in determined.values() if m == "exact")
    avg_count = sum(1 for v, m in determined.values() if m in ["avg_conflict", "fitted"])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("精确确定", exact_count)
    c2.metric("平均估算", avg_count)
    c3.metric("历史订单", len(orders))
    
    st.divider()
    
    # 显示结果
    if determined:
        st.subheader("计算结果")
        
        # 分类显示
        exact_items = {k: v for k, (v, m) in determined.items() if m == "exact"}
        avg_items = {k: (v, m) for k, (v, m) in determined.items() if m != "exact"}
        
        if exact_items:
            st.markdown("<div class='exact-box'><strong>✅ 精确解（方程组一致）</strong></div>", unsafe_allow_html=True)
            df_exact = pd.DataFrame(list(exact_items.items()), columns=['SKU', '藏价'])
            df_exact['藏价'] = df_exact['藏价'].apply(lambda x: f"{x:.2f}")
            st.dataframe(df_exact, use_container_width=True, hide_index=True)
        
        if avg_items:
            st.markdown("<div class='avg-box'><strong>⚠️ 平均估算（数据矛盾，已放大5%）</strong></div>", unsafe_allow_html=True)
            df_avg = pd.DataFrame([(k, f"{v:.2f}", "是" if m == "avg_conflict" else "否") 
                                  for k, (v, m) in avg_items.items()], 
                                 columns=['SKU', '藏价', '是否矛盾'])
            st.dataframe(df_avg, use_container_width=True, hide_index=True)
            
            st.caption("💡 同一个SKU在不同订单中推出了不同价格，已取平均并保守放大")
    
    if constraints:
        st.subheader("🔗 待求解约束（欠定）")
        for cons in constraints:
            st.markdown(f"<div class='constraint-box'>📌 {cons}</div>", unsafe_allow_html=True)
        st.caption("💡 录入只包含这些未知SKU的订单，即可求得确切值")
    
    if not determined and not constraints:
        st.info("录入第一个订单后开始计算")
    
    # 历史订单
    if orders:
        st.divider()
        st.subheader("📋 历史订单")
        
        _, items_data = solver.get_site_data(site)
        
        for order in orders:
            oid = order['order_id']
            order_items = [it for it in items_data if it['order_id'] == oid]
            items_str = ", ".join([f"{it['sku']}×{it['quantity']}" for it in order_items])
            
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
                
                with col1:
                    st.markdown(f"**{oid}**")
                    st.caption(f"{order['created_at'][:10]}")
                with col2:
                    st.text(items_str[:25] + "..." if len(items_str) > 25 else items_str)
                with col3:
                    st.markdown(f"{order['total_hidden_price']:.2f}")
                with col4:
                    confirm_key = f"conf_{oid}"
                    if confirm_key not in st.session_state.delete_confirm:
                        st.session_state.delete_confirm[confirm_key] = False
                    
                    if not st.session_state.delete_confirm[confirm_key]:
                        if st.button("删除", key=f"del_{oid}"):
                            st.session_state.delete_confirm[confirm_key] = True
                            st.rerun()
                    else:
                        if st.button("✓", key=f"yes_{oid}", type="primary"):
                            solver.delete_order(site, oid)
                            st.session_state.delete_confirm[confirm_key] = False
                            st.rerun()
                        if st.button("✕", key=f"no_{oid}"):
                            st.session_state.delete_confirm[confirm_key] = False
                            st.rerun()
