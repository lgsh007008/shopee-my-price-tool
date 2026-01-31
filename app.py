import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from sympy import symbols, Eq, solve, Matrix

st.set_page_config(page_title="SKU藏价求解器-符号代数版", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 3rem !important;}
    .constraint-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0;}
    .solved-box {background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class SymbolicSolver:
    def __init__(self):
        self.supabase = init_supabase()
    
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
        orders = self.supabase.table('orders').select("*").eq('site', site).execute().data
        items = self.supabase.table('order_items').select("*").eq('site', site).execute().data
        return orders, items
    
    def solve_symbolic(self, site: str):
        """核心：使用SymPy符号求解"""
        orders, items = self.get_site_data(site)
        
        if not orders:
            return {}, [], []  # 确定解, 约束关系, 原始订单
        
        # 收集所有SKU
        all_skus = sorted(list(set([item['sku'] for item in items])))
        if not all_skus:
            return {}, [], orders
        
        # 创建符号变量
        symbols_map = {sku: symbols(sku) for sku in all_skus}
        
        # 构建方程组
        equations = []
        for order in orders:
            order_items = [it for it in items if it['order_id'] == order['order_id']]
            expr = sum(it['quantity'] * symbols_map[it['sku']] for it in order_items)
            equations.append(Eq(expr, order['total_hidden_price']))
        
        # 求解
        solution = solve(equations, list(symbols_map.values()), dict=True)
        
        determined = {}      # 已确定的具体数值
        constraints = []     # 欠定约束关系（如 2D + E = 100）
        free_vars = []       # 自由变量列表
        
        if solution:
            sol = solution[0]  # 取第一个解（如果有多个，它们等价）
            
            # 分析每个变量
            for sku in all_skus:
                var = symbols_map[sku]
                if var in sol:
                    val = sol[var]
                    # 检查是具体数字还是表达式
                    if val.is_number:
                        determined[sku] = float(val)
                    else:
                        # 是表达式（包含其他变量），视为欠定
                        constraints.append(f"{sku} = {val}")
                        if sku not in free_vars:
                            free_vars.append(sku)
                else:
                    # 变量不在解中，说明是自由变量
                    free_vars.append(sku)
        
        # 如果没有得到显式解（可能系统欠定且无显式表达式），使用矩阵方法提取约束
        if not determined and not constraints and free_vars:
            constraints = self._extract_matrix_constraints(orders, items, all_skus)
        
        return determined, constraints, orders
    
    def _extract_matrix_constraints(self, orders, items, all_skus):
        """从矩阵提取约束关系（当sympy返回空时备用）"""
        sku_idx = {s: i for i, s in enumerate(all_skus)}
        n_skus = len(all_skus)
        n_orders = len(orders)
        
        # 构建矩阵
        A = np.zeros((n_orders, n_skus))
        b = np.zeros(n_orders)
        
        for i, order in enumerate(orders):
            b[i] = order['total_hidden_price']
            order_items = [it for it in items if it['order_id'] == order['order_id']]
            for it in order_items:
                A[i, sku_idx[it['sku']]] = it['quantity']
        
        # 计算行最简形
        M = Matrix(np.hstack([A, b.reshape(-1, 1)]))
        rref_matrix, pivot_cols = M.rref()
        
        constraints = []
        # 从rref提取方程
        for row in rref_matrix.tolist():
            coeffs = row[:-1]
            const = row[-1]
            
            # 只保留非零行
            if any(abs(c) > 1e-10 for c in coeffs):
                terms = []
                for i, c in enumerate(coeffs):
                    if abs(c) > 1e-10:
                        c_str = f"{int(c) if c == int(c) else f'{c:.2f}'}"
                        terms.append(f"{c_str}{all_skus[i]}")
                
                if terms:
                    expr = " + ".join(terms).replace("+ -", "- ")
                    constraints.append(f"{expr} = {float(const):.2f}")
        
        return constraints

# ============ 界面 ============
try:
    solver = SymbolicSolver()
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
existing_sites = list(set([o['site'] for o in solver.supabase.table('orders').select("site").execute().data or []]))
site_options = existing_sites + ["+ 新建站点"]

cols = st.columns([1, 3])
with cols[0]:
    st.markdown("**选择站点**")
with cols[1]:
    selected = st.selectbox("", site_options, 
                           index=site_options.index(st.session_state.current_site) if st.session_state.current_site in site_options else 0)

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
            if not order_id: st.error("请输入订单编号")
            elif not items: st.error("请输入产品编码")
            elif total <= 0: st.error("总藏价必须大于0")
            else:
                success, msg = solver.add_order(site, order_id, total, items)
                if success:
                    st.success("已保存")
                    st.session_state.sku_rows = [{"sku": "", "qty": 1}]
                    st.rerun()
                else:
                    st.error(msg)

with right:
    determined, constraints, orders = solver.solve_symbolic(site)
    
    # 统计
    c1, c2, c3 = st.columns(3)
    c1.metric("已确定SKU", len(determined))
    c2.metric("约束关系", len(constraints))
    c3.metric("历史订单", len(orders))
    
    st.divider()
    
    if determined:
        st.subheader("✅ 已确定藏价")
        df_det = pd.DataFrame(list(determined.items()), columns=['SKU', '藏价'])
        df_det['藏价'] = df_det['藏价'].apply(lambda x: f"{x:.2f}")
        st.dataframe(df_det, use_container_width=True, hide_index=True)
    
    if constraints:
        st.subheader("🔗 待求解约束（需更多订单）")
        for cons in constraints:
            st.markdown(f"<div class='constraint-box'>📌 {cons}</div>", unsafe_allow_html=True)
        st.caption("💡 录入只包含这些未知SKU的订单，即可解除约束求得确切值")
    
    if not determined and not constraints:
        st.info("录入第一个订单后开始计算")
    
    # 历史订单（带删除）
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
