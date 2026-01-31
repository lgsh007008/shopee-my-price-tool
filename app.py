import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.linalg import lstsq
from scipy.optimize import linprog
from supabase import create_client, Client

st.set_page_config(page_title="SKU藏价求解器-多站点版", layout="wide")

# ============ 数据库层 ============

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class MultiSiteSolver:
    def __init__(self):
        self.supabase = init_supabase()
        self.safety_factor = 1.05
        
    def add_order(self, site: str, order_id: str, total_hidden_price: float, items: list):
        """添加订单"""
        try:
            # 检查订单号是否已存在（同站点内）
            existing = self.supabase.table('orders').select("*").eq('site', site).eq('order_id', order_id).execute()
            if existing.data:
                st.error(f"站点 [{site}] 中已存在订单号 [{order_id}]，请使用其他编号")
                return False
            
            # 插入订单
            self.supabase.table('orders').insert({
                "site": site,
                "order_id": order_id,
                "total_hidden_price": total_hidden_price,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            # 插入明细
            for item in items:
                if item['sku']:  # 过滤空值
                    self.supabase.table('order_items').insert({
                        "site": site,
                        "order_id": order_id,
                        "sku": item['sku'].upper().strip(),  # 统一大写
                        "quantity": int(item['qty'])
                    }).execute()
            
            self._analyze_site(site)
            return True
        except Exception as e:
            st.error(f"保存失败: {e}")
            return False
    
    def delete_order(self, site: str, order_id: str):
        """删除特定订单并重新计算"""
        try:
            # 先删明细（外键约束）
            self.supabase.table('order_items').delete().eq('site', site).eq('order_id', order_id).execute()
            # 再删订单
            self.supabase.table('orders').delete().eq('site', site).eq('order_id', order_id).execute()
            # 重算该站点
            self._analyze_site(site)
            return True
        except Exception as e:
            st.error(f"删除失败: {e}")
            return False
    
    def _get_site_data(self, site: str):
        orders = self.supabase.table('orders').select("*").eq('site', site).execute().data
        items = self.supabase.table('order_items').select("*").eq('site', site).execute().data
        return pd.DataFrame(orders), pd.DataFrame(items)
    
    def _analyze_site(self, site: str):
        orders_df, items_df = self._get_site_data(site)
        
        if orders_df.empty or items_df.empty:
            # 清空该站点价格数据（如果没有订单了）
            self.supabase.table('sku_prices').delete().eq('site', site).execute()
            return
        
        orders_df = orders_df.dropna(subset=['total_hidden_price'])
        items_df = items_df.dropna(subset=['sku', 'quantity'])
        
        all_skus = sorted(items_df['sku'].unique())
        if len(all_skus) == 0:
            return
        
        sku_to_col = {sku: i for i, sku in enumerate(all_skus)}
        n_skus = len(all_skus)
        n_orders = len(orders_df)
        
        A = np.zeros((n_orders, n_skus))
        b = orders_df['total_hidden_price'].values
        
        for i, (_, order) in enumerate(orders_df.iterrows()):
            order_items = items_df[items_df['order_id'] == order['order_id']]
            for _, item in order_items.iterrows():
                if item['sku'] in sku_to_col:
                    A[i, sku_to_col[item['sku']]] = item['quantity']
        
        try:
            x, residuals, rank, _ = lstsq(A, b)
            x = np.maximum(x, 0)
            
            has_conflict = residuals > 1e-6 if isinstance(residuals, (int, float)) else len(residuals) > 0 and residuals[0] > 1e-6
            if has_conflict and n_orders > n_skus:
                x = x * self.safety_factor
                status_note = "lstsq_avg"
            else:
                status_note = "exact" if rank >= n_skus else "estimated"
            
            records = []
            for i, sku in enumerate(all_skus):
                sku_appearances = np.count_nonzero(A[:, i])
                is_determined = (rank >= n_skus) or (sku_appearances >= 2 and rank >= n_skus - 1)
                
                unit_price = float(x[i])
                
                if not is_determined:
                    min_v, max_v = self._calc_bounds(A, b, i)
                else:
                    min_v, max_v = unit_price, unit_price
                
                records.append({
                    "site": site,
                    "sku": sku,
                    "unit_price": round(unit_price, 2),
                    "status": "determined" if is_determined else "underdetermined",
                    "calc_method": status_note,
                    "confidence": int(sku_appearances),
                    "min_possible": round(float(min_v), 2) if min_v is not None else 0,
                    "max_possible": round(float(max_v), 2) if max_v is not None else None
                })
            
            self.supabase.table('sku_prices').delete().eq('site', site).execute()
            if records:
                self.supabase.table('sku_prices').insert(records).execute()
                
        except Exception as e:
            st.error(f"计算错误: {e}")
    
    def _calc_bounds(self, A, b, col_idx):
        n = A.shape[1]
        try:
            res_min = linprog(np.eye(n)[col_idx], A_eq=A, b_eq=b, bounds=[(0, None)]*n, method='highs')
            res_max = linprog(-np.eye(n)[col_idx], A_eq=A, b_eq=b, bounds=[(0, None)]*n, method='highs')
            return (res_min.x[col_idx] if res_min.success else 0), (res_max.x[col_idx] if res_max.success else None)
        except:
            return 0, None
    
    def get_sites(self):
        data = self.supabase.table('orders').select("site").execute().data
        if not data:
            return []
        return sorted(list(set([d['site'] for d in data])))
    
    def get_site_status(self, site: str):
        prices = self.supabase.table('sku_prices').select("*").eq('site', site).execute().data
        orders = self.supabase.table('orders').select("*").eq('site', site).order('created_at', desc=True).execute().data
        return pd.DataFrame(prices), pd.DataFrame(orders)
    
    def delete_site_data(self, site: str):
        self.supabase.table('order_items').delete().eq('site', site).execute()
        self.supabase.table('orders').delete().eq('site', site).execute()
        self.supabase.table('sku_prices').delete().eq('site', site).execute()

# ============ 界面初始化 ============

try:
    solver = MultiSiteSolver()
except Exception as e:
    st.error(f"数据库连接失败: {e}")
    st.stop()

# Session State 管理动态输入框
if 'sku_rows' not in st.session_state:
    st.session_state.sku_rows = [{"sku": "", "qty": 1}]

def add_row():
    st.session_state.sku_rows.append({"sku": "", "qty": 1})

def remove_row(index):
    if len(st.session_state.sku_rows) > 1:
        st.session_state.sku_rows.pop(index)

# ============ 主界面 ============

st.title("🌏 多站点 SKU 藏价求解器")
st.caption("支持动态添加多商品 | 站点数据隔离 | 矛盾订单自动平均")

# 侧边栏：站点管理
with st.sidebar:
    st.header("🌐 站点管理")
    existing_sites = solver.get_sites()
    
    if existing_sites:
        current_site = st.selectbox("选择当前站点", existing_sites + ["+ 新建站点"])
    else:
        current_site = "+ 新建站点"
    
    if current_site == "+ 新建站点":
        current_site = st.text_input("输入新站点代码（如：MY、SG、ID）", value="")
    
    if current_site and current_site not in ["+ 新建站点", ""]:
        st.caption(f"当前操作站点：**{current_site}**")
        if st.button(f"🗑️ 清空 [{current_site}] 全部数据", type="secondary"):
            confirm = st.text_input(f"输入 {current_site} 确认删除")
            if confirm == current_site:
                solver.delete_site_data(current_site)
                st.success("已清空")
                st.rerun()

# 主内容区
if not current_site or current_site in ["+ 新建站点", ""]:
    st.info("👈 请先在左侧选择或创建站点")
else:
    st.header(f"站点：{current_site}")
    
    # 录入表单
    with st.container(border=True):
        st.subheader("📝 录入新订单")
        
        # 订单号（独立一行）
        order_id = st.text_input("订单编号 *", 
                                value=f"{current_site}{datetime.now().strftime('%m%d%H%M')}",
                                key="order_id_input")
        
        st.markdown("**商品明细**")
        
        # 动态商品输入行
        items = []
        for i, row in enumerate(st.session_state.sku_rows):
            cols = st.columns([3, 2, 1])
            with cols[0]:
                sku = st.text_input(f"产品编码 {i+1}", 
                                   value=row["sku"], 
                                   key=f"sku_{i}",
                                   placeholder="如：SKU001")
            with cols[1]:
                qty = st.number_input(f"数量 {i+1}", 
                                     min_value=1, 
                                     value=row["qty"], 
                                     key=f"qty_{i}")
            with cols[2]:
                if len(st.session_state.sku_rows) > 1:
                    if st.button("❌", key=f"del_{i}"):
                        remove_row(i)
                        st.rerun()
            
            if sku:  # 只收集非空的
                items.append({"sku": sku, "qty": qty})
        
        # 添加商品按钮（放在商品列表下方）
        if st.button("➕ 添加商品", type="secondary"):
            add_row()
            st.rerun()
        
        # 总藏价（独立一行）
        total_price = st.number_input("订单总藏价 *", 
                                     min_value=0.0, 
                                     value=0.0, 
                                     step=10.0,
                                     key="total_price")
        
        # 提交按钮
        if st.button("🚀 提交计算", type="primary", use_container_width=True):
            if not order_id:
                st.error("请输入订单编号")
            elif len(items) == 0:
                st.error("请至少输入一个产品编码")
            elif total_price <= 0:
                st.error("总藏价必须大于0")
            else:
                with st.spinner("计算中..."):
                    if solver.add_order(current_site, order_id, total_price, items):
                        st.success(f"✅ 订单 {order_id} 已保存")
                        # 清空表单（保留站点）
                        st.session_state.sku_rows = [{"sku": "", "qty": 1}]
                        st.rerun()
    
    # 结果显示
    st.divider()
    prices_df, orders_df = solver.get_site_status(current_site)
    
    if not prices_df.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("✅ 已确定单价")
            det = prices_df[prices_df['status'] == 'determined']
            if not det.empty:
                st.dataframe(
                    det[['sku', 'unit_price', 'calc_method', 'confidence']].rename(columns={
                        'sku': '产品编码',
                        'unit_price': '单件藏价',
                        'calc_method': '计算方式',
                        'confidence': '置信度'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                if 'lstsq_avg' in det['calc_method'].values:
                    st.info("💡 该站点存在矛盾数据，已取平均并放大5%")
            else:
                st.info("录入更多订单来确定价格")
        
        with col2:
            st.subheader("🔍 待确定产品")
            undet = prices_df[prices_df['status'] == 'underdetermined']
            if not undet.empty:
                for _, row in undet.iterrows():
                    range_str = ""
                    if pd.notna(row['max_possible']):
                        range_str = f"范围: {row['min_possible']:.0f}-{row['max_possible']:.0f}"
                    st.metric(f"{row['sku']}", f"¥{row['unit_price']}", range_str)
            else:
                st.success("所有产品价格已确定")
    
    # 历史订单（带删除功能）
    st.divider()
    st.subheader("📋 历史订单")
    
    if not orders_df.empty:
        # 获取订单明细用于展示
        all_items = solver.supabase.table('order_items').select("*").eq('site', current_site).execute().data
        items_df = pd.DataFrame(all_items) if all_items else pd.DataFrame()
        
        for _, order in orders_df.iterrows():
            with st.container(border=True):
                cols = st.columns([3, 3, 2, 1])
                
                with cols[0]:
                    st.markdown(f"**{order['order_id']}**")
                    st.caption(f"{order['created_at'][:10]}")
                
                with cols[1]:
                    # 显示该订单的商品
                    if not items_df.empty:
                        order_items = items_df[items_df['order_id'] == order['order_id']]
                        item_text = ", ".join([f"{r['sku']}×{r['quantity']}" for _, r in order_items.iterrows()])
                        st.text(item_text)
                
                with cols[2]:
                    st.markdown(f"**¥{order['total_hidden_price']}**")
                
                with cols[3]:
                    if st.button("🗑️", key=f"del_order_{order['order_id']}", type="secondary"):
                        if st.checkbox(f"确认删除 {order['order_id']}？", key=f"confirm_{order['order_id']}"):
                            if solver.delete_order(current_site, order['order_id']):
                                st.rerun()
    else:
        st.info("暂无历史订单")

    # 跨站点对比
    st.divider()
    if st.checkbox("🔍 查看所有站点价格对比"):
        all_sites = solver.get_sites()
        comparison = []
        for s in all_sites:
            df, _ = solver.get_site_status(s)
            if not df.empty:
                for _, row in df.iterrows():
                    comparison.append({
                        "站点": s,
                        "产品": row['sku'],
                        "藏价": row['unit_price'],
                        "状态": "✅" if row['status'] == 'determined' else "🔍"
                    })
        if comparison:
            comp_df = pd.DataFrame(comparison)
            pivot = comp_df.pivot(index='产品', columns='站点', values='藏价').fillna('-')
            st.dataframe(pivot, use_container_width=True)
