import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.linalg import lstsq
from scipy.optimize import linprog
import json
from supabase import create_client, Client

st.set_page_config(page_title="SKU藏价求解器-多站点版", layout="wide")

# ============ 数据库层（Supabase） ============

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class MultiSiteSolver:
    def __init__(self):
        self.supabase = init_supabase()
        self.safety_factor = 1.05  # 矛盾时放大系数（5%缓冲）
        
    def add_order(self, site: str, order_id: str, total_hidden_price: float, items: list):
        """添加订单（带站点隔离）"""
        try:
            # 插入订单（带站点）
            self.supabase.table('orders').insert({
                "site": site,
                "order_id": order_id,
                "total_hidden_price": total_hidden_price,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            # 插入明细
            for item in items:
                self.supabase.table('order_items').insert({
                    "site": site,
                    "order_id": order_id,
                    "sku": item['sku'],
                    "quantity": item['qty']
                }).execute()
            
            # 只重新计算该站点
            self._analyze_site(site)
            return True
        except Exception as e:
            st.error(f"保存失败: {e}")
            return False
    
    def _get_site_data(self, site: str):
        """获取特定站点的数据"""
        orders = self.supabase.table('orders').select("*").eq('site', site).execute().data
        items = self.supabase.table('order_items').select("*").eq('site', site).execute().data
        return pd.DataFrame(orders), pd.DataFrame(items)
    
    def _analyze_site(self, site: str):
        """分析特定站点（站点间完全隔离）"""
        orders_df, items_df = self._get_site_data(site)
        
        if orders_df.empty or items_df.empty:
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
            # 核心：使用最小二乘（自动处理矛盾数据取平均）
            # 当方程数>未知数时，lstsq给出最小二乘解（即平均值）
            x, residuals, rank, _ = lstsq(A, b)
            x = np.maximum(x, 0)  # 非负
            
            # 如果有矛盾（残差>0），应用放大系数
            has_conflict = residuals > 1e-6 if isinstance(residuals, float) else len(residuals) > 0 and residuals[0] > 1e-6
            
            if has_conflict and n_orders > n_skus:
                # 方程过剩且矛盾，放大结果（保守估计）
                x = x * self.safety_factor
                status_note = "adjusted"  # 标记为调整后
            else:
                status_note = "exact" if rank >= n_skus else "underdetermined"
            
            # 保存结果（带站点标识）
            records = []
            for i, sku in enumerate(all_skus):
                # 判断确定度：如果该SKU出现在所有方程中，或秩满，则为确定
                sku_appearances = np.count_nonzero(A[:, i])
                is_determined = (rank >= n_skus) or (sku_appearances >= 2 and rank >= n_skus - 1)
                
                unit_price = float(x[i])
                
                # 欠定时计算范围，确定时范围就是值本身
                if not is_determined:
                    min_v, max_v = self._calc_bounds(A, b, i)
                else:
                    min_v, max_v = unit_price, unit_price
                
                records.append({
                    "site": site,
                    "sku": sku,
                    "unit_price": round(unit_price, 2),
                    "status": "determined" if is_determined else "underdetermined",
                    "calc_method": "lstsq_avg" if has_conflict else ("exact" if rank >= n_skus else "estimated"),
                    "confidence": int(sku_appearances),
                    "min_possible": round(float(min_v), 2) if min_v else 0,
                    "max_possible": round(float(max_v), 2) if max_v else None
                })
            
            # 先删除该站点旧数据，插入新数据
            self.supabase.table('sku_prices').delete().eq('site', site).execute()
            if records:
                self.supabase.table('sku_prices').insert(records).execute()
                
        except Exception as e:
            st.error(f"计算错误: {e}")
    
    def _calc_bounds(self, A, b, col_idx):
        """计算非负约束下的范围"""
        n = A.shape[1]
        try:
            res_min = linprog(np.eye(n)[col_idx], A_eq=A, b_eq=b, bounds=[(0, None)]*n, method='highs')
            res_max = linprog(-np.eye(n)[col_idx], A_eq=A, b_eq=b, bounds=[(0, None)]*n, method='highs')
            return (res_min.x[col_idx] if res_min.success else 0), (res_max.x[col_idx] if res_max.success else None)
        except:
            return 0, None
    
    def get_sites(self):
        """获取所有站点"""
        data = self.supabase.table('orders').select("site").execute().data
        if not data:
            return []
        return sorted(list(set([d['site'] for d in data])))
    
    def get_site_status(self, site: str):
        """获取特定站点状态"""
        prices = self.supabase.table('sku_prices').select("*").eq('site', site).execute().data
        orders = self.supabase.table('orders').select("*").eq('site', site).order('created_at', desc=True).execute().data
        return pd.DataFrame(prices), pd.DataFrame(orders)
    
    def delete_site_data(self, site: str):
        """清空特定站点"""
        self.supabase.table('order_items').delete().eq('site', site).execute()
        self.supabase.table('orders').delete().eq('site', site).execute()
        self.supabase.table('sku_prices').delete().eq('site', site).execute()

# ============ 界面 ============

st.title("🌏 多站点 SKU 藏价求解器")
st.caption("站点间数据完全隔离 | 矛盾数据自动取平均并放大5%")

try:
    solver = MultiSiteSolver()
except Exception as e:
    st.error(f"数据库连接失败: {e}")
    st.stop()

# 侧边栏：站点选择和管理
with st.sidebar:
    st.header("🌐 站点管理")
    
    # 获取所有站点
    existing_sites = solver.get_sites()
    if existing_sites:
        current_site = st.selectbox("选择当前站点", existing_sites + ["+ 新建站点"])
    else:
        current_site = st.text_input("新建站点名称（如：MY、SG、ID）", value="MY")
    
    if current_site == "+ 新建站点":
        current_site = st.text_input("输入新站点代码", value="")
    
    st.divider()
    
    if current_site and current_site not in ["+ 新建站点", ""]:
        if st.button(f"🗑️ 清空站点 {current_site} 数据", type="secondary"):
            if st.checkbox("确认删除？"):
                solver.delete_site_data(current_site)
                st.rerun()

# 主界面
if not current_site or current_site in ["+ 新建站点", ""]:
    st.info("请先在左侧选择或创建站点")
else:
    st.header(f"当前站点：🏷️ {current_site}")
    
    # 录入区
    with st.form("add_order"):
        cols = st.columns([2, 3, 2])
        with cols[0]:
            order_id = st.text_input("订单号", value=f"{current_site}{datetime.now().strftime('%m%d%H%M')}")
        with cols[1]:
            items_input = st.text_input("商品（格式：SKU:数量）", placeholder="A:1, B:2", 
                                       help="同一站点内相同SKU会自动平均藏价")
        with cols[2]:
            total = st.number_input("总藏价", min_value=0.0, value=100.0)
        
        submitted = st.form_submit_button("🚀 提交计算")
        
        if submitted and items_input:
            try:
                items = []
                for part in items_input.split(','):
                    sku, qty = part.strip().split(':')
                    items.append({"sku": sku.strip().upper(), "qty": int(qty)})  # 转大写避免重复
                
                with st.spinner("计算中..."):
                    if solver.add_order(current_site, order_id, total, items):
                        st.success("✅ 已保存并重新计算该站点价格")
                        st.balloons()
            except Exception as e:
                st.error(f"格式错误: {e}")

    # 结果展示
    st.divider()
    prices_df, orders_df = solver.get_site_status(current_site)
    
    if not prices_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ 已推导单价")
            det = prices_df[prices_df['status'] == 'determined']
            if not det.empty:
                show_cols = ['sku', 'unit_price', 'calc_method']
                st.dataframe(det[show_cols].rename(columns={
                    'sku': 'SKU', 
                    'unit_price': '藏价', 
                    'calc_method': '计算方式'
                }), hide_index=True, use_container_width=True)
                
                # 说明计算方式
                if 'lstsq_avg' in det['calc_method'].values:
                    st.info("💡 该站点存在矛盾订单，已取平均值并放大5%作为保守估计")
            else:
                st.info("录入更多订单来确定价格")
        
        with col2:
            st.subheader("🔍 待确定SKU（需更多数据）")
            undet = prices_df[prices_df['status'] == 'underdetermined']
            for _, row in undet.iterrows():
                range_str = ""
                if pd.notna(row['max_possible']):
                    range_str = f"可能范围: [{row['min_possible']:.0f} - {row['max_possible']:.0f}]"
                st.metric(f"{row['sku']}", f"¥{row['unit_price']}", range_str)
    
    with st.expander("📋 站点历史订单"):
        if not orders_df.empty:
            st.dataframe(orders_df[['order_id', 'total_hidden_price', 'created_at']], 
                        use_container_width=True, hide_index=True)
        else:
            st.info("暂无订单")

    # 跨站点对比（可选）
    st.divider()
    if st.checkbox("🔍 查看所有站点价格对比（同SKU不同站价差）"):
        all_sites = solver.get_sites()
        comparison = []
        for s in all_sites:
            df, _ = solver.get_site_status(s)
            if not df.empty:
                for _, row in df.iterrows():
                    comparison.append({
                        "站点": s,
                        "SKU": row['sku'],
                        "藏价": row['unit_price'],
                        "状态": row['status']
                    })
        if comparison:
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df.pivot(index='SKU', columns='站点', values='藏价'), 
                        use_container_width=True)
