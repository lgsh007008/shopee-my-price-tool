import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.linalg import lstsq
from scipy.optimize import linprog
from supabase import create_client, Client

st.set_page_config(page_title="SKU藏价求解器", layout="wide")

# 自定义 CSS - 增加顶部padding避开Streamlit Cloud状态条
st.markdown("""
<style>
    .block-container {padding-top: 3rem !important; padding-bottom: 2rem;}
    .stButton>button {border-radius: 6px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .site-selector {background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# ============ 数据库层（同上，省略重复代码）===========
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
        try:
            existing = self.supabase.table('orders').select("*").eq('site', site).eq('order_id', order_id).execute()
            if existing.data:
                return False, f"站点 [{site}] 中已存在订单号 [{order_id}]"
            
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
            
            self._analyze_site(site)
            return True, "保存成功"
        except Exception as e:
            return False, str(e)
    
    def delete_order(self, site: str, order_id: str):
        try:
            self.supabase.table('order_items').delete().eq('site', site).eq('order_id', order_id).execute()
            self.supabase.table('orders').delete().eq('site', site).eq('order_id', order_id).execute()
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
            self.supabase.table('sku_prices').delete().eq('site', site).execute()
            return
        
        orders_df = orders_df.dropna(subset=['total_hidden_price'])
        items_df = items_df.dropna(subset=['sku', 'quantity'])
        all_skus = sorted(items_df['sku'].unique())
        
        if len(all_skus) == 0:
            return
        
        sku_to_col = {sku: i for i, sku in enumerate(all_skus)}
        n_skus, n_orders = len(all_skus), len(orders_df)
        
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
            
            records = []
            for i, sku in enumerate(all_skus):
                sku_appearances = np.count_nonzero(A[:, i])
                is_determined = (rank >= n_skus) or (sku_appearances >= 2)
                unit_price = float(x[i])
                
                if not is_determined:
                    min_v, max_v = self._calc_bounds(A, b, i)
                else:
                    min_v, max_v = unit_price, unit_price
                
                records.append({
                    "site": site, "sku": sku, "unit_price": round(unit_price, 2),
                    "status": "determined" if is_determined else "underdetermined",
                    "calc_method": "avg" if has_conflict else ("exact" if rank >= n_skus else "est"),
                    "confidence": int(sku_appearances),
                    "min_possible": round(float(min_v), 2) if min_v else 0,
                    "max_possible": round(float(max_v), 2) if max_v else None
                })
            
            self.supabase.table('sku_prices').delete().eq('site', site).execute()
            if records:
                self.supabase.table('sku_prices').insert(records).execute()
        except:
            pass
    
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
        return sorted(list(set([d['site'] for d in data]))) if data else []
    
    def get_site_status(self, site: str):
        prices = self.supabase.table('sku_prices').select("*").eq('site', site).execute().data
        orders = self.supabase.table('orders').select("*").eq('site', site).order('created_at', desc=True).execute().data
        return pd.DataFrame(prices), pd.DataFrame(orders)

# ============ 初始化 ============
try:
    solver = MultiSiteSolver()
except Exception as e:
    st.error(f"数据库连接失败: {e}")
    st.stop()

# Session State
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

# ============ 主界面 ============
st.title("📦 SKU 藏价求解器")

# 站点选择区（放在标题下方，避开顶部状态条）
with st.container():
    cols = st.columns([2, 3])
    with cols[0]:
        st.markdown("**当前站点**")
        existing_sites = solver.get_sites()
        site_options = existing_sites + ["+ 新建站点"]
        
        index = 0
        if st.session_state.current_site in site_options:
            index = site_options.index(st.session_state.current_site)
        
        selected = st.selectbox("选择站点", site_options, index=index, label_visibility="collapsed")
        
        if selected == "+ 新建站点":
            new_site = st.text_input("输入新站点代码（如：MY、SG）", key="new_site_input")
            if new_site:
                st.session_state.current_site = new_site.strip().upper()
        else:
            st.session_state.current_site = selected
    
    with cols[1]:
        if st.session_state.current_site and st.session_state.current_site not in ["+ 新建站点", ""]:
            st.markdown(f"**正在管理: {st.session_state.current_site}**")
            if st.button("🗑️ 清空此站点所有数据", type="secondary"):
                confirm = st.text_input(f"输入站点名 {st.session_state.current_site} 确认删除", key="confirm_clear")
                if confirm == st.session_state.current_site:
                    solver.delete_order(st.session_state.current_site, "%")  # 实现批量删除需调整方法
                    st.success("已清空")
                    st.rerun()

if not st.session_state.current_site or st.session_state.current_site in ["+ 新建站点", ""]:
    st.info("👆 请在上方选择或创建站点")
else:
    site = st.session_state.current_site
    
    # 主体布局
    left_col, right_col = st.columns([5, 7])
    
    # ========== 左侧输入 ==========
    with left_col:
        with st.container(border=True):
            st.subheader("📝 录入新订单")
            
            order_id = st.text_input("订单编号", 
                                    value=f"{site}{datetime.now().strftime('%m%d%H%M')}",
                                    key="order_id")
            
            st.markdown("**商品明细**")
            items = []
            
            for i, row in enumerate(st.session_state.sku_rows):
                c1, c2, c3 = st.columns([4, 2, 1])
                
                with c1:
                    sku = st.text_input(f"sku_{i}", value=row["sku"], 
                                       key=f"sku_{i}", placeholder="产品编码",
                                       label_visibility="collapsed")
                with c2:
                    qty = st.number_input(f"qty_{i}", min_value=1, value=row["qty"], 
                                         key=f"qty_{i}", label_visibility="collapsed")
                with c3:
                    if len(st.session_state.sku_rows) > 1:
                        if st.button("✕", key=f"del_row_{i}"):
                            remove_row(i)
                            st.rerun()
                
                if sku.strip():
                    items.append({"sku": sku.strip().upper(), "qty": int(qty)})
            
            if st.button("➕ 添加商品行", type="secondary", use_container_width=True):
                add_row()
                st.rerun()
            
            total = st.number_input("订单总藏价", min_value=0.0, value=0.0, step=10.0, 
                                   key="total_price", format="%.2f")
            
            if st.button("🚀 提交计算", type="primary", use_container_width=True):
                if not order_id:
                    st.error("请输入订单编号")
                elif len(items) == 0:
                    st.error("请至少输入一个产品编码")
                elif total <= 0:
                    st.error("总藏价必须大于0")
                else:
                    with st.spinner("计算中..."):
                        success, msg = solver.add_order(site, order_id, total, items)
                        if success:
                            st.success("已保存并重新计算")
                            st.session_state.sku_rows = [{"sku": "", "qty": 1}]
                            st.rerun()
                        else:
                            st.error(msg)
    
    # ========== 右侧结果 ==========
    with right_col:
        prices_df, orders_df = solver.get_site_status(site)
        
        # 统计卡片
        if not prices_df.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                det_count = len(prices_df[prices_df['status'] == 'determined'])
                st.metric("已确定产品", f"{det_count}")
            with c2:
                undet_count = len(prices_df[prices_df['status'] == 'underdetermined'])
                st.metric("待定产品", f"{undet_count}" if undet_count else "0")
            with c3:
                st.metric("历史订单", f"{len(orders_df)}")
            
            st.divider()
        
        # 价格表格
        if not prices_df.empty:
            tab1, tab2 = st.tabs(["✅ 已确定", "🔍 待定"])
            
            with tab1:
                det = prices_df[prices_df['status'] == 'determined']
                if not det.empty:
                    display_df = det[['sku', 'unit_price', 'confidence']].copy()
                    display_df.columns = ['产品编码', '单件藏价', '数据支撑']
                    display_df['单件藏价'] = display_df['单件藏价'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True, height=250)
                    
                    if 'avg' in det['calc_method'].values:
                        st.caption("💡 该站点存在矛盾数据，已自动取平均并放大5%")
                else:
                    st.info("暂无确定价格")
            
            with tab2:
                undet = prices_df[prices_df['status'] == 'underdetermined']
                if not undet.empty:
                    for _, row in undet.iterrows():
                        cols = st.columns([3, 2, 3])
                        with cols[0]:
                            st.markdown(f"**{row['sku']}**")
                        with cols[1]:
                            st.markdown(f"{row['unit_price']:.2f}")
                        with cols[2]:
                            if pd.notna(row['max_possible']):
                                st.caption(f"范围: {row['min_possible']:.2f} - {row['max_possible']:.2f}")
                            else:
                                st.caption("需更多数据")
                else:
                    st.success("所有产品价格已确定")
        else:
            st.info("录入订单后将在此显示计算结果")
        
        # 历史订单（带删除）
        if not orders_df.empty:
            st.divider()
            st.subheader("📋 历史订单")
            
            all_items = solver.supabase.table('order_items').select("*").eq('site', site).execute().data
            items_map = {}
            for item in all_items:
                oid = item['order_id']
                if oid not in items_map:
                    items_map[oid] = []
                items_map[oid].append(f"{item['sku']}×{item['quantity']}")
            
            for _, order in orders_df.iterrows():
                oid = order['order_id']
                
                with st.container(border=True):
                    cols = st.columns([3, 4, 2, 2])
                    
                    with cols[0]:
                        st.markdown(f"**{oid}**")
                        st.caption(f"{order['created_at'][:10]}")
                    
                    with cols[1]:
                        if oid in items_map:
                            goods_text = " | ".join(items_map[oid])
                            st.text(goods_text[:30] + "..." if len(goods_text) > 30 else goods_text)
                    
                    with cols[2]:
                        st.markdown(f"{order['total_hidden_price']:.2f}")
                    
                    with cols[3]:
                        confirm_key = f"confirm_{oid}"
                        if confirm_key not in st.session_state.delete_confirm:
                            st.session_state.delete_confirm[confirm_key] = False
                        
                        if not st.session_state.delete_confirm[confirm_key]:
                            if st.button("删除", key=f"del_btn_{oid}"):
                                st.session_state.delete_confirm[confirm_key] = True
                                st.rerun()
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button("✓", key=f"yes_{oid}", type="primary"):
                                    if solver.delete_order(site, oid):
                                        st.session_state.delete_confirm[confirm_key] = False
                                        st.rerun()
                            with c2:
                                if st.button("✕", key=f"no_{oid}"):
                                    st.session_state.delete_confirm[confirm_key] = False
                                    st.rerun()
