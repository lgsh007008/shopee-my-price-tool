import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.linalg import svd, lstsq
from scipy.optimize import linprog
import json
from supabase import create_client, Client

st.set_page_config(page_title="SKU藏价求解器-云端版", layout="wide")

# ============ 数据库层（Supabase） ============

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class CloudSolver:
    def __init__(self):
        self.supabase = init_supabase()
        
    def add_order(self, order_id: str, total_hidden_price: float, items: list):
        """添加订单到云端"""
        try:
            # 插入订单
            self.supabase.table('orders').insert({
                "order_id": order_id,
                "total_hidden_price": total_hidden_price,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            # 插入明细
            for item in items:
                self.supabase.table('order_items').insert({
                    "order_id": order_id,
                    "sku": item['sku'],
                    "quantity": item['qty']
                }).execute()
            
            # 触发重算
            self._analyze_solvability()
            return True
        except Exception as e:
            st.error(f"保存失败: {e}")
            return False
    
    def _get_all_data(self):
        """从云端拉取全部数据"""
        orders = self.supabase.table('orders').select("*").execute().data
        items = self.supabase.table('order_items').select("*").execute().data
        return pd.DataFrame(orders), pd.DataFrame(items)
    
    def _analyze_solvability(self):
        """核心算法（同之前，但结果存回云端）"""
        orders_df, items_df = self._get_all_data()
        
        if orders_df.empty or items_df.empty:
            return
        
        # 数据对齐（处理空值）
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
            # SVD分析
            U, s, Vt = svd(A)
            rank = np.sum(s > 1e-10)
            
            x, _, _, _ = lstsq(A, b)
            x = np.maximum(x, 0)  # 非负
            
            # 确定哪些SKU是确定的
            null_space = self._nullspace(A)
            determined_skus = set()
            
            if null_space.shape[1] > 0:
                col_in_null = np.abs(null_space).max(axis=1) > 1e-10
                for i, sku in enumerate(all_skus):
                    if not col_in_null[i]:
                        determined_skus.add(sku)
            else:
                determined_skus = set(all_skus)
            
            # 批量更新到Supabase（先清空再插入）
            self.supabase.table('sku_prices').delete().neq('sku', 'placeholder').execute()
            
            records = []
            for i, sku in enumerate(all_skus):
                status = 'determined' if sku in determined_skus else 'underdetermined'
                min_v, max_v = self._calc_bounds(A, b, i) if status != 'determined' else (x[i], x[i])
                
                records.append({
                    "sku": sku,
                    "unit_price": round(float(x[i]), 2),
                    "status": status,
                    "confidence": int(rank) if status == 'determined' else 0,
                    "min_possible": round(float(min_v), 2) if min_v is not None else None,
                    "max_possible": round(float(max_v), 2) if max_v is not None else None
                })
            
            if records:
                self.supabase.table('sku_prices').insert(records).execute()
                
        except Exception as e:
            st.error(f"计算错误: {e}")
    
    def _nullspace(self, A, tol=1e-10):
        U, s, Vt = svd(A)
        rank = np.sum(s > tol)
        null_mask = np.ones(Vt.shape[0], dtype=bool)
        null_mask[:rank] = False
        null_space = Vt[null_mask].T
        return null_space
    
    def _calc_bounds(self, A, b, col_idx):
        n = A.shape[1]
        try:
            res_min = linprog(np.eye(n)[col_idx], A_eq=A, b_eq=b, bounds=[(0, None)]*n, method='highs')
            res_max = linprog(-np.eye(n)[col_idx], A_eq=A, b_eq=b, bounds=[(0, None)]*n, method='highs')
            return (res_min.x[col_idx] if res_min.success else 0), (res_max.x[col_idx] if res_max.success else None)
        except:
            return 0, None
    
    def get_current_status(self):
        """获取当前状态"""
        prices = self.supabase.table('sku_prices').select("*").execute().data
        orders = self.supabase.table('orders').select("*").order('created_at', desc=True).execute().data
        return pd.DataFrame(prices), pd.DataFrame(orders)
    
    def delete_all(self):
        """清空数据（危险操作）"""
        self.supabase.table('order_items').delete().neq('id', 0).execute()
        self.supabase.table('orders').delete().neq('order_id', 'none').execute()
        self.supabase.table('sku_prices').delete().neq('sku', 'none').execute()

# ============ Streamlit 界面 ============

st.title("🔍 SKU 藏价求解器（云端版）")
st.caption("数据存储在Supabase，支持多设备同步")

# 初始化（自动连接云端）
try:
    solver = CloudSolver()
    st.success("✅ 已连接云端数据库")
except Exception as e:
    st.error(f"连接失败，请检查Secrets配置: {e}")
    st.stop()

# 侧边栏
with st.sidebar:
    st.header("⚠️ 危险操作")
    if st.button("🗑️ 清空所有云端数据", type="secondary"):
        confirm = st.text_input("输入 DELETE 确认删除")
        if confirm == "DELETE":
            solver.delete_all()
            st.rerun()

# 主界面：录入
st.subheader("📝 录入新订单")
with st.form("add_order"):
    cols = st.columns([2, 3, 2])
    with cols[0]:
        order_id = st.text_input("订单号", value=f"ORD{datetime.now().strftime('%m%d%H%M')}")
    with cols[1]:
        items_input = st.text_input("商品（格式：A:1,B:2）", placeholder="A:1, B:1")
    with cols[2]:
        total = st.number_input("总藏价", min_value=0.0, value=100.0)
    
    if st.form_submit_button("🚀 提交计算", use_container_width=True):
        if items_input:
            try:
                items = []
                for part in items_input.split(','):
                    sku, qty = part.strip().split(':')
                    items.append({"sku": sku.strip(), "qty": int(qty)})
                
                with st.spinner("计算中..."):
                    if solver.add_order(order_id, total, items):
                        st.success("✅ 已保存到云端并重新计算")
            except Exception as e:
                st.error(f"格式错误: {e}")

# 显示结果
st.divider()
prices_df, orders_df = solver.get_current_status()

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ 已确定单价")
    if not prices_df.empty:
        det = prices_df[prices_df['status'] == 'determined'][['sku', 'unit_price']]
        if not det.empty:
            st.dataframe(det.rename(columns={'sku': 'SKU', 'unit_price': '单价'}), hide_index=True)
        else:
            st.info("录入更多订单来确定价格")
    else:
        st.info("暂无数据")

with col2:
    st.subheader("🔍 欠定SKU")
    if not prices_df.empty:
        undet = prices_df[prices_df['status'] == 'underdetermined']
        for _, row in undet.iterrows():
            range_str = ""
            if pd.notna(row['min_possible']) and pd.notna(row['max_possible']):
                range_str = f"[{row['min_possible']:.0f}-{row['max_possible']:.0f}]"
            st.metric(row['sku'], f"¥{row['unit_price']}", range_str)

# 历史记录
with st.expander("📋 历史订单（云端实时同步）"):
    if not orders_df.empty:
        st.dataframe(orders_df[['order_id', 'total_hidden_price', 'created_at']], hide_index=True)
    else:
        st.info("暂无历史订单")
