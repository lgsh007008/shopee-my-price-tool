import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from collections import defaultdict
import time
from functools import wraps

st.set_page_config(page_title="SKU藏价求解器", layout="wide")

# ============ 自动重试装饰器 ============
def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "Resource temporarily unavailable" in error_str or "Errno 11" in error_str or "unavailable" in error_str:
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))
                            continue
                    raise e
            return None
        return wrapper
    return decorator

# ============ 延迟初始化 + 连接管理 ============
class SupabaseManager:
    _client = None
    _last_used = None
    
    @classmethod
    def get_client(cls):
        now = datetime.now().timestamp()
        if cls._client is None or (cls._last_used and now - cls._last_used > 300):
            try:
                url = st.secrets["SUPABASE_URL"]
                key = st.secrets["SUPABASE_KEY"]
                cls._client = create_client(url, key)
                cls._last_used = now
            except Exception as e:
                st.error(f"数据库连接失败: {e}")
                raise e
        return cls._client
    
    @classmethod
    def reset(cls):
        cls._client = None
        cls._last_used = None

# ============ 数据库操作类 ============
class SymbolicSolver:
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = SupabaseManager.get_client()
        return self._client
    
    @retry_on_error(max_retries=3, delay=2)
    def add_order(self, site: str, order_id: str, total: float, items: list):
        try:
            existing = self.client.table('orders').select("*").eq('site', site).eq('order_id', order_id).execute()
            if existing.data:
                return False, "订单号已存在"
            
            self.client.table('orders').insert({
                "site": site, "order_id": order_id,
                "total_hidden_price": total,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            batch = [{
                "site": site, "order_id": order_id,
                "sku": item['sku'].upper().strip(),
                "quantity": int(item['qty'])
            } for item in items if item['sku']]
            
            if batch:
                self.client.table('order_items').insert(batch).execute()
            
            return True, "保存成功"
        except Exception as e:
            if "Resource" in str(e) or "Errno" in str(e):
                SupabaseManager.reset()
            raise e
    
    @retry_on_error(max_retries=3, delay=2)
    def delete_order(self, site: str, order_id: str):
        try:
            self.client.table('order_items').delete().eq('site', site).eq('order_id', order_id).execute()
            self.client.table('orders').delete().eq('site', site).eq('order_id', order_id).execute()
            return True
        except Exception as e:
            SupabaseManager.reset()
            raise e
    
    @retry_on_error(max_retries=3, delay=2)
    def set_manual_price(self, site: str, sku: str, price: float):
        try:
            self.client.table('manual_prices').upsert({
                "site": site, "sku": sku, "manual_price": price,
                "confirmed_at": datetime.now().isoformat()
            }).execute()
            return True
        except Exception as e:
            SupabaseManager.reset()
            raise e
    
    @retry_on_error(max_retries=3, delay=2)
    def clear_manual_price(self, site: str, sku: str):
        try:
            self.client.table('manual_prices').delete().eq('site', site).eq('sku', sku).execute()
            return True
        except:
            return False
    
    @retry_on_error(max_retries=3, delay=2)
    def get_site_data(self, site: str):
        try:
            orders = self.client.table('orders').select("*").eq('site', site).execute().data or []
            items = self.client.table('order_items').select("*").eq('site', site).execute().data or []
            manual = self.client.table('manual_prices').select("*").eq('site', site).execute().data or []
            return orders, items, {m['sku']: m['manual_price'] for m in manual}
        except Exception as e:
            SupabaseManager.reset()
            raise e

# ============ 求解算法类 ============
class SiteSolver:
    def __init__(self):
        self.db = SymbolicSolver()
    
    def solve(self, site: str):
        orders, items, manual = self.db.get_site_data(site)
        if not orders:
            return {}, {}, [], [], []
        return self._solve_logic(orders, items, manual)
    
    def _solve_logic(self, orders, items, manual_prices):
        order_map = {}
        for o in orders:
            order_map[o['order_id']] = {'total': o['total_hidden_price'], 'items': []}
        for it in items:
            if it['order_id'] in order_map:
                order_map[it['order_id']]['items'].append(it)
        
        all_skus = list(set(it['sku'] for it in items))
        determined = {}
        conflicts = {}
        constraints = []
        
        for sku, price in manual_prices.items():
            determined[sku] = (price, "manual")
        
        changed = True
        iteration = 0
        while changed and iteration < 50:
            changed = False
            iteration += 1
            
            for oid, data in order_map.items():
                total = data['total']
                o_items = data['items']
                
                known_sum = 0
                unknown_items = []
                
                for it in o_items:
                    sku = it['sku']
                    qty = it['quantity']
                    if sku in determined:
                        known_sum += qty * determined[sku][0]
                    else:
                        unknown_items.append((sku, qty))
                
                remaining = total - known_sum
                
                if len(unknown_items) == 1:
                    sku, qty = unknown_items[0]
                    if qty == 0:
                        val = 0
                    else:
                        val = remaining / qty
                    
                    if sku in determined:
                        old_val, old_src = determined[sku]
                        if abs(old_val - val) > 0.01:
                            if sku not in conflicts:
                                conflicts[sku] = []
                            conflict_info = {
                                'value': val,
                                'derived_from': oid,
                                'equation': f"{qty}×{sku} = {remaining:.2f} (订单{oid})",
                                'current': old_val,
                                'current_src': old_src
                            }
                            if not any(abs(c['value'] - val) < 0.01 for c in conflicts[sku]):
                                conflicts[sku].append(conflict_info)
                    else:
                        determined[sku] = (val, f"derived_{oid}")
                        changed = True
        
        underdetermined_skus = set(all_skus) - set(determined.keys())
        
        if underdetermined_skus:
            for oid, data in order_map.items():
                total = data['total']
                o_items = data['items']
                
                known_sum = 0
                unknown_terms = []
                missing_skus = []
                
                for it in o_items:
                    sku = it['sku']
                    qty = it['quantity']
                    if sku in determined:
                        known_sum += qty * determined[sku][0]
                    else:
                        unknown_terms.append(f"{qty}×{sku}")
                        missing_skus.append(sku)
                
                remaining = total - known_sum
                
                if len(unknown_terms) >= 2:
                    equation = " + ".join(unknown_terms) + f" = {remaining:.2f}"
                    constraints.append({
                        'order_id': oid,
                        'equation': equation,
                        'missing_skus': missing_skus
                    })
        
        return determined, conflicts, constraints, list(underdetermined_skus), orders

# ============ 界面初始化 ============
try:
    solver = SiteSolver()
except Exception as e:
    st.error(f"初始化失败: {e}")
    st.stop()

SITES = {
    'MX': '🇲🇽 墨西哥',
    'TH': '🇹🇭 泰国', 
    'PH': '🇵🇭 菲律宾'
}

if 'sku_rows' not in st.session_state:
    st.session_state.sku_rows = [{"sku": "", "qty": 1}]
if 'delete_confirm' not in st.session_state:
    st.session_state.delete_confirm = {}
if 'current_site' not in st.session_state:
    st.session_state.current_site = 'MX'

def add_row():
    st.session_state.sku_rows.append({"sku": "", "qty": 1})

def remove_row(index):
    if len(st.session_state.sku_rows) > 1:
        st.session_state.sku_rows.pop(index)
        st.rerun()

# ============ CSS ============
st.markdown("""
<style>
    .block-container {padding-top: 2rem !important;}
    .conflict-box {background-color: #f8d7da; border: 2px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 8px;}
    .exact-box {background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 5px 0;}
    .constraint-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0;}
    .derived-box {background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

# ============ 主界面 ============
st.title("📦 SKU 藏价求解器")

cols = st.columns(3)
for i, (key, label) in enumerate(SITES.items()):
    with cols[i]:
        btn_type = "primary" if st.session_state.current_site == key else "secondary"
        if st.button(label, key=f"site_{key}", type=btn_type, use_container_width=True):
            st.session_state.current_site = key
            st.rerun()

site = st.session_state.current_site
st.markdown(f"<h3 style='text-align: center;'>当前站点: {SITES[site]}</h3>", unsafe_allow_html=True)

left, right = st.columns([4, 6])

with left:
    st.subheader("📝 录入新订单")
    with st.container(border=True):
        order_id = st.text_input("订单编号", value=f"{site}{datetime.now().strftime('%m%d%H%M')}")
        
        items = []
        for i, row in enumerate(st.session_state.sku_rows):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                sku = st.text_input(f"产品编码_{i}", value=row["sku"], key=f"sku_{i}", placeholder="如: A", label_visibility="collapsed")
            with c2:
                qty = st.number_input(f"数量_{i}", min_value=1, value=row["qty"], key=f"qty_{i}", label_visibility="collapsed")
            with c3:
                if len(st.session_state.sku_rows) > 1:
                    if st.button("✕", key=f"del_{i}"):
                        remove_row(i)
            
            if sku.strip():
                items.append({"sku": sku.strip().upper(), "qty": qty})
        
        if st.button("➕ 添加商品行", use_container_width=True):
            add_row()
            st.rerun()
        
        total = st.number_input("订单总藏价", min_value=0.0, value=0.0, step=10.0, format="%.2f")
        
        if st.button("🚀 提交订单", type="primary", use_container_width=True):
            if not order_id: 
                st.error("请输入订单编号")
            elif not items: 
                st.error("请输入产品编码")
            elif total <= 0: 
                st.error("总藏价必须大于0")
            else:
                with st.spinner("保存中..."):
                    try:
                        success, msg = solver.db.add_order(site, order_id, total, items)
                        if success:
                            st.success("已保存")
                            st.session_state.sku_rows = [{"sku": "", "qty": 1}]
                            st.rerun()
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"保存失败: {e}")

with right:
    try:
        determined, conflicts, constraints, underdetermined, orders = solver.solve(site)
    except Exception as e:
        st.error(f"计算失败: {e}")
        determined, conflicts, constraints, underdetermined, orders = {}, {}, [], [], []
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已确定", len(determined))
    c2.metric("矛盾", len(conflicts))
    c3.metric("约束", len(constraints))
    c4.metric("订单", len(orders))
    
    st.divider()
    
    if determined:
        st.subheader("✅ 已确定藏价")
        data = []
        for sku, (val, source) in determined.items():
            if source == "manual":
                src_display = "📝 手动确认"
            elif source.startswith("derived_"):
                oid = source.replace("derived_", "")
                src_display = f"🤖 推导自 {oid}"
            else:
                src_display = source
            data.append({"SKU": sku, "藏价": f"{val:.2f}", "来源": src_display})
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    if conflicts:
        st.subheader("⚠️ 发现价格矛盾！")
        st.error("同一SKU在不同订单中推出不同值，请确认")
        
        for sku, conflict_list in conflicts.items():
            with st.container(border=True):
                st.markdown(f"**SKU: {sku}**")
                
                for c in conflict_list:
                    st.markdown(f"- 订单 **{c['derived_from']}**: 推导值 **{c['value']:.2f}** ({c['equation']})")
                
                current_val, current_src = determined.get(sku, (0, "无"))
                st.markdown(f"**当前采用**: {current_val:.2f} (来自{current_src})")
                
                cols = st.columns([2, 1])
                with cols[0]:
                    new_price = st.number_input(
                        f"确认价_{sku}", 
                        min_value=0.0,
                        value=float(conflict_list[0]['value']),
                        step=1.0,
                        key=f"manual_input_{sku}",
                        label_visibility="collapsed"
                    )
                with cols[1]:
                    if st.button(f"✓ 确认", key=f"confirm_{sku}", type="primary", use_container_width=True):
                        try:
                            solver.db.set_manual_price(site, sku, new_price)
                            st.success(f"已确认 {sku}={new_price:.2f}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"确认失败: {e}")
                
                if st.button(f"🗑️ 清除手动确认", key=f"clear_{sku}"):
                    try:
                        solver.db.clear_manual_price(site, sku)
                        st.rerun()
                    except:
                        pass
    
    if constraints:
        st.subheader("🔗 欠定约束")
        for cons in constraints:
            with st.container(border=True):
                st.markdown(f"**订单 {cons['order_id']}**: {cons['equation']}")
                st.caption(f"涉及: {', '.join(cons['missing_skus'])}")
    
    if not determined and not conflicts and not constraints:
        st.info("录入第一个订单后开始计算")
    
    if orders:
        st.divider()
        st.subheader("📋 历史订单")
        
        try:
            _, items_data, _ = solver.db.get_site_data(site)
        except:
            items_data = []
        
        for order in orders:
            oid = order['order_id']
            order_items = [it for it in items_data if it['order_id'] == oid]
            items_str = ", ".join([f"{it['sku']}×{it['quantity']}" for it in order_items])
            
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                
                with col1:
                    st.markdown(f"**{oid}**")
                    st.caption(f"{order['created_at'][:10]}")
                with col2:
                    st.text(items_str[:30] + "..." if len(items_str) > 30 else items_str)
                with col3:
                    st.markdown(f"{order['total_hidden_price']:.2f}")
                with col4:
                    confirm_key = f"conf_{oid}"
                    if confirm_key not in st.session_state.delete_confirm:
                        st.session_state.delete_confirm[confirm_key] = False
                    
                    if not st.session_state.delete_confirm[confirm_key]:
                        if st.button("🗑️", key=f"btn_{oid}"):
                            st.session_state.delete_confirm[confirm_key] = True
                            st.rerun()
                    else:
                        if st.button("✓", key=f"yes_{oid}", type="primary"):
                            try:
                                solver.db.delete_order(site, oid)
                                st.session_state.delete_confirm[confirm_key] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")
                        if st.button("✕", key=f"no_{oid}"):
                            st.session_state.delete_confirm[confirm_key] = False
                            st.rerun()
