import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
import time
from functools import wraps

st.set_page_config(page_title="SKU藏价求解器", layout="wide")

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "Resource temporarily unavailable" in error_str or "Errno 11" in error_str:
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))
                            continue
                    raise e
            return None
        return wrapper
    return decorator

class SupabaseManager:
    _client = None
    _last_used = None
    
    @classmethod
    def get_client(cls):
        now = datetime.now().timestamp()
        if cls._client is None or (cls._last_used and now - cls._last_used > 300):
            url = st.secrets["SUPABASE_URL"].strip()
            key = st.secrets["SUPABASE_KEY"].strip()
            cls._client = create_client(url, key)
            cls._last_used = now
        return cls._client
    
    @classmethod
    def reset(cls):
        cls._client = None

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
    
    @retry_on_error(max_retries=3, delay=2)
    def delete_order(self, site: str, order_id: str):
        self.client.table('order_items').delete().eq('site', site).eq('order_id', order_id).execute()
        self.client.table('orders').delete().eq('site', site).eq('order_id', order_id).execute()
        return True
    
    @retry_on_error(max_retries=3, delay=2)
    def set_manual_price(self, site: str, sku: str, price: float):
        self.client.table('manual_prices').upsert({
            "site": site, "sku": sku, "manual_price": price,
            "confirmed_at": datetime.now().isoformat()
        }).execute()
        return True
    
    @retry_on_error(max_retries=3, delay=2)
    def clear_manual_price(self, site: str, sku: str):
        self.client.table('manual_prices').delete().eq('site', site).eq('sku', sku).execute()
        return True
    
    @retry_on_error(max_retries=3, delay=2)
    def get_site_data(self, site: str):
        orders = self.client.table('orders').select("*").eq('site', site).execute().data or []
        items = self.client.table('order_items').select("*").eq('site', site).execute().data or []
        manual = self.client.table('manual_prices').select("*").eq('site', site).execute().data or []
        return orders, items, {m['sku']: m['manual_price'] for m in manual}

class SiteSolver:
    def __init__(self):
        self.db = SymbolicSolver()
    
    def solve(self, site: str):
        orders, items, manual = self.db.get_site_data(site)
        if not orders:
            return {}, {}, [], [], []
        return self._solve_logic(orders, items, manual)
    
    def _solve_logic(self, orders, items, manual_prices):
        order_map = {o['order_id']: {'total': o['total_hidden_price'], 'items': []} for o in orders}
        for it in items:
            if it['order_id'] in order_map:
                order_map[it['order_id']]['items'].append(it)
        
        all_skus = list(set(it['sku'] for it in items))
        determined = dict(manual_prices)  # 手动确认值优先
        conflicts = {}  # 矛盾记录：sku -> [可能值列表]
        inconsistent_orders = []  # 数据不一致的订单
        
        # 迭代求解直到没有新值
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
                        known_sum += qty * determined[sku]
                    else:
                        unknown_items.append((sku, qty))
                
                remaining = total - known_sum
                
                # 关键修复1：检查是否所有SKU都已确定但总价不匹配（数据矛盾）
                if len(unknown_items) == 0:
                    if abs(remaining) > 0.01:  # 总价不匹配
                        # 这意味着已确定的SKU值无法解释此订单的总价
                        # 如果订单只有一个SKU，这就是该SKU的矛盾
                        if len(o_items) == 1:
                            sku = o_items[0]['sku']
                            qty = o_items[0]['quantity']
                            implied_price = total / qty if qty != 0 else 0
                            if sku not in conflicts:
                                conflicts[sku] = []
                            conflict_info = {
                                'value': implied_price,
                                'derived_from': oid,
                                'equation': f"{qty}×{sku} = {total} (订单总价)",
                                'current': determined[sku],
                                'current_src': '已确定值',
                                'type': 'order_mismatch'
                            }
                            # 避免重复添加相同的矛盾
                            if not any(abs(c['value'] - implied_price) < 0.01 for c in conflicts[sku]):
                                conflicts[sku].append(conflict_info)
                        else:
                            # 多个SKU都已确定但总价不对，记录为不一致订单
                            inconsistent_orders.append({
                                'order_id': oid,
                                'expected': known_sum,
                                'actual': total,
                                'diff': remaining
                            })
                    continue  # 无需进一步处理
                
                # 只有一个未知数，可以求解
                if len(unknown_items) == 1:
                    sku, qty = unknown_items[0]
                    if qty == 0:
                        val = 0
                    else:
                        val = remaining / qty
                    
                    # 关键修复2：如果该SKU已有确定值，检查是否矛盾
                    if sku in determined:
                        old_val = determined[sku]
                        if abs(old_val - val) > 0.01:
                            if sku not in conflicts:
                                conflicts[sku] = []
                            conflict_info = {
                                'value': val,
                                'derived_from': oid,
                                'equation': f"{qty}×{sku} = {remaining:.2f} (基于订单{oid})",
                                'current': old_val,
                                'current_src': '之前确定',
                                'type': 'derivation_conflict'
                            }
                            if not any(abs(c['value'] - val) < 0.01 for c in conflicts[sku]):
                                conflicts[sku].append(conflict_info)
                    else:
                        # 新确定值
                        determined[sku] = val
                        changed = True
        
        # 收集欠定约束（多个未知数）
        constraints = []
        underdetermined = set(all_skus) - set(determined.keys())
        
        if underdetermined:
            for oid, data in order_map.items():
                total = data['total']
                o_items = data['items']
                
                known_sum = sum(it['quantity'] * determined[it['sku']] for it in o_items if it['sku'] in determined)
                unknown_terms = [(it['quantity'], it['sku']) for it in o_items if it['sku'] not in determined]
                remaining = total - known_sum
                
                if len(unknown_terms) >= 2:
                    equation = " + ".join([f"{qty}×{sku}" for qty, sku in unknown_terms]) + f" = {remaining:.2f}"
                    constraints.append({
                        'order_id': oid,
                        'equation': equation,
                        'missing_skus': [sku for _, sku in unknown_terms]
                    })
        
        return determined, conflicts, constraints, list(underdetermined), orders, inconsistent_orders

# ============ 界面 ============
try:
    solver = SiteSolver()
except Exception as e:
    st.error(f"初始化失败: {e}")
    st.stop()

SITES = {'MX': '🇲🇽 墨西哥', 'TH': '🇹🇭 泰国', 'PH': '🇵🇭 菲律宾'}

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

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important;}
    .conflict-box {background-color: #f8d7da; border: 2px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 8px;}
    .warning-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

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
                sku = st.text_input(f"sku_{i}", value=row["sku"], key=f"sku_{i}", placeholder="如: A", label_visibility="collapsed")
            with c2:
                qty = st.number_input(f"qty_{i}", min_value=1, value=row["qty"], key=f"qty_{i}", label_visibility="collapsed")
            with c3:
                if len(st.session_state.sku_rows) > 1 and st.button("✕", key=f"del_{i}"):
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
        determined, conflicts, constraints, underdetermined, orders, inconsistent = solver.solve(site)
    except Exception as e:
        st.error(f"计算失败: {e}")
        determined, conflicts, constraints, underdetermined, orders, inconsistent = {}, {}, [], [], [], []
    
    # 显示统计数据
    c1, c2, c3 = st.columns(3)
    c1.metric("已确定SKU", len(determined))
    c2.metric("矛盾待解决", len(conflicts))
    c3.metric("历史订单", len(orders))
    
    # 显示矛盾警告（红色大框）
    if conflicts:
        st.markdown("---")
        st.error("⚠️ 发现价格矛盾！以下SKU推导出多个不同值")
        
        for sku, conflict_list in conflicts.items():
            with st.container(border=True):
                st.markdown(f"#### SKU: {sku}")
                
                # 显示所有可能的值
                for i, c in enumerate(conflict_list, 1):
                    st.markdown(f"**推导{i}**: {c['value']:.2f} ({c['equation']})")
                
                # 显示当前采用的值
                current_val = determined.get(sku, "未确定")
                st.markdown(f"**当前系统保留值**: {current_val if isinstance(current_val, str) else f'{current_val:.2f}'}")
                
                # 手动确认输入
                st.markdown("---")
                st.markdown("**手动确认最终值：**")
                cols = st.columns([2, 1])
                with cols[0]:
                    # 默认取平均值或第一个冲突值
                    default_val = sum(c['value'] for c in conflict_list) / len(conflict_list)
                    new_price = st.number_input(
                        f"确认 {sku} 的藏价", 
                        min_value=0.0,
                        value=float(default_val),
                        step=0.5,
                        key=f"manual_{sku}"
                    )
                with cols[1]:
                    if st.button(f"✓ 确认", key=f"confirm_{sku}", type="primary", use_container_width=True):
                        try:
                            solver.db.set_manual_price(site, sku, new_price)
                            st.success(f"已确认 {sku} = {new_price:.2f}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"失败: {e}")
                
                if st.button(f"🗑️ 清除手动确认", key=f"clear_{sku}"):
                    solver.db.clear_manual_price(site, sku)
                    st.rerun()
    
    # 显示已确定价格
    if determined and not conflicts:  # 没有矛盾时才显示确定列表（避免混淆）
        st.markdown("---")
        st.subheader("✅ 已确定藏价")
        data = [{"SKU": k, "藏价": f"{v:.2f}"} for k, v in determined.items()]
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    # 显示欠定约束
    if constraints:
        st.markdown("---")
        st.subheader("🔗 欠定约束（需更多数据）")
        for cons in constraints:
            st.info(f"订单 {cons['order_id']}: {cons['equation']}")
        st.caption(f"涉及待定SKU: {', '.join(underdetermined)}")
    
    # 显示历史订单
    if orders:
        st.markdown("---")
        st.subheader("📋 历史订单")
        
        _, items_data, _ = solver.db.get_site_data(site)
        
        for order in orders:
            oid = order['order_id']
            o_items = [it for it in items_data if it['order_id'] == oid]
            items_str = ", ".join([f"{it['sku']}×{it['quantity']}" for it in o_items])
            
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                
                with col1:
                    st.markdown(f"**{oid}**")
                    st.caption(f"{order['created_at'][:10]}")
                with col2:
                    st.text(items_str[:25] + "..." if len(items_str) > 25 else items_str)
                with col3:
                    st.markdown(f"{order['total_hidden_price']:.2f}")
                with col4:
                    ckey = f"del_{oid}"
                    if ckey not in st.session_state.delete_confirm:
                        st.session_state.delete_confirm[ckey] = False
                    
                    if not st.session_state.delete_confirm[ckey]:
                        if st.button("🗑️", key=f"btn_{oid}"):
                            st.session_state.delete_confirm[ckey] = True
                            st.rerun()
                    else:
                        if st.button("✓", key=f"yes_{oid}", type="primary"):
                            solver.db.delete_order(site, oid)
                            st.session_state.delete_confirm[ckey] = False
                            st.rerun()
                        if st.button("✕", key=f"no_{oid}"):
                            st.session_state.delete_confirm[ckey] = False
                            st.rerun()
