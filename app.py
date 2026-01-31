import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from collections import defaultdict

st.set_page_config(page_title="SKU藏价求解器-符号代数版", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important;}
    .exact-box {background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 5px 0;}
    .conflict-box {background-color: #f8d7da; border: 2px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 8px;}
    .derived-box {background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 5px 0;}
    .constraint-box {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

# 固定三站点
SITES = {
    'MX': '🇲🇽 墨西哥',
    'TH': '🇹🇭 泰国', 
    'PH': '🇵🇭 菲律宾'
}

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class SymbolicSolver:
    def __init__(self):
        self.supabase = init_supabase()
    
    def add_order(self, site: str, order_id: str, total: float, items: list):
        try:
            existing = self.supabase.table('orders').select("*").eq('site', site).eq('order_id', order_id).execute()
            if existing.data:
                return False, "订单号已存在"
            
            self.supabase.table('orders').insert({
                "site": site, "order_id": order_id,
                "total_hidden_price": total,
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
            self.supabase.table('manual_prices').delete().eq('site', site).execute()  # 清除手动确认（因为数据变了）
            return True
        except:
            return False
    
    def set_manual_price(self, site: str, sku: str, price: float):
        try:
            self.supabase.table('manual_prices').upsert({
                "site": site, "sku": sku, "manual_price": price,
                "confirmed_at": datetime.now().isoformat()
            }).execute()
            return True
        except:
            return False
    
    def clear_manual_price(self, site: str, sku: str):
        try:
            self.supabase.table('manual_prices').delete().eq('site', site).eq('sku', sku).execute()
            return True
        except:
            return False
    
    def get_site_data(self, site: str):
        orders = self.supabase.table('orders').select("*").eq('site', site).execute().data or []
        items = self.supabase.table('order_items').select("*").eq('site', site).execute().data or []
        manual = self.supabase.table('manual_prices').select("*").eq('site', site).execute().data or []
        manual_dict = {m['sku']: m['manual_price'] for m in manual}
        return orders, items, manual_dict
    
    def solve_site(self, site: str):
        """
        核心算法：
        1. 先加载手动确认值作为硬约束
        2. 构建所有订单方程
        3. 迭代求解：只要方程中只有一个未知数，就能解出
        4. 检测矛盾：同一个SKU被不同方程解出不同值
        5. 剩余欠定方程显示约束关系（如2D+E=110）
        """
        orders, items, manual_prices = self.get_site_data(site)
        
        if not orders:
            return {}, {}, [], [], orders  # determined, conflicts, constraints, underdetermined, raw_orders
        
        # 数据结构
        order_map = {}  # order_id -> {items: [], total: x}
        for o in orders:
            order_map[o['order_id']] = {'total': o['total_hidden_price'], 'items': []}
        for it in items:
            if it['order_id'] in order_map:
                order_map[it['order_id']]['items'].append(it)
        
        all_skus = list(set(it['sku'] for it in items))
        
        # 结果存储
        determined = {}  # sku -> (value, source)  source可以是"manual"或"derived_from_order_X"
        conflicts = {}   # sku -> [possible_values]  矛盾候选值
        constraints = [] # 欠定约束方程字符串列表
        
        # 第一步：应用手动确认值
        for sku, price in manual_prices.items():
            determined[sku] = (price, "manual")
        
        # 第二步：迭代求解（基于当前已知值，解出能解的所有未知数）
        changed = True
        iteration = 0
        while changed and iteration < 50:  # 防止无限循环
            changed = False
            iteration += 1
            
            for oid, data in order_map.items():
                total = data['total']
                o_items = data['items']
                
                # 计算已知部分
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
                
                # 情况1：只有一个未知数 -> 可解
                if len(unknown_items) == 1:
                    sku, qty = unknown_items[0]
                    if remaining < 0:  # 矛盾检查：剩余为负
                        val = 0  # 最小0，但标记矛盾
                    else:
                        val = remaining / qty
                    
                    # 检查是否已有值
                    if sku in determined:
                        old_val, old_src = determined[sku]
                        if abs(old_val - val) > 0.01:  # 发现矛盾！
                            if sku not in conflicts:
                                conflicts[sku] = []
                            # 添加这个新推导值作为矛盾候选
                            conflict_info = {
                                'value': val,
                                'derived_from': oid,
                                'equation': f"{qty}×{sku} = {remaining:.2f} (基于订单{oid})",
                                'against': f"当前值 {old_val} (来自{old_src})"
                            }
                            if not any(abs(c['value'] - val) < 0.01 for c in conflicts[sku]):
                                conflicts[sku].append(conflict_info)
                    else:
                        # 全新确定
                        determined[sku] = (val, f"derived_{oid}")
                        changed = True
                
                # 情况2：零个未知数 -> 验证一致性
                elif len(unknown_items) == 0:
                    if abs(remaining) > 0.01:  # 矛盾！所有已知加起来不等于total
                        pass  # 可以在这里记录方程不一致错误
        
        # 第三步：收集欠定约束（还有多个未知数的方程）
        underdetermined_skus = set(all_skus) - set(determined.keys())
        
        if underdetermined_skus:
            for oid, data in order_map.items():
                total = data['total']
                o_items = data['items']
                
                known_sum = 0
                unknown_terms = []
                
                for it in o_items:
                    sku = it['sku']
                    qty = it['quantity']
                    if sku in determined:
                        known_sum += qty * determined[sku][0]
                    else:
                        unknown_terms.append(f"{qty}×{sku}")
                
                remaining = total - known_sum
                
                if len(unknown_terms) >= 2:  # 欠定
                    equation = " + ".join(unknown_terms) + f" = {remaining:.2f}"
                    constraints.append({
                        'order_id': oid,
                        'equation': equation,
                        'missing_skus': [sku for sku, qty in [(it['sku'], it['quantity']) for it in o_items] if sku in underdetermined_skus]
                    })
        
        return determined, conflicts, constraints, list(underdetermined_skus), orders

# ============ 初始化 ============
solver = SymbolicSolver()

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

# ============ 标题 + 站点选择 ============
st.title("📦 SKU 藏价求解器 - 符号代数版")

cols = st.columns(3)
for i, (key, label) in enumerate(SITES.items()):
    with cols[i]:
        btn_type = "primary" if st.session_state.current_site == key else "secondary"
        if st.button(label, key=f"site_{key}", type=btn_type, use_container_width=True):
            st.session_state.current_site = key
            st.rerun()

site = st.session_state.current_site
st.markdown(f"<h3 style='text-align: center; color: #666;'>当前站点: {SITES[site]}</h3>", unsafe_allow_html=True)

# 主体布局
left, right = st.columns([4, 6])

with left:
    st.subheader("📝 录入新订单")
    with st.container(border=True):
        order_id = st.text_input("订单编号", value=f"{site}{datetime.now().strftime('%m%d%H%M')}")
        
        items = []
        for i, row in enumerate(st.session_state.sku_rows):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                sku = st.text_input(f"SKU_{i}", value=row["sku"], key=f"sku_{i}", placeholder="如: A", label_visibility="collapsed")
            with c2:
                qty = st.number_input(f"Qty_{i}", min_value=1, value=row["qty"], key=f"qty_{i}", label_visibility="collapsed")
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
        
        if st.button("🚀 提交订单", type="primary", use_container_width=True):
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
    determined, conflicts, constraints, underdetermined, orders = solver.solve_site(site)
    
    # 统计
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("已确定", len(determined))
    c2.metric("⚠️ 矛盾", len(conflicts))
    c3.metric("欠定约束", len(constraints))
    c4.metric("待定SKU", len(underdetermined))
    
    # 1. 显示已确定（含推导路径）
    if determined:
        st.subheader("✅ 已确定藏价")
        
        data = []
        for sku, (val, source) in determined.items():
            source_display = ""
            if source == "manual":
                source_display = "📝 手动确认"
            elif source.startswith("derived_"):
                oid = source.replace("derived_", "")
                source_display = f"🤖 推导自订单 {oid}"
            data.append({"SKU": sku, "藏价": f"{val:.2f}", "来源": source_display})
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 2. 矛盾处理（核心功能）
    if conflicts:
        st.subheader("⚠️ 发现价格矛盾！需手动确认")
        st.error("以下SKU从不同订单推导出了不同价格，请确认最终值")
        
        for sku, conflict_list in conflicts.items():
            with st.container(border=True):
                st.markdown(f"**SKU: {sku}** 发现 {len(conflict_list)} 个不同推导值：")
                
                # 显示矛盾详情
                for c in conflict_list:
                    st.markdown(f"- 订单 **{c['derived_from']}**: 推导 **{c['value']:.2f}** ({c['equation']})")
                
                st.markdown(f"**当前系统采用值**: {determined.get(sku, ('无', ''))[0]:.2f}")
                
                # 手动输入确认
                cols = st.columns([2, 1])
                with cols[0]:
                    new_price = st.number_input(
                        f"确认 {sku} 的最终藏价", 
                        min_value=0.0,
                        value=float(conflict_list[0]['value']),
                        step=1.0,
                        key=f"manual_input_{sku}"
                    )
                with cols[1]:
                    if st.button(f"✓ 确认并重新计算", key=f"confirm_{sku}", type="primary", use_container_width=True):
                        solver.set_manual_price(site, sku, new_price)
                        st.success(f"已确认 {sku} = {new_price:.2f}，系统将基于该值重新推导其他SKU")
                        st.rerun()
                
                st.caption("💡 确认后，系统会自动基于该值重新计算所有能推导的SKU（如基于A的新值重新算出B）")
    
    # 3. 欠定约束显示（如 2D+E=110）
    if constraints:
        st.subheader("🔗 欠定约束关系（无法唯一确定）")
        for cons in constraints:
            with st.container(border=True):
                st.markdown(f"<div class='constraint-box'><strong>订单 {cons['order_id']}</strong>: {cons['equation']}</div>", unsafe_allow_html=True)
                st.caption(f"涉及待定SKU: {', '.join(cons['missing_skus'])}")
        
        if not conflicts:
            st.info("💡 录入只包含上述待定SKU的订单（如单独的D订单），即可解除约束求得确切值")
    
    # 4. 历史订单（带删除）
    if orders:
        st.divider()
        st.subheader(f"📋 {SITES[site]} 历史订单")
        
        _, items_data, _ = solver.get_site_data(site)
        
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
                    st.markdown(f"**{order['total_hidden_price']:.2f}**")
                with col4:
                    confirm_key = f"del_{oid}"
                    if confirm_key not in st.session_state.delete_confirm:
                        st.session_state.delete_confirm[confirm_key] = False
                    
                    if not st.session_state.delete_confirm[confirm_key]:
                        if st.button("🗑️", key=f"btn_{oid}"):
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
