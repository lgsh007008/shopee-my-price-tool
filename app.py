import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from collections import defaultdict

st.set_page_config(page_title="SKU藏价求解器", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 3rem !important;}
    .conflict-box {background-color: #f8d7da; border: 2px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 8px;}
    .resolved-box {background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 5px 0;}
    .site-badge {font-size: 1.2rem; font-weight: bold; padding: 5px 15px; border-radius: 20px; background-color: #e9ecef;}
</style>
""", unsafe_allow_html=True)

# 站点映射
SITES = {
    'MX': '🇲🇽 墨西哥 (Mexico)',
    'TH': '🇹🇭 泰国 (Thailand)', 
    'PH': '🇵🇭 菲律宾 (Philippines)'
}

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

class ConflictSolver:
    def __init__(self):
        self.supabase = init_supabase()
        
    def init_db(self):
        """初始化数据库表（包含手动确认表）"""
        # 这里假设之前的orders和order_items表已创建
        # 新增manual_prices表保存用户确认值
        try:
            self.supabase.table('manual_prices').select("*").limit(1).execute()
        except:
            # 表不存在的话需要在Supabase SQL Editor执行：
            # CREATE TABLE manual_prices (
            #     site TEXT NOT NULL,
            #     sku TEXT NOT NULL,
            #     manual_price REAL NOT NULL,
            #     note TEXT,
            #     confirmed_at TIMESTAMP DEFAULT NOW(),
            #     PRIMARY KEY (site, sku)
            # );
            pass
    
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
    
    def get_manual_prices(self, site: str):
        """获取用户手动确认的价格"""
        data = self.supabase.table('manual_prices').select("*").eq('site', site).execute().data or []
        return {d['sku']: d['manual_price'] for d in data}
    
    def set_manual_price(self, site: str, sku: str, price: float, note: str = ""):
        """保存用户手动确认的价格"""
        try:
            self.supabase.table('manual_prices').upsert({
                "site": site,
                "sku": sku,
                "manual_price": price,
                "note": note,
                "confirmed_at": datetime.now().isoformat()
            }).execute()
            return True
        except Exception as e:
            st.error(f"保存手动价格失败: {e}")
            return False
    
    def clear_manual_price(self, site: str, sku: str):
        """清除手动确认的价格"""
        try:
            self.supabase.table('manual_prices').delete().eq('site', site).eq('sku', sku).execute()
            return True
        except:
            return False
    
    def detect_conflicts(self, site: str):
        """
        检测逻辑：
        1. 先检查是否有手动确认值（作为已知）
        2. 尝试推导所有可能的值
        3. 发现同一个SKU有多个不同推导值时，标记为矛盾
        """
        orders, items = self.get_site_data(site)
        manual_prices = self.get_manual_prices(site)
        
        if not orders:
            return {}, [], [], orders, manual_prices  # determined, conflicts, underdetermined, orders, manual
        
        # 构建订单-商品映射
        order_items_map = defaultdict(list)
        for it in items:
            order_items_map[it['order_id']].append(it)
        
        all_skus = sorted(list(set([it['sku'] for it in items])))
        
        # 为每个SKU收集可能的推导值
        sku_possible_values = defaultdict(list)  # SKU -> [(order_id, derived_value, equation)]
        
        # 先处理可以独立计算的SKU（出现在只有它未知的订单中）
        # 逐步迭代直到没有新值可推导
        determined = dict(manual_prices)  # 从手动确认值开始
        changed = True
        iterations = 0
        
        while changed and iterations < 10:
            changed = False
            iterations += 1
            
            for order in orders:
                oid = order['order_id']
                total = order['total_hidden_price']
                o_items = order_items_map[oid]
                
                # 已知部分
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
                
                if len(unknown_items) == 1 and remaining >= 0:
                    # 只有一个未知数，可以直接算出
                    sku, qty = unknown_items[0]
                    derived_value = remaining / qty
                    
                    if sku not in determined:
                        determined[sku] = derived_value
                        changed = True
                    elif abs(determined[sku] - derived_value) > 0.01:  # 允许0.01误差
                        # 发现矛盾！记录这个推导值
                        sku_possible_values[sku].append({
                            'order_id': oid,
                            'value': derived_value,
                            'equation': f"{qty}×{sku} = {remaining} (订单{oid})",
                            'context': [it['sku'] for it in o_items]
                        })
                elif len(unknown_items) == 0:
                    # 验证一致性
                    if abs(remaining) > 0.01:
                        # 矛盾：已知值加起来不等于总藏价
                        pass  # 数据错误，但先忽略
        
        # 收集矛盾
        conflicts = {}
        for sku, values in sku_possible_values.items():
            if sku not in determined:  # 如果有确定的manual值，不视为矛盾
                # 去重，保留不同的值
                unique_values = []
                seen = set()
                for v in values:
                    key = round(v['value'], 2)
                    if key not in seen:
                        seen.add(key)
                        unique_values.append(v)
                
                if len(unique_values) > 1:
                    conflicts[sku] = unique_values
        
        # 欠定：有SKU没被确定且没有矛盾（即完全无法推导）
        underdetermined = []
        for sku in all_skus:
            if sku not in determined and sku not in conflicts:
                # 检查是否真的无法推导
                underdetermined.append(sku)
        
        return determined, conflicts, underdetermined, orders, manual_prices

# ============ 初始化 ============
try:
    solver = ConflictSolver()
    solver.init_db()
except Exception as e:
    st.error(f"连接失败: {e}")
    st.stop()

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

# ============ 界面 ============
st.title("📦 SKU 藏价求解器")

# 固定三站点选择（横向排列）
cols = st.columns(3)
site_keys = ['MX', 'TH', 'PH']
for i, (key, label) in enumerate(SITES.items()):
    with cols[i]:
        if st.button(label, key=f"site_{key}", 
                    type="primary" if st.session_state.current_site == key else "secondary",
                    use_container_width=True):
            st.session_state.current_site = key
            st.rerun()

site = st.session_state.current_site

st.markdown(f"<div style='text-align: center; margin: 10px 0;'>当前操作站点：<span class='site-badge'>{SITES[site]}</span></div>", unsafe_allow_html=True)

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
    determined, conflicts, underdetermined, orders, manual_prices = solver.detect_conflicts(site)
    
    # 统计
    c1, c2, c3 = st.columns(3)
    c1.metric("已确定SKU", len(determined))
    c2.metric("矛盾待处理", len(conflicts))
    c3.metric("待定SKU", len(underdetermined))
    
    st.divider()
    
    # 1. 显示已确定（含手动确认）
    if determined:
        st.subheader("✅ 已确定藏价")
        
        # 分离手动确认和自动推导
        manual_items = {k: v for k, v in determined.items() if k in manual_prices}
        auto_items = {k: v for k, v in determined.items() if k not in manual_prices}
        
        if manual_items:
            st.markdown("**📝 手动确认值**")
            for sku, price in manual_items.items():
                cols = st.columns([3, 2, 1])
                with cols[0]:
                    st.markdown(f"<div class='resolved-box'><strong>{sku}</strong>: {price:.2f}</div>", unsafe_allow_html=True)
                with cols[2]:
                    if st.button("重置", key=f"reset_{sku}"):
                        solver.clear_manual_price(site, sku)
                        st.rerun()
        
        if auto_items:
            st.markdown("**🤖 自动推导值**")
            df = pd.DataFrame(list(auto_items.items()), columns=['SKU', '藏价'])
            df['藏价'] = df['藏价'].apply(lambda x: f"{x:.2f}")
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 2. 显示矛盾（核心功能）
    if conflicts:
        st.subheader("⚠️ 发现矛盾，需手动确认")
        st.error("以下SKU在不同订单中推导出了不同价格，请确认最终值")
        
        for sku, conflict_list in conflicts.items():
            with st.container(border=True):
                st.markdown(f"**SKU: {sku}**")
                
                # 显示各个推导来源
                for i, conflict in enumerate(conflict_list):
                    st.markdown(f"- 订单 **{conflict['order_id']}**: 推导值 = **{conflict['value']:.2f}** ({conflict['equation']})")
                
                # 手动输入确认
                cols = st.columns([3, 2])
                with cols[0]:
                    manual_val = st.number_input(
                        f"确认 {sku} 的最终藏价", 
                        min_value=0.0, 
                        value=float(conflict_list[0]['value']),  # 默认第一个
                        step=1.0,
                        key=f"manual_{sku}"
                    )
                with cols[1]:
                    note = st.text_input("备注（可选）", placeholder="为什么这么定", key=f"note_{sku}")
                
                if st.button(f"✓ 确认 {sku} = {manual_val:.2f}", key=f"confirm_{sku}", type="primary"):
                    solver.set_manual_price(site, sku, manual_val, note)
                    st.success(f"已确认 {sku} = {manual_val:.2f}")
                    st.rerun()
                
                st.divider()
    
    # 3. 显示欠定（无法推导的）
    if underdetermined:
        st.subheader("❓ 待定SKU（数据不足）")
        st.info(f"以下SKU暂时无法推导，需要录入包含它们的订单：{', '.join(underdetermined)}")
    
    if not determined and not conflicts and not underdetermined:
        st.info("录入第一个订单后开始计算")
    
    # 4. 历史订单
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
