import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine, text

st.set_page_config(page_title="跨境 SKU 藏价系统", layout="wide")

# --- 1. 连接 Supabase 数据库 ---
# 注意：Secrets 中必须使用 port 6543 的连接池地址，否则可能报错
conn = st.connection("db", type="sql")

# 初始化数据库表
def init_db():
    try:
        with conn.session as s:
            s.execute(text("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    site TEXT NOT NULL,
                    items TEXT NOT NULL,
                    total_price FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            s.commit()
    except Exception as e:
        st.error(f"数据库初始化失败: {e}")

# 页面加载时尝试初始化
init_db()

# --- 2. 数据库读写操作 ---
def get_data():
    try:
        return conn.query("SELECT * FROM orders ORDER BY created_at DESC;", ttl=0)
    except Exception:
        return pd.DataFrame(columns=['order_id', 'site', 'items', 'total_price', 'created_at'])

def save_order(order_id, site, items_dict, total_price):
    items_json = json.dumps(items_dict)
    with conn.session as s:
        s.execute(
            text("INSERT INTO orders (order_id, site, items, total_price, created_at) VALUES (:oid, :site, :items, :price, :time)"),
            params={
                "oid": order_id, 
                "site": site, 
                "items": items_json, 
                "price": total_price,
                "time": datetime.now()
            }
        )
        s.commit()

# --- 3. 核心算法：最小二乘法 (计算加权平均 + 安全边际) ---
def solve_prices_smart(df, safety_margin=0.0):
    if df.empty:
        return {}

    all_skus = set()
    rows = []     
    results = []  
    
    parsed_orders = []
    for _, row in df.iterrows():
        try:
            item_str = str(row['items']).replace("'", '"')
            items = json.loads(item_str)
            parsed_orders.append({'items': items, 'price': float(row['total_price'])})
            all_skus.update(items.keys())
        except:
            continue
