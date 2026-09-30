"""
Phase 8: Canonical Inventory Ledger
SQLite-backed single source of truth for stock levels and transactions.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/ledger.db"

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_ledger():
    """Initializes inventory stock table and transaction ledger."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Current state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory_levels (
            store_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            on_hand_qty REAL NOT NULL DEFAULT 0.0,
            reserved_qty REAL NOT NULL DEFAULT 0.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (store_id, product_id)
        )
    """)
    
    # Immutable transaction log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transaction_log (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            store_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            delta_qty REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def record_transaction(source: str, store_id: str, product_id: str, delta_qty: float, notes: str = ""):
    """Updates inventory level and writes immutable log entry."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Upsert stock level
    cursor.execute("""
        INSERT INTO inventory_levels (store_id, product_id, on_hand_qty, last_updated)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(store_id, product_id) DO UPDATE SET
            on_hand_qty = on_hand_qty + excluded.on_hand_qty,
            last_updated = CURRENT_TIMESTAMP
    """, (store_id, product_id, delta_qty))
    
    # Log transaction
    cursor.execute("""
        INSERT INTO transaction_log (source, store_id, product_id, delta_qty, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (source, store_id, product_id, delta_qty, notes))
    
    conn.commit()
    
    # Fetch updated level
    cursor.execute("SELECT on_hand_qty FROM inventory_levels WHERE store_id = ? AND product_id = ?", (store_id, product_id))
    new_qty = cursor.fetchone()['on_hand_qty']
    conn.close()
    return new_qty

def get_stock(store_id: str, product_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT on_hand_qty FROM inventory_levels WHERE store_id = ? AND product_id = ?", (store_id, product_id))
    row = cursor.fetchone()
    conn.close()
    return row['on_hand_qty'] if row else 0.0
