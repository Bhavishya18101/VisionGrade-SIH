import sqlite3
from datetime import datetime

DB_NAME = "mandi_offline.db"

def init_db():
    """Initializes SQLite database tables for lots and grading logs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Farmers & Slots Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lot_records (
            lot_id TEXT PRIMARY KEY,
            farmer_name TEXT NOT NULL,
            farmer_id TEXT NOT NULL,
            mandi_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_count INTEGER,
            grade_a_count INTEGER,
            grade_b_count INTEGER,
            grade_c_count INTEGER,
            reject_count INTEGER,
            total_weight_kg REAL,
            estimated_payout REAL,
            sync_status TEXT DEFAULT 'QUEUED'
        )
    ''')
    
    conn.commit()
    conn.close()

def save_lot_record(lot_id, farmer_name, farmer_id, mandi_name, total_count, grade_a, grade_b, grade_c, reject, weight_kg, payout):
    """Saves a completed lot assessment to the local SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO lot_records 
        (lot_id, farmer_name, farmer_id, mandi_name, total_count, grade_a_count, grade_b_count, grade_c_count, reject_count, total_weight_kg, estimated_payout)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (lot_id, farmer_name, farmer_id, mandi_name, total_count, grade_a, grade_b, grade_c, reject, weight_kg, payout))
    
    conn.commit()
    conn.close()

def fetch_recent_lots(limit=10):
    """Retrieves recent lot assessment records."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT lot_id, farmer_name, total_weight_kg, grade_a_count, grade_b_count, grade_c_count, reject_count, estimated_payout, timestamp 
        FROM lot_records 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    records = cursor.fetchall()
    conn.close()
    return records

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
    