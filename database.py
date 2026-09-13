"""SQLite persistence for VisionGrade's offline-first mandi ledger.

The database is intentionally local: a grading result is committed on the
device first and can be synchronized with a remote service later.
"""

import sqlite3
import uuid
import random
from datetime import datetime

DB_NAME = 'visiongrade.db'

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def setup_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Farmers (
            farmer_token TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            address TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Lots (
            batch_id TEXT PRIMARY KEY,
            farmer_token TEXT,
            grade_a INTEGER,
            grade_b INTEGER,
            grade_c INTEGER,
            reject INTEGER,
            quantity REAL,
            payout REAL,
            date TEXT,
            FOREIGN KEY (farmer_token) REFERENCES Farmers(farmer_token)
        )
    ''')

    conn.commit()
    conn.close()

def seed_sample_data():
    """Injects sample demo data for the hackathon pitch if the DB is empty."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM Farmers")
    if c.fetchone()[0] == 0:
        # 1. Insert Mock Farmers
        farmers = [
            ("VG-FARM-1001", "Ramesh Kumar", "pass123", "8708910185", "Sonipat, Haryana"),
            ("VG-FARM-1002", "Suresh Singh", "pass123", "9999999999", "Karnal, Haryana")
        ]
        c.executemany("INSERT INTO Farmers VALUES (?, ?, ?, ?, ?)", farmers)
        
        # 2. Insert Mock Batches for Today
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lots = [
            ("BATCH-001", "VG-FARM-1001", 120, 45, 10, 5, 180, 5525.0, today),
            ("BATCH-002", "VG-FARM-1002", 80, 60, 20, 10, 170, 4900.0, today),
            ("BATCH-003", "VG-FARM-1001", 150, 30, 5, 2, 187, 6800.0, today)
        ]
        c.executemany("INSERT INTO Lots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", lots)
        conn.commit()
        print("✅ Sample demo data injected successfully.")
        
    conn.close()

def add_new_farmer(farmer_token, name, password, phone_number, address):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Farmers (farmer_token, name, password, phone_number, address)
            VALUES (?, ?, ?, ?, ?)
        ''', (farmer_token, name, password, phone_number, address))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def insert_lot_data(batch_id, farmer_token, grade_a, grade_b, grade_c, reject, quantity, payout, date):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Lots (batch_id, farmer_token, grade_a, grade_b, grade_c, reject, quantity, payout, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (batch_id, farmer_token, grade_a, grade_b, grade_c, reject, quantity, payout, date))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()