import sqlite3
import pandas as pd
import os

DB_PATH = "data/jobmatcher.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_jobs_from_df(df):
    conn = sqlite3.connect(DB_PATH)
    df[["text"]].to_sql("jobs", conn, if_exists="append", index=False)
    conn.close()

def insert_single_job(text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO jobs (text) VALUES (?)", (text,))
    conn.commit()
    conn.close()

def load_jobs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    return df
