import sqlite3
import pandas as pd
from datetime import datetime

def init_db():
    conn = sqlite3.connect('diabepredict.db')
    c = conn.cursor()
    
    # Table utilisateurs
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT,
                  created_at TIMESTAMP)''')
    
    # Table historique des analyses
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  analysis_type TEXT,
                  parameters TEXT,
                  result TEXT,
                  probability REAL,
                  created_at TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

def add_user(username, password, email):
    conn = sqlite3.connect('diabepredict.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, email, created_at) VALUES (?, ?, ?, ?)",
                  (username, password, email, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Utilisateur existe déjà
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('diabepredict.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def add_analysis(user_id, analysis_type, parameters, result, probability):
    conn = sqlite3.connect('diabepredict.db')
    c = conn.cursor()
    c.execute("INSERT INTO analyses (user_id, analysis_type, parameters, result, probability, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, analysis_type, str(parameters), result, probability, datetime.now()))
    conn.commit()
    conn.close()

def get_user_analyses(user_id):
    conn = sqlite3.connect('diabepredict.db')
    df = pd.read_sql("SELECT * FROM analyses WHERE user_id=? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close()
    return df

# Initialiser la DB au premier import
init_db()