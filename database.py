"""
database.py – SQLite persistence for PhishGuard AI
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            url       TEXT    NOT NULL,
            result    TEXT    NOT NULL,
            risk_score REAL   NOT NULL,
            scanned_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# initialise on import
init_db()


def save_scan(url: str, result: str, risk_score: float):
    conn = _connect()
    conn.execute(
        "INSERT INTO scans (url, result, risk_score) VALUES (?, ?, ?)",
        (url, result, round(risk_score, 4)),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 200):
    conn = _connect()
    rows = conn.execute(
        "SELECT url, result, risk_score, scanned_at "
        "FROM scans ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = _connect()
    row = conn.execute("""
        SELECT
            COUNT(*)                                  AS total,
            SUM(result='Phishing')                    AS phishing,
            SUM(result='Legitimate')                  AS legitimate,
            SUM(result='Suspicious')                  AS suspicious,
            ROUND(AVG(risk_score)*100, 1)             AS avg_risk,
            ROUND(MAX(risk_score)*100, 1)             AS max_risk
        FROM scans
    """).fetchone()
    conn.close()
    return dict(row) if row else {}


def delete_all():
    conn = _connect()
    conn.execute("DELETE FROM scans")
    conn.commit()
    conn.close()
