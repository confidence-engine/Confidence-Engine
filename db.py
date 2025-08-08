import sqlite3
import json
from datetime import datetime, timezone

_DB = "tracer.db"

def _conn():
    return sqlite3.connect(_DB)

def init_db():
    with _conn() as c, open("schema.sql", "r", encoding="utf-8") as f:
        c.executescript(f.read())

def save_run(payload: dict) -> int:
    fields = [
        "ts_utc","symbol","bars_lookback_min","headlines_count","raw_headlines",
        "btc_filtered_headlines","finbert_score","llm_score","raw_narrative","decayed_narrative",
        "price_score","volume_z","divergence","divergence_threshold","confidence",
        "action","reason","summary","detail"
    ]
    row = [payload[k] for k in fields]
    with _conn() as c:
        cur = c.execute("""
          INSERT INTO runs (
            ts_utc, symbol, bars_lookback_min, headlines_count, raw_headlines,
            btc_filtered_headlines, finbert_score, llm_score, raw_narrative, decayed_narrative,
            price_score, volume_z, divergence, divergence_threshold, confidence,
            action, reason, summary, detail
          ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, row)
        return cur.lastrowid

def list_runs(limit: int = 10):
    with _conn() as c:
        cur = c.execute("SELECT id, ts_utc, symbol, action, reason, summary FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        return cur.fetchall()

def get_run(run_id: int) -> dict:
    with _conn() as c:
        cur = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        if not row: return {}
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
