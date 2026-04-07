from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import SQLITE_DB_PATH


def _connect() -> sqlite3.Connection:
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gmail_id TEXT UNIQUE,
                thread_id TEXT,
                sender TEXT,
                subject TEXT,
                body TEXT,
                snippet TEXT,
                timestamp TEXT,
                internal_date INTEGER,
                predicted_category TEXT,
                confidence REAL,
                priority_score INTEGER,
                priority_band TEXT,
                explanation TEXT,
                extracted_signals TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.commit()


def get_state(key: str) -> Optional[str]:
    with _connect() as conn:
        cur = conn.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def insert_email(record: Dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO emails (
                gmail_id, thread_id, sender, subject, body, snippet, timestamp, internal_date,
                predicted_category, confidence, priority_score, priority_band, explanation,
                extracted_signals, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("gmail_id"),
                record.get("thread_id"),
                record.get("sender"),
                record.get("subject"),
                record.get("body"),
                record.get("snippet"),
                record.get("timestamp"),
                record.get("internal_date"),
                record.get("predicted_category"),
                record.get("confidence"),
                record.get("priority_score"),
                record.get("priority_band"),
                json.dumps(record.get("explanation", [])),
                json.dumps(record.get("extracted_signals", {})),
                record.get("created_at"),
            ),
        )
        conn.commit()


def list_emails(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT * FROM emails ORDER BY internal_date DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cur.fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["explanation"] = json.loads(item.get("explanation") or "[]")
        item["extracted_signals"] = json.loads(item.get("extracted_signals") or "{}")
        results.append(item)
    return results

