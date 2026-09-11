"""
Minimal SQLite helper for FloodLens.

We avoid heavy ORM dependencies and use the stdlib sqlite3 module. Two tables:

    predictions      – history of predictions made through the UI/API
    model_metadata   – key/value store for model metadata snapshots
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import current_app, g

from config import Config


def _db_path() -> Path:
    url = Config.DATABASE_URL
    if url.startswith("sqlite:///"):
        return Path(url.replace("sqlite:///", ""))
    return Config.INSTANCE_DIR / "floodlens.db"


def get_db() -> sqlite3.Connection:
    if "_db" not in g:
        p = _db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(p)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g._db = conn
    return g._db


def close_db(_e=None):
    db = g.pop("_db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    from flask import current_app
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            risk_index INTEGER NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            important_factors TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'web',
            model_version TEXT
        );
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT,
            stored_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
    """)
    db.commit()


def record_prediction(result: Dict[str, Any], source: str = "web") -> int:
    db = get_db()
    cur = db.execute(
        """INSERT INTO predictions
           (created_at, input_summary, risk_level, risk_index, confidence,
            probabilities, important_factors, source, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            json.dumps(result.get("input_summary", {})),
            result.get("risk_level"),
            int(result.get("risk_index", -1)),
            float(result.get("confidence", 0.0)),
            json.dumps(result.get("probabilities", {})),
            json.dumps(result.get("important_factors", [])),
            source,
            result.get("model_version"),
        ),
    )
    db.commit()
    return cur.lastrowid


def recent_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["input_summary"] = json.loads(d["input_summary"])
        d["probabilities"] = json.loads(d["probabilities"])
        d["important_factors"] = json.loads(d["important_factors"])
        out.append(d)
    return out


def count_predictions() -> int:
    db = get_db()
    return int(db.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"])


def risk_counts() -> Dict[str, int]:
    db = get_db()
    rows = db.execute(
        "SELECT risk_level, COUNT(*) AS c FROM predictions GROUP BY risk_level"
    ).fetchall()
    return {r["risk_level"]: int(r["c"]) for r in rows}


def clear_history() -> None:
    db = get_db()
    db.execute("DELETE FROM predictions")
    db.commit()


def store_model_snapshot(metadata: Dict[str, Any]) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO model_metadata (model_version, stored_at, payload) VALUES (?, ?, ?)",
        (
            metadata.get("model_version"),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            json.dumps(metadata, default=str),
        ),
    )
    db.commit()
