"""SQLite-backed tool-result cache with 24h TTL. Keyed by (tool_name, args)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from app.config import CACHE_PATH, CACHE_TTL_SECONDS, DEBUG
import sys

_conn = sqlite3.connect(CACHE_PATH, check_same_thread=False)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS cache ("
    "  key TEXT PRIMARY KEY,"
    "  value TEXT NOT NULL,"
    "  created_at REAL NOT NULL"
    ")"
)
_conn.commit()


def _make_key(tool_name: str, args: dict) -> str:
    """Stable hash of tool + args. sort_keys ensures {a:1,b:2} == {b:2,a:1}."""
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def get(tool_name: str, args: dict) -> str | None:
    """Return cached result if present and not expired, else None."""
    key = _make_key(tool_name, args)
    row = _conn.execute(
        "SELECT value, created_at FROM cache WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    value, created_at = row
    if time.time() - created_at > CACHE_TTL_SECONDS:
        _conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        _conn.commit()
        return None
    if DEBUG:
        print(f"[CACHE HIT] {tool_name}({args})", file=sys.stderr)
    return value


def set(tool_name: str, args: dict, value: str) -> None:
    """Store a result. Caller must NOT cache error payloads."""
    key = _make_key(tool_name, args)
    _conn.execute(
        "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
        (key, value, time.time()),
    )
    _conn.commit()