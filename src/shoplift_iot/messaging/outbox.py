"""SQLite Outbox for reliable QoS-1 MQTT alert event publishing with automatic retry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from shoplift_iot.contracts import AlertEvent


class SQLiteOutbox:
    """Thread-safe SQLite outbox table storing pending/sent/failed MQTT alert payloads."""

    def __init__(self, db_path: Path | str, max_retries: int = 10) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max(1, int(max_retries))
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox_events (
                    event_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def enqueue(self, event: AlertEvent | dict[str, Any]) -> str:
        """Persist an `AlertEvent` in the outbox with status `'pending'`."""
        payload = event.to_dict() if isinstance(event, AlertEvent) else dict(event)
        event_id = str(payload["event_id"])
        now_iso = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, ensure_ascii=False)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO outbox_events (event_id, payload_json, status, attempts, last_error, created_at, updated_at)
                VALUES (?, ?, 'pending', 0, '', ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (event_id, payload_json, now_iso, now_iso),
            )
            conn.commit()
        return event_id

    def fetch_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return up to `limit` pending outbox entries eligible for publishing."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, payload_json, status, attempts, last_error, created_at, updated_at
                FROM outbox_events
                WHERE status = 'pending' AND attempts < ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (self.max_retries, int(limit)),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item["payload_json"])
            results.append(item)
        return results

    def mark_sent(self, event_id: str) -> None:
        """Mark `event_id` as successfully published over MQTT."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE outbox_events
                SET status = 'sent', attempts = attempts + 1, last_error = '', updated_at = ?
                WHERE event_id = ?
                """,
                (now_iso, event_id),
            )
            conn.commit()

    def mark_retry(self, event_id: str, error_msg: str = "") -> None:
        """Increment retry counter for `event_id`, marking as `'failed'` if `max_retries` reached."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE outbox_events
                SET attempts = attempts + 1,
                    status = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END,
                    last_error = ?,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (self.max_retries, str(error_msg), now_iso, event_id),
            )
            conn.commit()

    def get_entry(self, event_id: str) -> dict[str, Any] | None:
        """Return outbox row for `event_id`."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_id, payload_json, status, attempts, last_error, created_at, updated_at
                FROM outbox_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            item = dict(row)
            item["payload"] = json.loads(item["payload_json"])
            return item

    def count_by_status(self, status: str = "pending") -> int:
        """Return count of outbox rows with the given `status`."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM outbox_events WHERE status = ?",
                (status,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
