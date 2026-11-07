"""SQLite persistence layer for anomaly events and human review labels."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

VALID_DECISIONS = ("pending", "confirmed", "ignored")


class EventDatabase:
    """Manage SQLite storage for anomaly alert events and human review decisions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    cam_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    clip_path TEXT NOT NULL,
                    decision TEXT NOT NULL DEFAULT 'pending',
                    decided_at TEXT
                )
                """
            )
            conn.commit()

    def upsert_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert a newly received MQTT alert event into SQLite."""
        event_id = str(payload["event_id"])
        ts = str(payload["ts"])
        cam_id = str(payload["cam_id"])
        track_id = int(payload["track_id"])
        score = float(payload["score"])
        threshold = float(payload["threshold"])
        clip_path = str(payload["clip_path"])

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (event_id, ts, cam_id, track_id, score, threshold, clip_path, decision, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL)
                ON CONFLICT(event_id) DO UPDATE SET
                    ts = excluded.ts,
                    cam_id = excluded.cam_id,
                    track_id = excluded.track_id,
                    score = excluded.score,
                    threshold = excluded.threshold,
                    clip_path = excluded.clip_path
                """,
                (event_id, ts, cam_id, track_id, score, threshold, clip_path),
            )
            conn.commit()
        res = self.get_event(event_id)
        assert res is not None
        return res

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Fetch a single event by `event_id`."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """List newest alert events ordered by timestamp descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_decision(self, event_id: str, decision: str) -> dict[str, Any] | None:
        """Update human review decision ('confirmed' for Xác nhận, 'ignored' for Bỏ qua)."""
        if decision not in ("confirmed", "ignored"):
            raise ValueError(f"Decision must be 'confirmed' or 'ignored', got '{decision}'")
        decided_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE events
                SET decision = ?, decided_at = ?
                WHERE event_id = ?
                """,
                (decision, decided_at, event_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_event(event_id)

