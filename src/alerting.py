"""Step 7: Temporal score smoothing, K-consecutive window alerting, cooldown, and 5s clip saving."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np
import yaml


@dataclass
class AlertEvent:
    """Anomaly alert event metadata ready for MQTT publication."""

    event_id: str
    ts: str
    cam_id: str
    track_id: int
    score: float
    threshold: float
    clip_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert AlertEvent to the required MQTT JSON payload dictionary."""
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "cam_id": self.cam_id,
            "track_id": int(self.track_id),
            "score": round(float(self.score), 6),
            "threshold": round(float(self.threshold), 6),
            "clip_path": self.clip_path,
        }


class AlertManager:
    """Manage temporal score smoothing, K-consecutive threshold checks, cooldown, and clip saving."""

    def __init__(
        self,
        threshold: float,
        clips_dir: Path,
        cam_id: str = "cam_01",
        score_smooth_window: int = 3,
        consecutive_windows_k: int = 3,
        cooldown_seconds: float = 10.0,
        clip_duration_seconds: float = 5.0,
        target_fps: float = 15.0,
        project_root: Path | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.cam_id = cam_id
        self.score_smooth_window = max(1, int(score_smooth_window))
        self.consecutive_windows_k = max(1, int(consecutive_windows_k))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.clip_duration_seconds = max(1.0, float(clip_duration_seconds))
        self.target_fps = max(1.0, float(target_fps))
        self.project_root = project_root

        max_clip_frames = max(1, int(round(self.clip_duration_seconds * self.target_fps)))
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=max_clip_frames)
        self._score_history: dict[int, deque[float]] = {}
        self._consecutive_counts: dict[int, int] = {}
        self._last_alert_ts: dict[int, float] = {}

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        project_root: Path | None = None,
        threshold_override: float | None = None,
    ) -> AlertManager:
        """Instantiate AlertManager from YAML config and calibrated threshold file if available."""
        root = project_root or config_path.resolve().parent.parent
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        alert_cfg = cfg["alerting"]
        threshold = float(alert_cfg["default_threshold"])
        th_file = root / cfg["paths"]["threshold_file"]
        if threshold_override is not None:
            threshold = float(threshold_override)
        elif bool(alert_cfg.get("use_threshold_file", False)) and th_file.exists():
            import json

            with open(th_file, "r", encoding="utf-8") as tf:
                th_data = json.load(tf)
                if "threshold" in th_data:
                    threshold = float(th_data["threshold"])

        return cls(
            threshold=threshold,
            clips_dir=root / cfg["paths"]["clips_dir"],
            cam_id=str(cfg["mqtt"]["cam_id"]),
            score_smooth_window=int(alert_cfg["score_smooth_window"]),
            consecutive_windows_k=int(alert_cfg["consecutive_windows_k"]),
            cooldown_seconds=float(alert_cfg["cooldown_seconds"]),
            clip_duration_seconds=float(alert_cfg["clip_duration_seconds"]),
            target_fps=float(cfg["video"]["target_fps"]),
            project_root=root,
        )

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Append a video frame to the rolling 5-second clip buffer."""
        self._frame_buffer.append(frame_bgr.copy())

    def get_smoothed_score(self, track_id: int) -> float | None:
        """Return the latest temporally smoothed score for `track_id`, or None if unseen."""
        hist = self._score_history.get(track_id)
        if not hist:
            return None
        return float(np.mean(hist))

    def update_score(
        self,
        track_id: int,
        raw_score: float,
        timestamp_sec: float,
    ) -> AlertEvent | None:
        """Update score history for `track_id` and return AlertEvent if triggered."""
        if track_id not in self._score_history:
            self._score_history[track_id] = deque(maxlen=self.score_smooth_window)
            self._consecutive_counts[track_id] = 0

        hist = self._score_history[track_id]
        hist.append(float(raw_score))
        smoothed_score = float(np.mean(hist))

        if smoothed_score > self.threshold:
            self._consecutive_counts[track_id] += 1
        else:
            self._consecutive_counts[track_id] = 0
            return None

        if self._consecutive_counts[track_id] < self.consecutive_windows_k:
            return None

        last_ts = self._last_alert_ts.get(track_id)
        if last_ts is not None and (timestamp_sec - last_ts) < self.cooldown_seconds:
            return None

        self._last_alert_ts[track_id] = float(timestamp_sec)
        self._consecutive_counts[track_id] = 0

        event_id = f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        clip_path = self.save_clip(event_id)
        ts_iso = datetime.now(timezone.utc).isoformat()

        return AlertEvent(
            event_id=event_id,
            ts=ts_iso,
            cam_id=self.cam_id,
            track_id=int(track_id),
            score=smoothed_score,
            threshold=self.threshold,
            clip_path=clip_path,
        )

    def save_clip(self, event_id: str) -> str:
        """Save buffered frames as a short MP4 video clip and return its relative POSIX path."""
        out_file = self.clips_dir / f"{event_id}.mp4"
        frames = list(self._frame_buffer)
        if not frames:
            frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.target_fps,
            (w, h),
        )
        for frm in frames:
            if frm.shape[:2] != (h, w):
                frm = cv2.resize(frm, (w, h))
            writer.write(frm)
        writer.release()

        if self.project_root is not None:
            try:
                return out_file.resolve().relative_to(self.project_root.resolve()).as_posix()
            except ValueError:
                pass
        return out_file.as_posix()


def main() -> None:
    """CLI smoke test for AlertManager."""
    parser = argparse.ArgumentParser(description="Test AlertManager smoothing, K-window, and cooldown.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    manager = AlertManager.from_config(Path(args.config), threshold_override=0.5)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    scores = [0.2, 0.8, 0.9, 1.0, 1.1, 1.2]
    fired: list[AlertEvent] = []
    for i, sc in enumerate(scores):
        manager.push_frame(dummy_frame)
        ev = manager.update_score(track_id=1, raw_score=sc, timestamp_sec=float(i))
        if ev is not None:
            fired.append(ev)
    print(f"[AlertManager] Fired {len(fired)} alert(s). First payload: {fired[0].to_dict() if fired else None}")


if __name__ == "__main__":
    main()
