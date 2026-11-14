"""Step 7: Temporal score smoothing, K-consecutive window alerting, cooldown, and clip saving."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from shoplift_iot.alerting.clip_recorder import ClipRecorder
from shoplift_iot.config import AppConfig, load_app_config
from shoplift_iot.contracts import AlertEvent
from shoplift_iot.model.bundle import ModelBundle


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
        pre_event_seconds: float | None = None,
        post_event_seconds: float = 2.0,
        target_fps: float = 15.0,
        project_root: Path | None = None,
        clip_recorder: ClipRecorder | None = None,
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

        pre_sec = (
            float(pre_event_seconds)
            if pre_event_seconds is not None
            else max(1.0, self.clip_duration_seconds - float(post_event_seconds))
        )
        self.clip_recorder = clip_recorder or ClipRecorder(
            clips_dir=self.clips_dir,
            target_fps=self.target_fps,
            pre_event_seconds=pre_sec,
            post_event_seconds=float(post_event_seconds),
            project_root=self.project_root,
        )

        self._score_history: dict[int, deque[float]] = {}
        self._consecutive_counts: dict[int, int] = {}
        self._last_alert_ts: dict[int, float] = {}

    @classmethod
    def from_config(
        cls,
        config_or_path: Path | str | AppConfig | Mapping[str, Any],
        project_root: Path | None = None,
        threshold_override: float | None = None,
        bundle: ModelBundle | None = None,
    ) -> AlertManager:
        """Instantiate AlertManager from `AppConfig` and `ModelBundle` threshold."""
        app_cfg = load_app_config(config_or_path, project_root=project_root)
        root = app_cfg.project_root
        alert_cfg = app_cfg.alerting
        pipe_cfg = app_cfg.pipeline

        loaded_bundle = bundle or ModelBundle.load(app_cfg.bundle_dir, cfg=app_cfg, project_root=root)
        threshold = float(loaded_bundle.threshold)
        if threshold_override is not None:
            threshold = float(threshold_override)

        return cls(
            threshold=threshold,
            clips_dir=app_cfg.resolve_path(app_cfg.paths["clips_dir"]),
            cam_id=str(app_cfg.mqtt["cam_id"]),
            score_smooth_window=int(alert_cfg["score_smooth_window"]),
            consecutive_windows_k=int(alert_cfg["consecutive_windows_k"]),
            cooldown_seconds=float(alert_cfg["cooldown_seconds"]),
            clip_duration_seconds=float(alert_cfg["clip_duration_seconds"]),
            pre_event_seconds=float(pipe_cfg.get("pre_event_seconds", 3.0)),
            post_event_seconds=float(pipe_cfg.get("post_event_seconds", 2.0)),
            target_fps=float(app_cfg.video["target_fps"]),
            project_root=root,
        )

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Append a video frame to the pre-event ring buffer and active post-event clips."""
        self.clip_recorder.push_frame(frame_bgr)

    def flush_clips(self) -> None:
        """Flush all pending post-event video clips to disk."""
        self.clip_recorder.flush_all()

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

        event_id = str(uuid.uuid4())
        clip_path = self.save_clip(event_id)

        return AlertEvent.create(
            event_id=event_id,
            cam_id=self.cam_id,
            track_id=int(track_id),
            score=smoothed_score,
            threshold=self.threshold,
            clip_path=clip_path,
        )

    def save_clip(self, event_id: str) -> str:
        """Trigger pre/post-event MP4 clip recording for `event_id` and return its relative POSIX path."""
        return self.clip_recorder.trigger_clip(event_id)


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
    manager.flush_clips()
    print(f"[AlertManager] Fired {len(fired)} alert(s). First payload: {fired[0].to_dict() if fired else None}")


if __name__ == "__main__":
    main()
